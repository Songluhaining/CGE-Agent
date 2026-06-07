"""Chain-of-Thought with Self-Consistency baseline (Wang et al. 2022).

Reproduces Wang et al. 2022, "Self-Consistency Improves Chain of Thought
Reasoning in Language Models" (ICLR 2023), on the six benchmarks.

Implementation choices:
  - Prompts are imported from cot_baseline_evaluation.py — same few-shot
    exemplars per dataset (Wei et al. for GSM8K, Minerva for MATH,
    self-constructed for HotpotQA/DROP, zero-shot for HumanEval/MBPP).
  - N=5 samples per problem at temperature=0.7 (Wang et al. defaults).
  - Per-domain majority voting on the extracted final answer:
        GSM8K     -> normalized number string
        MATH      -> last \boxed{...} contents
        HotpotQA  -> SQuAD-normalized answer
        DROP      -> SQuAD-normalized answer
        HumanEval -> sanitized source-code string equality
        MBPP      -> sanitized source-code string equality
  - Tie-break: stable (first-occurrence wins).
  - The prediction returned for scoring is the FIRST sample whose vote-key
    equals the winning key (so we score a real model output, not an
    aggregate that the model never produced).

Run:
    python examples/cot_sc_baseline_evaluation.py --dataset gsm8k
    python examples/cot_sc_baseline_evaluation.py --dataset math --concurrency 5
    python examples/cot_sc_baseline_evaluation.py --dataset hotpotqa --limit 50
    python examples/cot_sc_baseline_evaluation.py --dataset humaneval
    python examples/cot_sc_baseline_evaluation.py --dataset mbpp
    python examples/cot_sc_baseline_evaluation.py --dataset drop

Per-sample predictions and the aggregated score are written to
    examples/output/cot_sc_baseline/<dataset>_<model>.json
"""
import argparse
import asyncio
import json
import os
import re
import string
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv

from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import (
    AFlowDROP,
    AFlowGSM8K,
    AFlowHotPotQA,
    AFlowHumanEval,
    AFlowMATH,
    AFlowMBPP,
)

# Reuse the CoT prompts and per-domain postprocessors from the canonical
# CoT baseline so CoT and CoT-SC differ ONLY in sampling + voting.
from cot_baseline_evaluation import (  # noqa: E402
    _gsm8k_cot_prompt,
    _math_cot_prompt,
    _hotpotqa_cot_prompt,
    _drop_cot_prompt,
    _humaneval_cot_prompt,
    _mbpp_cot_prompt,
    _postprocess_gsm8k,
    _postprocess_math,
    _postprocess_hotpotqa,
    _postprocess_drop,
    _postprocess_code,
)


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "datasets"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "cot_sc_baseline"


# ------------------------------------------------------------------ LLM --

# Locally redefined so we can pass a non-zero temperature; the LLM in
# cot_baseline_evaluation.py hardcodes temperature=0.0 inside its config.
def _make_llm(model: str, temperature: float, max_tokens: int = 2048) -> OpenAILLM:
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm_config = OpenAILLMConfig(
        model=model,
        openai_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=600,
    )
    return OpenAILLM(config=llm_config)


async def _call_llm(llm: OpenAILLM, prompt: str, system: str = None) -> str:
    sys_msgs = [system] if system else None
    messages = llm.formulate_messages(prompts=[prompt], system_messages=sys_msgs)[0]
    out = await llm.single_generate_async(messages=messages)
    return out if isinstance(out, str) else str(out)


# =================================================================== #
# ===================== Per-domain vote keys ======================== #
# =================================================================== #

_PUNCT = set(string.punctuation)


def _normalize_qa(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _vote_key_gsm8k(processed: str) -> str:
    """`processed` is already the bare number string (or empty)."""
    s = (processed or "").strip()
    # Drop trailing zeros / unify formatting: "12.0" and "12" should vote together
    try:
        v = float(s.replace(",", ""))
        if v == int(v):
            return str(int(v))
        return repr(v)
    except (ValueError, AttributeError):
        return s


def _vote_key_math(processed: str) -> str:
    """Use the LAST \\boxed{...} contents; if absent, fall back to last line."""
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    matches = re.findall(pattern, processed or "", re.DOTALL)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in (processed or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _vote_key_qa(processed: str) -> str:
    return _normalize_qa(processed or "")


def _vote_key_code(processed: str) -> str:
    """Strip markdown fences + leading/trailing whitespace before equality."""
    s = (processed or "").strip()
    s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _majority(keys: List[str]) -> Tuple[str, int]:
    """Return (winning_key, vote_count). Stable: ties broken by first-occurrence."""
    counter = Counter()
    for k in keys:
        counter[k] += 1
    if not counter:
        return "", 0
    winner, count = counter.most_common(1)[0]
    return winner, count


# =================================================================== #
# =========================== Scoring =============================== #
# =================================================================== #


def _score_hotpotqa(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("f1", 0.0))


def _score_math(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("solve_rate", m.get("f1", 0.0)))


def _score_gsm8k(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("solve_rate", 0.0))


def _score_drop(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("f1", 0.0))


def _score_humaneval(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("pass@1", 0.0))


def _score_mbpp(bench, pred, ex):
    m = bench.evaluate(pred, bench._get_label(ex))
    return m, float(m.get("pass@1", 0.0))


DATASETS: Dict[str, Dict[str, Any]] = {
    "hotpotqa":  {"cls": AFlowHotPotQA,  "prompt": _hotpotqa_cot_prompt,  "post": _postprocess_hotpotqa, "vote": _vote_key_qa,    "score": _score_hotpotqa,  "metric": "f1"},
    "math":      {"cls": AFlowMATH,      "prompt": _math_cot_prompt,      "post": _postprocess_math,     "vote": _vote_key_math,  "score": _score_math,      "metric": "solve_rate"},
    "gsm8k":     {"cls": AFlowGSM8K,     "prompt": _gsm8k_cot_prompt,     "post": _postprocess_gsm8k,    "vote": _vote_key_gsm8k, "score": _score_gsm8k,     "metric": "solve_rate"},
    "drop":      {"cls": AFlowDROP,      "prompt": _drop_cot_prompt,      "post": _postprocess_drop,     "vote": _vote_key_qa,    "score": _score_drop,      "metric": "f1"},
    "humaneval": {"cls": AFlowHumanEval, "prompt": _humaneval_cot_prompt, "post": _postprocess_code,     "vote": _vote_key_code,  "score": _score_humaneval, "metric": "pass@1"},
    "mbpp":      {"cls": AFlowMBPP,      "prompt": _mbpp_cot_prompt,      "post": _postprocess_code,     "vote": _vote_key_code,  "score": _score_mbpp,      "metric": "pass@1"},
}


# =================================================================== #
# ============================= Runner ============================== #
# =================================================================== #


async def _run_one(
    idx: int,
    example: Dict[str, Any],
    bench,
    llm: OpenAILLM,
    build_prompt: Callable,
    postprocess: Callable,
    vote_fn: Callable,
    score_fn: Callable,
    sem: asyncio.Semaphore,
    n_samples: int,
) -> Dict[str, Any]:
    prompt_text = build_prompt(example)
    raw_samples: List[Any] = []
    errors: List[str] = []
    async with sem:
        # Sample N paths sequentially to keep within per-task concurrency limit.
        # (Outer semaphore already limits how many problems run in parallel.)
        for _ in range(n_samples):
            try:
                raw_samples.append(await _call_llm(llm, prompt_text))
            except Exception as exc:
                raw_samples.append(None)
                errors.append(str(type(exc).__name__) + ": " + str(exc))

    processed: List[str] = []
    for r in raw_samples:
        if r is None:
            continue
        try:
            processed.append(postprocess(r))
        except Exception:
            processed.append(r)

    if not processed:
        return {
            "idx": idx,
            "id": bench._get_id(example),
            "prompt_preview": prompt_text[:200],
            "raw_samples": raw_samples,
            "processed_samples": processed,
            "vote_keys": [],
            "winner_key": None,
            "winner_count": 0,
            "winner_prediction": None,
            "metrics": {},
            "primary_score": 0.0,
            "errors": errors,
        }

    vote_keys = [vote_fn(p) for p in processed]
    winner_key, winner_count = _majority(vote_keys)
    # Score the FIRST sample that voted for the winner — that's an actual model output.
    winner_pred = next(p for p, k in zip(processed, vote_keys) if k == winner_key)

    try:
        metrics, primary = score_fn(bench, winner_pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0

    return {
        "idx": idx,
        "id": bench._get_id(example),
        "prompt_preview": prompt_text[:200],
        "raw_samples": raw_samples,
        "processed_samples": processed,
        "vote_keys": vote_keys,
        "winner_key": winner_key,
        "winner_count": winner_count,
        "winner_prediction": winner_pred,
        "metrics": metrics,
        "primary_score": float(primary),
        "errors": errors,
    }


async def _main_async(args: argparse.Namespace) -> None:
    cfg = DATASETS[args.dataset]
    bench_cls = cfg["cls"]
    build_prompt = cfg["prompt"]
    postprocess = cfg["post"]
    vote_fn = cfg["vote"]
    score_fn = cfg["score"]

    bench = bench_cls(path=str(DATA_DIR), mode="test")
    test_data: List[Dict[str, Any]] = bench.get_test_data() or []
    if args.limit is not None and args.limit > 0:
        test_data = test_data[: args.limit]
    total = len(test_data)
    if total == 0:
        raise RuntimeError("No test data loaded for dataset=" + args.dataset)

    metric_name = cfg["metric"]
    print(
        ">>> [CoT-SC baseline] dataset={d}  model={m}  total={n}  N={k}  T={t}  concurrency={c}  metric={x}".format(
            d=args.dataset, m=args.model, n=total, k=args.n_samples,
            t=args.temperature, c=args.concurrency, x=metric_name,
        )
    )

    llm = _make_llm(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(
            _run_one(i, ex, bench, llm, build_prompt, postprocess, vote_fn, score_fn, sem, args.n_samples)
        )
        for i, ex in enumerate(test_data)
    ]

    results: List[Dict[str, Any]] = []
    start = time.time()
    done_count = 0
    step = max(1, total // 20)
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done_count += 1
        if done_count % step == 0 or done_count == total:
            running_avg = sum(x["primary_score"] for x in results) / len(results)
            elapsed = time.time() - start
            print("  [{d}/{t}] running {k}={v:.4f}  elapsed={e:.1f}s".format(
                d=done_count, t=total, k=metric_name, v=running_avg, e=elapsed
            ))

    results.sort(key=lambda x: x["idx"])
    scores = [r["primary_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    num_errors = sum(1 for r in results if r.get("errors"))

    summary = {
        "method": "cot_self_consistency",
        "paper": "Wang et al. 2022 (ICLR 2023) -- Self-Consistency Improves CoT Reasoning",
        "dataset": args.dataset,
        "model": args.model,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "metric": metric_name,
        "count": len(results),
        "num_errors": num_errors,
        "score": avg,
        "elapsed_sec": round(time.time() - start, 2),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    out_path = OUTPUT_DIR / (args.dataset + "_" + safe_model + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(">>> CoT-SC baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   N samples  = {n}  (T={t})".format(n=args.n_samples, t=args.temperature))
    print(">>>   count      = {n} (errors={e})".format(n=len(results), e=num_errors))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoT with Self-Consistency baseline (Wang et al. 2022)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()),
                   help="Which benchmark to evaluate (uses its test split).")
    p.add_argument("--model", default="gemini-2.5-flash-lite", help="Model name passed to OpenAILLMConfig.")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=5,
                   help="Number of CoT paths to sample per question (Wang et al. default = 5).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for CoT paths (Wang et al. default = 0.7).")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Max parallel problems (each does N_samples sequential LLM calls).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of test samples (for quick smoke runs).")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        print("\n>>> interrupted by user")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()