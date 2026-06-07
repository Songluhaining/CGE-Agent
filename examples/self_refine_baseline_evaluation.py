import os
"""Self-Refine baseline (Madaan et al. 2023).

Reproduces Madaan et al. 2023, "Self-Refine: Iterative Refinement with
Self-Feedback" (NeurIPS 2023), on the six benchmarks.

Implementation:
  - Initial response uses the same few-shot CoT prompt as
    cot_baseline_evaluation.py (Wei et al. for GSM8K, Minerva for MATH,
    self-constructed for HotpotQA/DROP, zero-shot for HumanEval/MBPP).
  - Iterate up to N_max times:
      1) Critic prompt: ask the LLM to find concrete issues in the previous
         response. If the critic returns "OK" / "no issues" / etc. on its
         first non-empty line, we early-stop.
      2) Refine prompt: feed back the original task + previous response +
         critic feedback, ask for an improved response.
  - The final scored prediction is the LAST refined response (or the
    initial one if the critic accepted on round 0).
  - Defaults: N_max=5 (matches ADAS's Reflexion seed); Madaan paper uses
    up to 4 iterations on most tasks. Configurable via --n-max.

Run:
    python examples/self_refine_baseline_evaluation.py --dataset gsm8k
    python examples/self_refine_baseline_evaluation.py --dataset math --concurrency 5
    python examples/self_refine_baseline_evaluation.py --dataset hotpotqa --limit 50

Per-sample predictions and the aggregated score are written to
    examples/output/self_refine_baseline/<dataset>_<model>.json
"""
import argparse
import asyncio
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

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

# Reuse the CoT prompts and per-domain postprocessors so Self-Refine and CoT
# differ ONLY in the iterative critic+refine loop, not in the initial prompt.
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
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "self_refine_baseline"


# ------------------------------------------------------------------ LLM --

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
# ============= Critic / Refine prompts (Madaan 2023 style) ========== #
# =================================================================== #

CRITIC_INSTRUCTION = (
    "Carefully review the response above against the original task. "
    "Identify any specific mistakes, missing reasoning steps, or formatting "
    "issues. If the response is correct AND in the required final-answer "
    "format, reply with exactly 'OK' on the first line and nothing else. "
    "Otherwise, list the concrete issues as bullet points."
)

REFINE_INSTRUCTION = (
    "Given the original task, the previous response, and the feedback above, "
    "produce an improved response that addresses every issue raised in the "
    "feedback. Use the SAME final-answer format as before."
)


def _critic_says_ok(feedback: str) -> bool:
    """Heuristic for early-stop: critic indicates the previous response is fine."""
    if not feedback:
        return False
    s = feedback.strip().lower()
    first_line = s.split("\n", 1)[0].strip().rstrip(".").rstrip(",")
    if first_line in (
        "ok", "looks good", "no issues", "no mistakes", "correct",
        "the response is correct", "the answer is correct",
    ):
        return True
    head = s[:240]
    no_issue_phrases = (
        "no issues", "no mistakes", "no errors",
        "looks correct", "looks good", "is correct",
        "is fully correct", "no further improvements",
    )
    return any(p in head for p in no_issue_phrases)


def _build_critic_prompt(task_prompt: str, prev_response: str) -> str:
    return (
        "Original task:\n"
        + task_prompt
        + "\n\n----------\nPrevious response:\n"
        + prev_response
        + "\n\n----------\n"
        + CRITIC_INSTRUCTION
    )


def _build_refine_prompt(task_prompt: str, prev_response: str, feedback: str) -> str:
    return (
        "Original task:\n"
        + task_prompt
        + "\n\n----------\nPrevious response:\n"
        + prev_response
        + "\n\n----------\nFeedback:\n"
        + feedback
        + "\n\n----------\n"
        + REFINE_INSTRUCTION
    )


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
    "hotpotqa":  {"cls": AFlowHotPotQA,  "prompt": _hotpotqa_cot_prompt,  "post": _postprocess_hotpotqa, "score": _score_hotpotqa,  "metric": "f1"},
    "math":      {"cls": AFlowMATH,      "prompt": _math_cot_prompt,      "post": _postprocess_math,     "score": _score_math,      "metric": "solve_rate"},
    "gsm8k":     {"cls": AFlowGSM8K,     "prompt": _gsm8k_cot_prompt,     "post": _postprocess_gsm8k,    "score": _score_gsm8k,     "metric": "solve_rate"},
    "drop":      {"cls": AFlowDROP,      "prompt": _drop_cot_prompt,      "post": _postprocess_drop,     "score": _score_drop,      "metric": "f1"},
    "humaneval": {"cls": AFlowHumanEval, "prompt": _humaneval_cot_prompt, "post": _postprocess_code,     "score": _score_humaneval, "metric": "pass@1"},
    "mbpp":      {"cls": AFlowMBPP,      "prompt": _mbpp_cot_prompt,      "post": _postprocess_code,     "score": _score_mbpp,      "metric": "pass@1"},
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
    score_fn: Callable,
    sem: asyncio.Semaphore,
    n_max: int,
) -> Dict[str, Any]:
    task_prompt = build_prompt(example)
    rounds: List[Dict[str, Any]] = []
    response: str = ""

    async with sem:
        try:
            response = await _call_llm(llm, task_prompt)
        except Exception as exc:
            return {
                "idx": idx,
                "id": bench._get_id(example),
                "prompt_preview": task_prompt[:200],
                "rounds": [],
                "final_raw": None,
                "final_pred": None,
                "stopped_at": None,
                "error": "initial_call_failed: " + str(type(exc).__name__) + ": " + str(exc),
                "metrics": {},
                "primary_score": 0.0,
            }
        rounds.append({"role": "initial", "response": response})

        stopped_at = "max_iterations"
        for i in range(n_max):
            critic_prompt = _build_critic_prompt(task_prompt, response)
            try:
                feedback = await _call_llm(llm, critic_prompt)
            except Exception as exc:
                rounds.append({"role": "critic", "iteration": i, "error": str(type(exc).__name__) + ": " + str(exc)})
                stopped_at = "critic_call_failed"
                break
            rounds.append({"role": "critic", "iteration": i, "feedback": feedback})

            if _critic_says_ok(feedback):
                stopped_at = "critic_ok_iter_" + str(i)
                break

            refine_prompt = _build_refine_prompt(task_prompt, response, feedback)
            try:
                response = await _call_llm(llm, refine_prompt)
            except Exception as exc:
                rounds.append({"role": "refine", "iteration": i, "error": str(type(exc).__name__) + ": " + str(exc)})
                stopped_at = "refine_call_failed"
                break
            rounds.append({"role": "refine", "iteration": i, "response": response})

    try:
        pred = postprocess(response)
    except Exception:
        pred = response

    try:
        metrics, primary = score_fn(bench, pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0

    return {
        "idx": idx,
        "id": bench._get_id(example),
        "prompt_preview": task_prompt[:200],
        "rounds": rounds,
        "final_raw": response,
        "final_pred": pred,
        "stopped_at": stopped_at,
        "metrics": metrics,
        "primary_score": float(primary),
    }


async def _main_async(args: argparse.Namespace) -> None:
    cfg = DATASETS[args.dataset]
    bench_cls = cfg["cls"]
    build_prompt = cfg["prompt"]
    postprocess = cfg["post"]
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
        ">>> [Self-Refine baseline] dataset={d}  model={m}  total={n}  N_max={x}  T={t}  concurrency={c}  metric={k}".format(
            d=args.dataset, m=args.model, n=total, x=args.n_max,
            t=args.temperature, c=args.concurrency, k=metric_name,
        )
    )

    llm = _make_llm(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(
            _run_one(i, ex, bench, llm, build_prompt, postprocess, score_fn, sem, args.n_max)
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
    num_errors = sum(1 for r in results if r.get("error"))
    iters_used = [
        sum(1 for rd in r["rounds"] if rd.get("role") == "refine")
        for r in results
    ]
    avg_iters = sum(iters_used) / len(iters_used) if iters_used else 0.0
    early_stop_count = sum(1 for r in results if isinstance(r.get("stopped_at"), str) and r["stopped_at"].startswith("critic_ok"))

    summary = {
        "method": "self_refine",
        "paper": "Madaan et al. 2023 (NeurIPS) -- Self-Refine: Iterative Refinement with Self-Feedback",
        "dataset": args.dataset,
        "model": args.model,
        "n_max": args.n_max,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "metric": metric_name,
        "count": len(results),
        "num_errors": num_errors,
        "avg_refine_iterations": round(avg_iters, 3),
        "early_stop_count": early_stop_count,
        "score": avg,
        "elapsed_sec": round(time.time() - start, 2),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    out_path = OUTPUT_DIR / (args.dataset + "_" + safe_model + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(">>> Self-Refine baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   N_max      = {x}  (avg refine iters used = {a:.2f})".format(x=args.n_max, a=avg_iters))
    print(">>>   early_stop = {c}/{n} (critic accepted)".format(c=early_stop_count, n=len(results)))
    print(">>>   count      = {n} (errors={e})".format(n=len(results), e=num_errors))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-Refine baseline (Madaan et al. 2023)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()),
                   help="Which benchmark to evaluate (uses its test split).")
    p.add_argument("--model", default="gemini-2.5-flash-lite", help="Model name passed to OpenAILLMConfig.")
    p.add_argument("--n-max", dest="n_max", type=int, default=5,
                   help="Max critic+refine iterations (Madaan paper uses up to 4-5).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for both initial and refine calls.")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Max parallel problems (each does up to 2*N_max+1 sequential LLM calls).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of test samples.")
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