"""I/O baseline: feed each test problem directly to the LLM and score the
raw response with the same benchmark.evaluate() logic used by the
self-evolving workflow.

Run on the server, e.g.:

    python examples/io_baseline_evaluation.py --dataset hotpotqa
    python examples/io_baseline_evaluation.py --dataset math --concurrency 20
    python examples/io_baseline_evaluation.py --dataset humaneval --limit 50
    python examples/io_baseline_evaluation.py --dataset mbpp --model qwen-plus
    python examples/io_baseline_evaluation.py --dataset gsm8k
    python examples/io_baseline_evaluation.py --dataset drop

Per-sample predictions and the aggregated score are written to
    examples/output/io_baseline/<dataset>_<model>.json
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List
from evoagentx.models import AliyunLLM, AliyunLLMConfig, OpenAILLM, OpenRouterConfig, OpenRouterLLM, OpenAILLMConfig
from dotenv import load_dotenv

from evoagentx.benchmark import (
    AFlowDROP,
    AFlowGSM8K,
    AFlowHotPotQA,
    AFlowHumanEval,
    AFlowMATH,
    AFlowMBPP,
)



load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "datasets"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "io_baseline"


# ------------------------------------------------------------------ LLM --

# def _make_llm(model: str, temperature: float = 0.0, max_tokens: int = 4096) -> AliyunLLM:
#     api_key = (os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
#     if api_key.lower().startswith("bearer "):
#         api_key = api_key.split(None, 1)[1].strip()
#     if not api_key:
#         raise RuntimeError(
#             "Set ALIYUN_API_KEY or DASHSCOPE_API_KEY in the environment before running."
#         )
#     cfg = AliyunLLMConfig(
#         model=model,
#         aliyun_api_key=api_key,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         timeout=600,
#     )
#     return AliyunLLM(config=cfg)


def _make_llm(model: str, temperature: float = 0.0, max_tokens: int = 4096) -> OpenAILLM:
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm_config = OpenAILLMConfig(
        model="gemini-2.5-flash-lite",
        openai_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.0,
        max_tokens=2048,
        timeout=600,
    )
    return OpenAILLM(config=llm_config)

async def _call_llm(llm: AliyunLLM, prompt: str, system: str = None) -> str:
    sys_msgs = [system] if system else None
    messages = llm.formulate_messages(prompts=[prompt], system_messages=sys_msgs)[0]
    out = await llm.single_generate_async(messages=messages)
    return out if isinstance(out, str) else str(out)


# --------------------------------------------------------- dataset configs --

def _hotpotqa_prompt(example: Dict[str, Any]) -> str:
    paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
    context_str = "\n".join(" ".join(p) for p in paragraphs)
    return (
        "Context:\n" + context_str + "\n\n"
        "Question: " + example["question"] + "\n\n"
        "Answer the question above using ONLY the context. "
        "Return the shortest direct answer (a name, entity, or short phrase) "
        "with no explanation, no quotes, and no extra punctuation."
    )


def _math_prompt(example: Dict[str, Any]) -> str:
    bs = chr(92)  # single backslash
    return (
        "Solve the following competition-math problem. "
        "Show brief reasoning if you wish, but the final answer MUST be enclosed "
        "in " + bs + "boxed{...} on the last line.\n\n"
        "Problem: " + example["problem"]
    )


def _gsm8k_prompt(example: Dict[str, Any]) -> str:
    return (
        "Solve the following grade-school math word problem.\n"
        "Give the reasoning briefly if needed, but the VERY LAST LINE must contain "
        "ONLY the final numeric answer (no units, no currency symbol, no thousands "
        "separators, no trailing period, no words).\n\n"
        "Question: " + example["question"]
    )


def _drop_prompt(example: Dict[str, Any]) -> str:
    # `context` already ends with "Answer:" in the AFlow DROP format.
    return (
        example["context"] + "\n"
        "Reply with ONLY the final answer (a number, date, or short span copied "
        "verbatim from the passage). No explanation."
    )


def _humaneval_prompt(example: Dict[str, Any]) -> str:
    return (
        "Complete the following Python function. Return ONLY raw Python source code "
        "(no markdown fences, no explanation). The code must define the function "
        "`" + example["entry_point"] + "` with the exact signature from the prompt "
        "and must be directly executable.\n\n"
        + example["prompt"]
    )


def _mbpp_prompt(example: Dict[str, Any]) -> str:
    return (
        "Write a Python solution for the following task. Return ONLY raw Python "
        "source code (no markdown fences, no explanation, no asserts, no examples). "
        "The code must define the function `" + example["entry_point"] + "` with the "
        "exact signature from the prompt and be directly executable.\n\n"
        + example["prompt"]
    )


# score functions: return (metrics_dict, primary_score_float)

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
    "hotpotqa":  {"cls": AFlowHotPotQA,  "prompt": _hotpotqa_prompt,  "score": _score_hotpotqa,  "metric": "f1"},
    "math":      {"cls": AFlowMATH,      "prompt": _math_prompt,      "score": _score_math,      "metric": "solve_rate"},
    "gsm8k":     {"cls": AFlowGSM8K,     "prompt": _gsm8k_prompt,     "score": _score_gsm8k,     "metric": "solve_rate"},
    "drop":      {"cls": AFlowDROP,      "prompt": _drop_prompt,      "score": _score_drop,      "metric": "f1"},
    "humaneval": {"cls": AFlowHumanEval, "prompt": _humaneval_prompt, "score": _score_humaneval, "metric": "pass@1"},
    "mbpp":      {"cls": AFlowMBPP,      "prompt": _mbpp_prompt,      "score": _score_mbpp,      "metric": "pass@1"},
}


# ------------------------------------------------------------------ run --

async def _run_one(
    idx: int,
    example: Dict[str, Any],
    bench,
    llm: AliyunLLM,
    build_prompt: Callable,
    score_fn: Callable,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    prompt_text = build_prompt(example)
    async with sem:
        try:
            pred = await _call_llm(llm, prompt_text)
        except Exception as exc:
            return {
                "idx": idx,
                "id": bench._get_id(example),
                "prompt_preview": prompt_text[:200],
                "prediction": None,
                "error": str(type(exc).__name__) + ": " + str(exc),
                "metrics": {},
                "primary_score": 0.0,
            }
    try:
        metrics, primary = score_fn(bench, pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0
    return {
        "idx": idx,
        "id": bench._get_id(example),
        "prompt_preview": prompt_text[:200],
        "prediction": pred,
        "metrics": metrics,
        "primary_score": float(primary),
    }


async def _main_async(args: argparse.Namespace) -> None:
    cfg = DATASETS[args.dataset]
    bench_cls = cfg["cls"]
    build_prompt = cfg["prompt"]
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
        ">>> dataset={d}  model={m}  total={n}  concurrency={c}  metric={k}".format(
            d=args.dataset, m=args.model, n=total, c=args.concurrency, k=metric_name
        )
    )

    llm = _make_llm(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(
            _run_one(i, ex, bench, llm, build_prompt, score_fn, sem)
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

    summary = {
        "dataset": args.dataset,
        "model": args.model,
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
    print(">>> I/O baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   count      = {n} (errors={e})".format(n=len(results), e=num_errors))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I/O baseline evaluator (raw LLM, no workflow)")
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Which benchmark to evaluate (uses its test split).",
    )
    p.add_argument("--model", default="gemini-2.5-flash-lite", help="Aliyun/DashScope model name.")
    p.add_argument("--concurrency", type=int, default=20, help="Max concurrent LLM calls.")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of test samples (for quick smoke runs).",
    )
    p.add_argument("--temperature", type=float, default=0.0)
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
