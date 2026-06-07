"""Chain-of-Thought (CoT) baseline evaluation.

Reproduces Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models" (NeurIPS 2022), on the six benchmarks available
in `data/datasets/`.

Exemplar sources (documented per dataset below):
  - GSM8K    : Wei et al. 2022, Table 20 (8-shot, verbatim).
  - MATH     : Lewkowycz et al. 2022 (Minerva) 4-shot exemplars, the
               de-facto standard CoT prompt for MATH that follows the
               Wei et al. few-shot-with-rationale recipe.
  - HotpotQA : Self-constructed 3-shot CoT exemplars -- the original CoT
               paper does NOT cover HotpotQA. Exemplars follow the
               Wei et al. format (question -> reasoning -> short answer).
  - DROP     : Self-constructed 3-shot CoT exemplars -- not covered by
               the original paper. Follows the same few-shot CoT recipe.
  - HumanEval: Zero-shot CoT ("Let's think step by step") following
               Kojima et al. 2022. The original CoT paper does not cover
               code-generation benchmarks, and no canonical few-shot CoT
               exemplars exist for HumanEval.
  - MBPP     : Zero-shot CoT, same rationale as HumanEval.

Run (on the server, with DASHSCOPE_API_KEY set):
    python examples/cot_baseline_evaluation.py --dataset gsm8k
    python examples/cot_baseline_evaluation.py --dataset math --concurrency 10
    python examples/cot_baseline_evaluation.py --dataset hotpotqa --limit 50
    python examples/cot_baseline_evaluation.py --dataset humaneval --model qwen-plus
    python examples/cot_baseline_evaluation.py --dataset mbpp
    python examples/cot_baseline_evaluation.py --dataset drop

Per-sample predictions and the aggregated score are written to
    examples/output/cot_baseline/<dataset>_<model>.json
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

from dotenv import load_dotenv

from evoagentx.models import AliyunLLM, AliyunLLMConfig, OpenAILLM, OpenRouterConfig, OpenRouterLLM, OpenAILLMConfig
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
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "cot_baseline"


# ------------------------------------------------------------------ LLM --

# def _make_llm(model: str, temperature: float = 0.0, max_tokens: int = 2048) -> AliyunLLM:
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

def _make_llm(model: str, temperature: float = 0.0, max_tokens: int = 2048) -> OpenAILLM:
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


# =================================================================== #
# ==================== Few-shot CoT exemplars ======================= #
# =================================================================== #

# ------ GSM8K: Wei et al. 2022, Table 20 (8-shot, verbatim) -------- #

GSM8K_COT_EXEMPLARS = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]


# ------ MATH: Minerva (Lewkowycz et al. 2022) 4-shot exemplars ----- #
# Standard CoT prompt for MATH, following the Wei et al. few-shot-with-rationale recipe.

def _math_exemplars() -> List[tuple]:
    bs = chr(92)  # literal backslash
    return [
        (
            "Find the domain of the expression " + bs + "frac{" + bs + "sqrt{x-2}}{" + bs + "sqrt{5-x}}.",
            "The expressions inside each square root must be non-negative. Therefore, "
            "x - 2 >= 0, so x >= 2, and 5 - x >= 0, so x <= 5. Also, the denominator "
            "cannot be equal to zero, so 5 - x > 0, which gives x < 5. Therefore, "
            "the domain of the expression is " + bs + "boxed{[2,5)}.",
        ),
        (
            "If " + bs + "det " + bs + "mathbf{A} = 2 and " + bs + "det " + bs + "mathbf{B} = 12, "
            "then find " + bs + "det (" + bs + "mathbf{A} " + bs + "mathbf{B}).",
            "We have that det(AB) = (det A)(det B) = (2)(12) = " + bs + "boxed{24}.",
        ),
        (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights "
            "instead, how many times must Terrell lift them in order to lift the same total weight?",
            "If Terrell lifts two 20-pound weights 12 times, he lifts a total of 2 * 12 * 20 = 480 "
            "pounds of weight. If he lifts two 15-pound weights instead for n times, he will lift a "
            "total of 2 * 15 * n = 30n pounds of weight. Equating this to 480 pounds, we can solve "
            "for n: 30n = 480, so n = " + bs + "boxed{16}.",
        ),
        (
            "If the system of equations 6x - 4y = a, 6y - 9x = b has a solution (x, y) where x and y "
            "are both nonzero, find a/b, assuming b is nonzero.",
            "If we multiply the first equation by -3/2, we obtain 6y - 9x = -3a/2. Since we also "
            "know that 6y - 9x = b, we have -3a/2 = b, so a/b = " + bs + "boxed{-" + bs + "frac{2}{3}}.",
        ),
    ]


MATH_COT_EXEMPLARS = _math_exemplars()


# ------ HotpotQA: self-constructed 3-shot CoT exemplars ------------ #
# NOTE: Not from the original CoT paper. Follows the Wei et al. format:
# short passage context, multi-hop reasoning, terse final answer.

HOTPOTQA_COT_EXEMPLARS = [
    (
        "Context:\nScott Derrickson is an American filmmaker. He is best known for directing horror films. "
        "Ed Wood was an American filmmaker, actor and author. He is known for directing low-budget genre films.\n\n"
        "Question: Were Scott Derrickson and Ed Wood of the same nationality?",
        "Scott Derrickson is American. Ed Wood was also American. Both are Americans, so they share the same nationality. "
        "The answer is yes.",
    ),
    (
        "Context:\nThe Oberoi family is an Indian family famous for their involvement in hotels. "
        "The Oberoi Group was founded in 1934 and operates in several countries. "
        "The Oberoi Group's flagship hotel is The Oberoi Udaivilas.\n\n"
        "Question: The Oberoi family is part of a hotel company that has a head office in what city?",
        "The Oberoi family runs The Oberoi Group. The Oberoi Group is headquartered in Delhi, India. "
        "The answer is Delhi.",
    ),
    (
        "Context:\nAllen Iverson is an American former professional basketball player. He played for the Philadelphia 76ers. "
        "The Philadelphia 76ers are an American professional basketball team based in Philadelphia.\n\n"
        "Question: In what city is the basketball team Allen Iverson played for based?",
        "Allen Iverson played for the Philadelphia 76ers. The Philadelphia 76ers are based in Philadelphia. "
        "The answer is Philadelphia.",
    ),
]


# ------ DROP: self-constructed 3-shot CoT exemplars ---------------- #
# NOTE: Not from the original CoT paper. DROP requires discrete
# reasoning (arithmetic, counting, sorting) over passages.

DROP_COT_EXEMPLARS = [
    (
        "Passage: The Broncos scored 24 points in the first quarter, 7 points in the second quarter, "
        "14 points in the third quarter, and 0 points in the fourth quarter.\n"
        "Question: How many points did the Broncos score in the first half?\nAnswer:",
        "The first half includes the first and second quarters. The Broncos scored 24 in Q1 and 7 in Q2. "
        "So in the first half they scored 24 + 7 = 31 points. The answer is 31.",
    ),
    (
        "Passage: In the 2010 census, the city had 45,012 residents. By the 2020 census, the population had "
        "grown to 52,874 residents.\n"
        "Question: How many more residents did the city have in 2020 than in 2010?\nAnswer:",
        "The 2020 population is 52,874 and the 2010 population is 45,012. "
        "The difference is 52,874 - 45,012 = 7,862. The answer is 7862.",
    ),
    (
        "Passage: The team made three touchdowns: a 5-yard run, a 12-yard pass, and a 27-yard pass.\n"
        "Question: How many yards was the longest touchdown?\nAnswer:",
        "The three touchdown distances are 5, 12, and 27 yards. The longest is 27. The answer is 27.",
    ),
]


# =================================================================== #
# ================== Prompt constructors per task ================== #
# =================================================================== #


def _gsm8k_cot_prompt(example: Dict[str, Any]) -> str:
    parts = []
    for q, a in GSM8K_COT_EXEMPLARS:
        parts.append("Q: " + q + "\nA: " + a)
    parts.append("Q: " + example["question"] + "\nA:")
    header = (
        "Solve each math word problem step by step. End with a line of the form "
        "'The answer is N' where N is the final numeric answer (no units, no commas).\n\n"
    )
    return header + "\n\n".join(parts)


def _math_cot_prompt(example: Dict[str, Any]) -> str:
    bs = chr(92)
    parts = []
    for q, a in MATH_COT_EXEMPLARS:
        parts.append("Problem: " + q + "\nSolution: " + a)
    parts.append("Problem: " + example["problem"] + "\nSolution:")
    header = (
        "Solve each competition-math problem step by step. The final answer MUST "
        "be enclosed in " + bs + "boxed{...} on the last line.\n\n"
    )
    return header + "\n\n".join(parts)


def _hotpotqa_cot_prompt(example: Dict[str, Any]) -> str:
    paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
    context_str = "\n".join(" ".join(p) for p in paragraphs)
    parts = []
    for q, a in HOTPOTQA_COT_EXEMPLARS:
        parts.append(q + "\nAnswer: " + a)
    question_block = (
        "Context:\n" + context_str + "\n\n"
        "Question: " + example["question"]
    )
    parts.append(question_block + "\nAnswer:")
    header = (
        "Answer each question using the given context. Reason step by step, then end "
        "with a line of the form 'The answer is X' where X is the shortest direct answer "
        "(a name, entity, or short phrase), with no quotes and no trailing punctuation.\n\n"
    )
    return header + "\n\n".join(parts)


def _drop_cot_prompt(example: Dict[str, Any]) -> str:
    # `example["context"]` already ends with "Answer:" in the AFlow DROP format.
    ctx = example["context"]
    if not ctx.rstrip().endswith("Answer:"):
        ctx = ctx.rstrip() + "\nAnswer:"
    parts = []
    for q, a in DROP_COT_EXEMPLARS:
        parts.append(q + " " + a)
    parts.append(ctx)
    header = (
        "Answer each question by reasoning over the passage step by step. End with a "
        "line of the form 'The answer is X' where X is the final answer (a number, date, "
        "or short span copied from the passage), with no extra punctuation.\n\n"
    )
    return header + "\n\n".join(parts)


def _humaneval_cot_prompt(example: Dict[str, Any]) -> str:
    # Zero-shot CoT (Kojima et al. 2022). The original Wei CoT paper does not
    # cover HumanEval, and no canonical few-shot CoT exemplars exist for it.
    return (
        "Complete the following Python function.\n"
        "Let's think step by step about the algorithm first (in a brief Python comment block "
        "at the top, prefixed with '# Reasoning:'), then write the full implementation.\n"
        "Return ONLY raw Python source code (no markdown fences, no explanation outside "
        "of the reasoning comment). The code must define the function `"
        + example["entry_point"] + "` with the exact signature from the prompt and be "
        "directly executable.\n\n"
        + example["prompt"]
    )


def _mbpp_cot_prompt(example: Dict[str, Any]) -> str:
    # Zero-shot CoT, same rationale as HumanEval.
    return (
        "Write a Python solution for the following task.\n"
        "Let's think step by step about the algorithm first (in a brief Python comment block "
        "at the top, prefixed with '# Reasoning:'), then write the full implementation.\n"
        "Return ONLY raw Python source code (no markdown fences, no explanation outside of "
        "the reasoning comment, no asserts, no examples). The code must define the function `"
        + example["entry_point"] + "` with the exact signature from the prompt and be "
        "directly executable.\n\n"
        + example["prompt"]
    )


# =================================================================== #
# =========== Post-processing: strip reasoning, keep answer ========= #
# =================================================================== #


def _extract_final_answer_text(pred: str) -> str:
    """For GSM8K / HotpotQA / DROP: pull the payload after 'The answer is'."""
    if not pred:
        return pred
    matches = list(re.finditer(r"[Tt]he\s+answer\s+is[:\s]+([^\n]+)", pred))
    if matches:
        tail = matches[-1].group(1).strip()
        tail = re.sub(r"\.\s*$", "", tail)
        return tail
    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    return lines[-1] if lines else pred


def _postprocess_gsm8k(pred: str) -> str:
    """Return a bare number on the last line, as AFlowGSM8K expects."""
    tail = _extract_final_answer_text(pred)
    tail = tail.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", tail)
    if m:
        return m.group(0)
    return tail


def _postprocess_hotpotqa(pred: str) -> str:
    return _extract_final_answer_text(pred)


def _postprocess_drop(pred: str) -> str:
    return _extract_final_answer_text(pred)


def _postprocess_math(pred: str) -> str:
    return pred


def _postprocess_code(pred: str) -> str:
    """Strip markdown fences if the model ignored instructions."""
    if not pred:
        return pred
    s = pred.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


# =================================================================== #
# ======================== Scoring functions ======================== #
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
    llm: AliyunLLM,
    build_prompt: Callable,
    postprocess: Callable,
    score_fn: Callable,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    prompt_text = build_prompt(example)
    async with sem:
        try:
            raw_pred = await _call_llm(llm, prompt_text)
        except Exception as exc:
            return {
                "idx": idx,
                "id": bench._get_id(example),
                "prompt_preview": prompt_text[:200],
                "raw_prediction": None,
                "prediction": None,
                "error": str(type(exc).__name__) + ": " + str(exc),
                "metrics": {},
                "primary_score": 0.0,
            }
    pred = postprocess(raw_pred)
    try:
        metrics, primary = score_fn(bench, pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0
    return {
        "idx": idx,
        "id": bench._get_id(example),
        "prompt_preview": prompt_text[:200],
        "raw_prediction": raw_pred,
        "prediction": pred,
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
        ">>> [CoT baseline] dataset={d}  model={m}  total={n}  concurrency={c}  metric={k}".format(
            d=args.dataset, m=args.model, n=total, c=args.concurrency, k=metric_name
        )
    )

    llm = _make_llm(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(
            _run_one(i, ex, bench, llm, build_prompt, postprocess, score_fn, sem)
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
        "method": "chain_of_thought",
        "paper": "Wei et al. 2022 (NeurIPS) -- Chain-of-Thought Prompting Elicits Reasoning in LLMs",
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
    print(">>> CoT baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   count      = {n} (errors={e})".format(n=len(results), e=num_errors))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chain-of-Thought baseline evaluator (Wei et al. 2022)")
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Which benchmark to evaluate (uses its test split).",
    )
    p.add_argument("--model", default="gemini-2.5-flash-lite", help="Aliyun/DashScope model name.")
    p.add_argument("--concurrency", type=int, default=10, help="Max concurrent LLM calls.")
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
