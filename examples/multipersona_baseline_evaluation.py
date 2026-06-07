"""MultiPersona / Solo Performance Prompting (SPP) baseline (Wang et al. NAACL 2024).

Reproduces the prompting strategy from
    Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, Heng Ji.
    "Unleashing the Emergent Cognitive Synergy in Large Language Models:
    A Task-Solving Agent through Multi-Persona Self-Collaboration".
    NAACL 2024 (Long Papers), pp. 257-279.
on the six benchmarks in `data/datasets/`.

Method essence (reproduced 1:1 from the official implementation at
https://github.com/MikeWangWZHL/Solo-Performance-Prompting):

  - SPP is ONE LLM call per test sample.  The "multi-round collaboration"
    happens INSIDE that single response: the model is prompted to
    (i) identify several personas, (ii) simulate a back-and-forth
    dialogue between them, (iii) emit "Final answer:" followed by the
    solution.  No iterative API calls, no retrieval, no ensemble.
  - Prompt template = the exact `spp_prompt` variable from
    `prompts/trivia_creative_writing.py` -- system instruction +
    two few-shot demonstrations (24-game math + CHATGPT poem) +
    "Now, identify the participants ... Task: <X>" trailer.  Verbatim,
    including the small calculation typo in the second-to-last line of
    Demo 1 (preserved for fidelity).

Hyperparameters (from official `run.py` defaults + scripts/*.sh):
  - temperature        : 0.0
  - top_p              : 1.0  (we leave it at the LLM client default,
                               which is 1.0 for OpenAI-compatible APIs)
  - num_generation     : 1     (single sample)
  - system_message     : "You are an AI assistant that helps people find information."
  - prompt variant     : `spp_prompt` (the standard one with two demos;
                                       NOT spp_profile / spp_fixed_persona /
                                       spp_less_demo)
  - answer parsing     : response.split("Final answer:")[1].strip();
                         on parse failure, return the full response
                         (matches `tasks/trivia_creative_writing.py`
                          prompt_unwrap behaviour exactly)

Dataset usage matches the other baseline scripts exactly:
  - Test data is loaded with bench_cls(path=DATA_DIR, mode="test").get_test_data().
  - Per-domain postprocessors are imported VERBATIM from
    cot_baseline_evaluation.py and applied to the SPP-stripped answer
    text, so the answer-extraction logic is identical to the CoT and
    CoT-SC baselines.
  - benchmark.evaluate() does the scoring.

Run:
    python examples/multipersona_baseline_evaluation.py --dataset gsm8k
    python examples/multipersona_baseline_evaluation.py --dataset math --concurrency 40
    python examples/multipersona_baseline_evaluation.py --dataset hotpotqa --limit 50
    python examples/multipersona_baseline_evaluation.py --dataset humaneval
    python examples/multipersona_baseline_evaluation.py --dataset mbpp
    python examples/multipersona_baseline_evaluation.py --dataset drop

Per-sample predictions and the aggregated score are written to
    examples/output/multipersona_baseline/<dataset>_<model>.json
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

from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import (
    AFlowDROP,
    AFlowGSM8K,
    AFlowHotPotQA,
    AFlowHumanEval,
    AFlowMATH,
    AFlowMBPP,
)

# Reuse the per-domain postprocessors from the canonical CoT baseline so
# answer extraction is identical to CoT / CoT-SC / Medprompt.
from cot_baseline_evaluation import (  # noqa: E402
    _postprocess_gsm8k,
    _postprocess_math,
    _postprocess_hotpotqa,
    _postprocess_drop,
    _postprocess_code,
)


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "datasets"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "multipersona_baseline"


# Hyperparameters from the official SPP implementation.
SPP_TEMPERATURE = 0.0
SPP_SYSTEM_MESSAGE = "You are an AI assistant that helps people find information."


# ====================================================================== #
# ====== SPP prompt header -- VERBATIM from the official repo ========== #
# ====================================================================== #
# Source: https://github.com/MikeWangWZHL/Solo-Performance-Prompting
#         prompts/trivia_creative_writing.py , variable `spp_prompt`
# The block below is the official text up to (but not including) the
# task description on the last line, which is dataset-specific and
# inserted by the per-dataset prompt builders below.
#
# Note: the calculation "12 + 12 = 12" near the end of Example 1 is a
# typo present in the official repository.  We preserve it verbatim
# rather than silently fixing it, so that this baseline reproduces the
# published SPP behaviour exactly.

SPP_HEADER = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

"""


# ====================================================================== #
# ============================== LLM =================================== #
# ====================================================================== #

def _make_llm(temperature: float = SPP_TEMPERATURE, max_tokens: int = 2048) -> OpenAILLM:
    """Same model + endpoint as cot_sc_baseline_evaluation.py."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm_config = OpenAILLMConfig(
        model="gemini-2.5-flash-lite",
        openai_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=600,
    )
    return OpenAILLM(config=llm_config)


async def _call_llm(llm: OpenAILLM, prompt: str, system: str = SPP_SYSTEM_MESSAGE) -> str:
    sys_msgs = [system] if system else None
    messages = llm.formulate_messages(prompts=[prompt], system_messages=sys_msgs)[0]
    out = await llm.single_generate_async(messages=messages)
    return out if isinstance(out, str) else str(out)


# ====================================================================== #
# ============= Per-dataset task descriptions (Task: ...) ============== #
# ====================================================================== #
#
# These take the place of the "Task: Write a short and coherent story..."
# trailer in the official spp_prompt template.  Field names match the
# AFlow benchmark schemas used by all four existing baseline scripts.

def _spp_task_gsm8k(ex: Dict[str, Any]) -> str:
    return (
        "Task: Solve the following grade-school math word problem. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "MUST contain only the final numeric value (no units, no currency symbol, "
        "no thousands separators, no trailing period).\n\n"
        "Question: " + ex["question"]
    )


def _spp_task_math(ex: Dict[str, Any]) -> str:
    bs = chr(92)
    return (
        "Task: Solve the following competition-math problem. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "MUST be enclosed in " + bs + "boxed{...}.\n\n"
        "Problem: " + ex["problem"]
    )


def _spp_task_hotpotqa(ex: Dict[str, Any]) -> str:
    paragraphs = [item[1] for item in ex["context"] if isinstance(item[1], list)]
    context_str = "\n".join(" ".join(p) for p in paragraphs)
    return (
        "Task: Answer the following question using ONLY the given context. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "should be the shortest direct answer (a name, entity, or short phrase), "
        "with no quotes and no trailing punctuation.\n\n"
        "Context:\n" + context_str + "\n\n"
        "Question: " + ex["question"]
    )


def _spp_task_drop(ex: Dict[str, Any]) -> str:
    # `context` already contains "Passage: ...\nQuestion: ...\nAnswer:"
    return (
        "Task: Answer the following question by reasoning over the passage. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "should be the final answer ONLY (a number, date, or short span copied "
        "verbatim from the passage), with no extra punctuation.\n\n"
        + ex["context"]
    )


def _spp_task_humaneval(ex: Dict[str, Any]) -> str:
    return (
        "Task: Complete the following Python function. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "MUST be raw, directly-executable Python source code defining the function "
        "`" + ex["entry_point"] + "` with the EXACT signature shown in the prompt "
        "(plus any required imports above it). Do NOT wrap the code in markdown "
        "fences, do NOT add explanatory text after the code, do NOT include "
        "example invocations, asserts, or an __main__ block.\n\n"
        + ex["prompt"]
    )


def _spp_task_mbpp(ex: Dict[str, Any]) -> str:
    return (
        "Task: Write a Python solution for the following task. "
        "Your final answer (the content following the 'Final answer:' prefix) "
        "MUST be raw, directly-executable Python source code defining the function "
        "`" + ex["entry_point"] + "` with the EXACT signature shown in the prompt "
        "(plus any required imports above it). Do NOT wrap the code in markdown "
        "fences, do NOT add explanatory text, do NOT include asserts, examples, "
        "or an __main__ block. The first non-blank line of your final answer "
        "MUST begin with `import`, `from`, `def`, or `class`.\n\n"
        + ex["prompt"]
    )


def _build_spp_prompt(task_text: str) -> str:
    """SPP_HEADER (verbatim) + per-dataset task description."""
    return SPP_HEADER + task_text


# ====================================================================== #
# =========== Parse "Final answer:" tail from SPP response ============= #
# ====================================================================== #

def _parse_spp_response(response: str) -> str:
    """Mirrors `prompt_unwrap` in tasks/trivia_creative_writing.py:

        if "Final answer:" in response:
            return response.split("Final answer:")[1].strip()
        return response

    On parse failure, return the full response (official fallback).  We
    use split-with-maxsplit-1 to avoid corrupting the answer when the
    model echoes the literal substring "Final answer:" inside its own
    persona dialogue before the real one.
    """
    if not response:
        return response
    if "Final answer:" in response:
        # Take everything AFTER the first "Final answer:" occurrence,
        # consistent with the official `[1]` indexing.
        return response.split("Final answer:", 1)[1].strip()
    return response


# ====================================================================== #
# ======================= Scoring (reused) ============================= #
# ====================================================================== #

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
    "hotpotqa":  {"cls": AFlowHotPotQA,  "task": _spp_task_hotpotqa,  "post": _postprocess_hotpotqa, "score": _score_hotpotqa,  "metric": "f1"},
    "math":      {"cls": AFlowMATH,      "task": _spp_task_math,      "post": _postprocess_math,     "score": _score_math,      "metric": "solve_rate"},
    "gsm8k":     {"cls": AFlowGSM8K,     "task": _spp_task_gsm8k,     "post": _postprocess_gsm8k,    "score": _score_gsm8k,     "metric": "solve_rate"},
    "drop":      {"cls": AFlowDROP,      "task": _spp_task_drop,      "post": _postprocess_drop,     "score": _score_drop,      "metric": "f1"},
    "humaneval": {"cls": AFlowHumanEval, "task": _spp_task_humaneval, "post": _postprocess_code,     "score": _score_humaneval, "metric": "pass@1"},
    "mbpp":      {"cls": AFlowMBPP,      "task": _spp_task_mbpp,      "post": _postprocess_code,     "score": _score_mbpp,      "metric": "pass@1"},
}


# ====================================================================== #
# ============================== Runner ================================ #
# ====================================================================== #

async def _run_one(
    idx: int,
    example: Dict[str, Any],
    bench,
    llm: OpenAILLM,
    cfg: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    task_text = cfg["task"](example)
    full_prompt = _build_spp_prompt(task_text)

    async with sem:
        try:
            raw_response = await _call_llm(llm, full_prompt)
        except Exception as exc:
            return {
                "idx": idx,
                "id": bench._get_id(example),
                "prompt_preview": full_prompt[-300:],
                "raw_response": None,
                "spp_answer": None,
                "prediction": None,
                "final_answer_parsed": False,
                "error": str(type(exc).__name__) + ": " + str(exc),
                "metrics": {},
                "primary_score": 0.0,
            }

    parsed_ok = ("Final answer:" in (raw_response or ""))
    spp_answer = _parse_spp_response(raw_response)
    pred = cfg["post"](spp_answer)

    try:
        metrics, primary = cfg["score"](bench, pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0

    return {
        "idx": idx,
        "id": bench._get_id(example),
        "prompt_preview": full_prompt[-300:],
        "raw_response": raw_response,
        "spp_answer": spp_answer,
        "prediction": pred,
        "final_answer_parsed": parsed_ok,
        "metrics": metrics,
        "primary_score": float(primary),
    }


async def _main_async(args: argparse.Namespace) -> None:
    cfg = DATASETS[args.dataset]
    bench_cls = cfg["cls"]

    bench = bench_cls(path=str(DATA_DIR), mode="test")
    test_data: List[Dict[str, Any]] = bench.get_test_data() or []
    if args.limit is not None and args.limit > 0:
        test_data = test_data[: args.limit]
    total = len(test_data)
    if total == 0:
        raise RuntimeError("No test data loaded for dataset=" + args.dataset)

    metric_name = cfg["metric"]
    print(
        ">>> [MultiPersona / SPP baseline] dataset={d}  model={m}  total={n}  T={t}  concurrency={c}  metric={k}".format(
            d=args.dataset, m=args.model, n=total, t=args.temperature, c=args.concurrency, k=metric_name,
        )
    )

    llm = _make_llm(temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_run_one(i, ex, bench, llm, cfg, sem))
        for i, ex in enumerate(test_data)
    ]

    results: List[Dict[str, Any]] = []
    start = time.time()
    step = max(1, total // 20)
    done = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done += 1
        if done % step == 0 or done == total:
            running = sum(x["primary_score"] for x in results) / len(results)
            print("  [{d}/{t}] running {k}={v:.4f}  elapsed={e:.1f}s".format(
                d=done, t=total, k=metric_name, v=running, e=time.time() - start,
            ))

    results.sort(key=lambda x: x["idx"])
    scores = [r["primary_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    num_errors = sum(1 for r in results if r.get("error"))
    num_unparsed = sum(1 for r in results if r.get("final_answer_parsed") is False and not r.get("error"))

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "method": "spp",
        "temperature": args.temperature,
        "system_message": SPP_SYSTEM_MESSAGE,
        "max_tokens": args.max_tokens,
        "metric": metric_name,
        "count": len(results),
        "num_errors": num_errors,
        "num_unparsed_final_answer": num_unparsed,
        "score": avg,
        "elapsed_sec": round(time.time() - start, 2),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    out_path = OUTPUT_DIR / (args.dataset + "_" + safe_model + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(">>> MultiPersona / SPP baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   count      = {n} (errors={e}, unparsed_final_answer={u})".format(
        n=len(results), e=num_errors, u=num_unparsed,
    ))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MultiPersona / SPP baseline (Wang et al. NAACL 2024)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--concurrency", type=int, default=40,
                   help="Max concurrent test problems (single LLM call per problem).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on test samples (smoke runs).")
    p.add_argument("--temperature", type=float, default=SPP_TEMPERATURE,
                   help="Sampling temperature (default 0.0; SPP official value).")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096,
                   help="Output token cap.  SPP responses include a long persona "
                        "dialogue followed by 'Final answer:' so this should be "
                        "generous; bump if responses get truncated before 'Final answer:'.")
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