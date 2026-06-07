"""Medprompt baseline evaluation (Nori et al. 2023).

Reproduces the prompting strategy from
    Harsha Nori et al., "Can Generalist Foundation Models Outcompete
    Special-Purpose Tuning? Case Study in Medicine", arXiv:2311.16452, 2023
on the six benchmarks in `data/datasets/`.

Medprompt has three components:

  (1) Dynamic kNN few-shot exemplar selection
        - For every test problem, retrieve the top-k most similar problems
          from a pool of "training-set" examples.
        - Embedding model: sentence-transformers/all-MiniLM-L6-v2
          (paper used text-embedding-ada-002 -- we use a local model
          to avoid hard-coding an embedding API endpoint, but the
          retrieval algorithm -- cosine similarity, top-k -- is identical).

  (2) Self-Generated Chain-of-Thought
        - Run the model (T=0) zero-shot CoT on the entire training pool.
        - Cross-check the predicted answer against ground truth via the
          benchmark.evaluate() method; keep only the examples whose
          self-generated CoT produced the correct answer.

  (3) Choice Shuffling Ensemble  (the only piece that does not transfer
      verbatim, because none of the six AFlow benchmarks is multiple
      choice)
        - The original paper introduces ensemble diversity by shuffling
          the order of A/B/C/D choices.  Per the user decision (Plan B
          in the design discussion), we replace this with N=5 high-
          temperature samples (T=0.7) of the same prompt and majority-
          vote the answer.  This is the standard non-multiple-choice
          adaptation of Medprompt used in the AFlow paper reported
          numbers (no implementation released).

Hyperparameters (from microsoft/promptbase reference implementation):
  - k (kNN neighbours)        : 5    [num_examples=5 in MMLU/MMLU.py]
  - N (ensemble samples)      : 5    [num_repeat=5  in MMLU/MMLU.py]
  - Inference temperature     : 0.7  [Plan B; see above]
  - Self-generated CoT temp.  : 0.0  [user-chosen; deterministic so the
                                      pool is reproducible]
  - Pool source               : *_validate.jsonl  (dev split)
  - Pool filter (correctness):
        gsm8k/math/humaneval/mbpp -> primary_score >= 1.0
        hotpotqa/drop             -> primary_score >= 0.5  (F1)

Dataset usage matches the other baseline scripts exactly:
  - Test data is read with bench_cls(path=DATA_DIR, mode="test").get_test_data().
  - The dev pool is read with bench_cls(path=DATA_DIR, mode="dev").get_dev_data().
  - Postprocessors and vote keys are imported verbatim from
    cot_baseline_evaluation.py and cot_sc_baseline_evaluation.py so that
    Medprompt and CoT-SC differ ONLY in
        (a) few-shot exemplar selection (kNN retrieval vs. fixed),
        (b) the source of the exemplar CoT (self-generated vs.
            hand-written from the literature).

Run:
    python examples/medprompt_baseline_evaluation.py --dataset gsm8k
    python examples/medprompt_baseline_evaluation.py --dataset math --concurrency 40
    python examples/medprompt_baseline_evaluation.py --dataset hotpotqa --limit 50
    python examples/medprompt_baseline_evaluation.py --dataset humaneval
    python examples/medprompt_baseline_evaluation.py --dataset mbpp
    python examples/medprompt_baseline_evaluation.py --dataset drop

The exemplar pool is built once and persisted to
    examples/output/medprompt_exemplars/<dataset>.json   (metadata)
    examples/output/medprompt_exemplars/<dataset>.npy    (embeddings)
Re-runs reuse it; pass --rebuild-pool to regenerate.

Per-sample predictions and the aggregated score are written to
    examples/output/medprompt_baseline/<dataset>_<model>.json
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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

from cot_sc_baseline_evaluation import (  # noqa: E402
    _vote_key_gsm8k,
    _vote_key_math,
    _vote_key_qa,
    _vote_key_code,
    _majority,
)


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "datasets"
POOL_DIR = REPO_ROOT / "examples" / "output" / "medprompt_exemplars"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "medprompt_baseline"

K_NEIGHBOURS = 5
N_ENSEMBLE = 5
INFERENCE_TEMPERATURE = 0.5
POOL_COT_TEMPERATURE = 0.0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _make_llm(temperature: float, max_tokens: int = 2048) -> OpenAILLM:
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


async def _call_llm(llm: OpenAILLM, prompt: str, system: str = None) -> str:
    sys_msgs = [system] if system else None
    messages = llm.formulate_messages(prompts=[prompt], system_messages=sys_msgs)[0]
    out = await llm.single_generate_async(messages=messages)
    return out if isinstance(out, str) else str(out)


def _qtext_gsm8k(ex):       return ex["question"]
def _qtext_math(ex):        return ex["problem"]
def _qtext_hotpotqa(ex):    return ex["question"]
def _qtext_drop(ex):        return ex["context"]
def _qtext_humaneval(ex):   return ex["prompt"]
def _qtext_mbpp(ex):        return ex["prompt"]


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


def _format_hotpotqa_context(ex):
    paragraphs = [item[1] for item in ex["context"] if isinstance(item[1], list)]
    return "\n".join(" ".join(p) for p in paragraphs)


def _mp_prompt_gsm8k(test_ex, pool_entries):
    parts = []
    for ent in pool_entries:
        parts.append("Q: " + ent["question_text"] + "\nA: " + ent["cot_response"].strip())
    parts.append("Q: " + test_ex["question"] + "\nA:")
    header = (
        "Solve each math word problem step by step. End with a line of the form "
        "'The answer is N' where N is the final numeric answer (no units, no commas).\n\n"
    )
    return header + "\n\n".join(parts)


def _mp_prompt_math(test_ex, pool_entries):
    bs = chr(92)
    parts = []
    for ent in pool_entries:
        parts.append("Problem: " + ent["question_text"] + "\nSolution: " + ent["cot_response"].strip())
    parts.append("Problem: " + test_ex["problem"] + "\nSolution:")
    header = (
        "Solve each competition-math problem step by step. The final answer MUST "
        "be enclosed in " + bs + "boxed{...} on the last line.\n\n"
    )
    return header + "\n\n".join(parts)


def _mp_prompt_hotpotqa(test_ex, pool_entries):
    parts = []
    for ent in pool_entries:
        ent_block = (
            "Context:\n" + ent.get("context_text", "") + "\n\n"
            "Question: " + ent["question_text"] + "\nAnswer: " + ent["cot_response"].strip()
        )
        parts.append(ent_block)
    test_block = (
        "Context:\n" + _format_hotpotqa_context(test_ex) + "\n\n"
        "Question: " + test_ex["question"] + "\nAnswer:"
    )
    parts.append(test_block)
    header = (
        "Answer each question using the given context. Reason step by step, then end "
        "with a line of the form 'The answer is X' where X is the shortest direct answer "
        "(a name, entity, or short phrase), with no quotes and no trailing punctuation.\n\n"
    )
    return header + "\n\n".join(parts)


def _mp_prompt_drop(test_ex, pool_entries):
    parts = []
    for ent in pool_entries:
        ent_ctx = ent["question_text"]
        if not ent_ctx.rstrip().endswith("Answer:"):
            ent_ctx = ent_ctx.rstrip() + "\nAnswer:"
        parts.append(ent_ctx + " " + ent["cot_response"].strip())
    test_ctx = test_ex["context"]
    if not test_ctx.rstrip().endswith("Answer:"):
        test_ctx = test_ctx.rstrip() + "\nAnswer:"
    parts.append(test_ctx)
    header = (
        "Answer each question by reasoning over the passage step by step. End with a "
        "line of the form 'The answer is X' where X is the final answer (a number, date, "
        "or short span copied from the passage), with no extra punctuation.\n\n"
    )
    return header + "\n\n".join(parts)


def _mp_prompt_humaneval(test_ex, pool_entries):
    header = (
        "Complete each Python function based on its signature and docstring.\n"
        "Let's think step by step about the algorithm first (in a brief Python comment block "
        "at the top, prefixed with '# Reasoning:'), then write the full implementation.\n"
        "Return ONLY raw Python source code (no markdown fences, no explanation outside "
        "of the reasoning comment). The code must define the function with the exact "
        "signature from each prompt and be directly executable.\n\n"
    )
    parts = []
    for ent in pool_entries:
        parts.append(
            "# ===== Example =====\n"
            + ent["question_text"].rstrip()
            + "\n# Solution:\n"
            + ent["cot_response"].rstrip()
        )
    parts.append(
        "# ===== Now complete this function (entry_point: "
        + test_ex["entry_point"] + ") =====\n"
        + test_ex["prompt"]
    )
    return header + "\n\n".join(parts)


def _mp_prompt_mbpp(test_ex, pool_entries):
    header = (
        "Write a Python solution for each task.\n"
        "Let's think step by step about the algorithm first (in a brief Python comment block "
        "at the top, prefixed with '# Reasoning:'), then write the full implementation.\n"
        "Return ONLY raw Python source code (no markdown fences, no explanation outside of "
        "the reasoning comment, no asserts, no examples). The code must define the function "
        "with the exact signature from each prompt and be directly executable.\n\n"
    )
    parts = []
    for ent in pool_entries:
        parts.append(
            "# ===== Example =====\n"
            + ent["question_text"].rstrip()
            + "\n# Solution:\n"
            + ent["cot_response"].rstrip()
        )
    parts.append(
        "# ===== Now solve this task (entry_point: "
        + test_ex["entry_point"] + ") =====\n"
        + test_ex["prompt"]
    )
    return header + "\n\n".join(parts)


DATASETS = {
    "hotpotqa": {
        "cls": AFlowHotPotQA, "qtext": _qtext_hotpotqa,
        "pool_prompt": _hotpotqa_cot_prompt, "mp_prompt": _mp_prompt_hotpotqa,
        "post": _postprocess_hotpotqa, "vote": _vote_key_qa,
        "score": _score_hotpotqa, "metric": "f1", "pool_threshold": 0.5,
    },
    "math": {
        "cls": AFlowMATH, "qtext": _qtext_math,
        "pool_prompt": _math_cot_prompt, "mp_prompt": _mp_prompt_math,
        "post": _postprocess_math, "vote": _vote_key_math,
        "score": _score_math, "metric": "solve_rate", "pool_threshold": 1.0,
    },
    "gsm8k": {
        "cls": AFlowGSM8K, "qtext": _qtext_gsm8k,
        "pool_prompt": _gsm8k_cot_prompt, "mp_prompt": _mp_prompt_gsm8k,
        "post": _postprocess_gsm8k, "vote": _vote_key_gsm8k,
        "score": _score_gsm8k, "metric": "solve_rate", "pool_threshold": 1.0,
    },
    "drop": {
        "cls": AFlowDROP, "qtext": _qtext_drop,
        "pool_prompt": _drop_cot_prompt, "mp_prompt": _mp_prompt_drop,
        "post": _postprocess_drop, "vote": _vote_key_qa,
        "score": _score_drop, "metric": "f1", "pool_threshold": 0.5,
    },
    "humaneval": {
        "cls": AFlowHumanEval, "qtext": _qtext_humaneval,
        "pool_prompt": _humaneval_cot_prompt, "mp_prompt": _mp_prompt_humaneval,
        "post": _postprocess_code, "vote": _vote_key_code,
        "score": _score_humaneval, "metric": "pass@1", "pool_threshold": 1.0,
    },
    "mbpp": {
        "cls": AFlowMBPP, "qtext": _qtext_mbpp,
        "pool_prompt": _mbpp_cot_prompt, "mp_prompt": _mp_prompt_mbpp,
        "post": _postprocess_code, "vote": _vote_key_code,
        "score": _score_mbpp, "metric": "pass@1", "pool_threshold": 1.0,
    },
}


_EMBEDDER = None


def _get_embedder(model_name_or_path=None):
    """Lazy-load the sentence-transformers model.

    Accepts an explicit `model_name_or_path` so the caller can pass a
    local directory when huggingface.co is unreachable (see
    --embedding-model on the CLI).
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        target = model_name_or_path or EMBEDDING_MODEL_NAME
        print(">>> Loading embedding model:", target)
        _EMBEDDER = SentenceTransformer(target)
    return _EMBEDDER


def _embed_texts(texts, batch_size=64, model_name_or_path=None):
    model = _get_embedder(model_name_or_path=model_name_or_path)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32)


async def _pool_one(idx, example, dev_bench, llm, cfg, sem):
    prompt_text = cfg["pool_prompt"](example)
    ctx_text = (
        _format_hotpotqa_context(example) if cfg["qtext"] is _qtext_hotpotqa else ""
    )
    base = {
        "idx": idx,
        "id": dev_bench._get_id(example),
        "question_text": cfg["qtext"](example),
        "context_text": ctx_text,
    }
    async with sem:
        try:
            raw_response = await _call_llm(llm, prompt_text)
        except Exception as exc:
            base.update({
                "raw_response": None, "cot_response": None,
                "primary_score": 0.0,
                "error": str(type(exc).__name__) + ": " + str(exc),
            })
            return base
    pred = cfg["post"](raw_response)
    try:
        _metrics, primary = cfg["score"](dev_bench, pred, example)
    except Exception as exc:
        base.update({
            "raw_response": raw_response, "cot_response": raw_response,
            "primary_score": 0.0,
            "error": "score_error: " + str(type(exc).__name__) + ": " + str(exc),
        })
        return base
    base.update({
        "raw_response": raw_response,
        "cot_response": raw_response,
        "primary_score": float(primary),
    })
    return base


def _raw_pool_path(dataset):
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    return POOL_DIR / (dataset + ".raw.json")


def _save_raw_results(dataset, raw_results, dev_size, threshold):
    """Persist PHASE 1 LLM outputs immediately, BEFORE the embedding step.

    The embedding step depends on the HuggingFace model cache and can fail
    on machines without internet access; without this checkpoint, a network
    failure would force re-running every dev LLM call.  With it, the user
    can fix the network problem and resume without losing the LLM work.
    """
    raw_path = _raw_pool_path(dataset)
    payload = {
        "dataset": dataset,
        "dev_size": dev_size,
        "pool_threshold": threshold,
        "pool_cot_temperature": POOL_COT_TEMPERATURE,
        "raw_results": raw_results,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(">>> Saved raw self-CoT results to: {p} ({n} entries)".format(
        p=raw_path, n=len(raw_results),
    ))


def _load_raw_results(dataset):
    raw_path = _raw_pool_path(dataset)
    if not raw_path.exists():
        return None
    with open(raw_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("raw_results", [])


async def _build_pool_async(dataset, cfg, args):
    """PHASE 1 in two stages:

    Stage A (LLM): produce raw self-CoT results from dev split.  Reuse
                   cached <dataset>.raw.json if present so we never
                   re-burn LLM tokens after a network failure in Stage B.
    Stage B (embed): encode the kept examples with sentence-transformers.

    Saving Stage A immediately on completion is the explicit reason this
    function does not just call _embed_texts at the end of the LLM loop.
    """
    raw_path = _raw_pool_path(dataset)
    raw_results = None
    if raw_path.exists() and not args.rebuild_pool:
        raw_results = _load_raw_results(dataset)
        if raw_results:
            print(">>> Reusing cached raw self-CoT results: {p} ({n} entries) -- skipping Stage A LLM step.".format(
                p=raw_path, n=len(raw_results),
            ))

    if raw_results is None:
        dev_bench = cfg["cls"](path=str(DATA_DIR), mode="dev")
        dev_data = dev_bench.get_dev_data() or []
        if args.pool_limit is not None and args.pool_limit > 0:
            dev_data = dev_data[: args.pool_limit]
        if not dev_data:
            raise RuntimeError(
                "No dev/validate data loaded for dataset=" + dataset
                + ".  Expected file: " + dataset + "_validate.jsonl under " + str(DATA_DIR)
            )
        print(
            ">>> [Medprompt PHASE 1 / Stage A: self-CoT] dataset={d}  dev_size={n}  pool_temp={t}  concurrency={c}".format(
                d=dataset, n=len(dev_data), t=POOL_COT_TEMPERATURE, c=args.concurrency,
            )
        )
        llm = _make_llm(temperature=POOL_COT_TEMPERATURE, max_tokens=args.max_tokens)
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [
            asyncio.create_task(_pool_one(i, ex, dev_bench, llm, cfg, sem))
            for i, ex in enumerate(dev_data)
        ]
        raw_results = []
        start = time.time()
        step = max(1, len(dev_data) // 20)
        done = 0
        try:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                raw_results.append(r)
                done += 1
                if done % step == 0 or done == len(dev_data):
                    kept_so_far = sum(1 for x in raw_results if x.get("primary_score", 0.0) >= cfg["pool_threshold"])
                    print("  pool [{d}/{t}]  kept_so_far={k}  elapsed={e:.1f}s".format(
                        d=done, t=len(dev_data), k=kept_so_far, e=time.time() - start,
                    ))
        finally:
            # Always checkpoint whatever finished, even on KeyboardInterrupt,
            # so partial work is recoverable.
            if raw_results:
                raw_results.sort(key=lambda x: x["idx"])
                _save_raw_results(dataset, raw_results, len(dev_data), cfg["pool_threshold"])

    kept = [r for r in raw_results if r.get("primary_score", 0.0) >= cfg["pool_threshold"]]
    print(">>> [Medprompt PHASE 1] kept {k} / {n} dev examples (threshold={th})".format(
        k=len(kept), n=len(raw_results), th=cfg["pool_threshold"],
    ))
    if not kept:
        raise RuntimeError(
            "Self-generated CoT produced no examples passing the correctness "
            "threshold for dataset=" + dataset + ".  Cannot build a Medprompt pool."
        )

    print(">>> [Medprompt PHASE 1 / Stage B: embed] embedding {n} kept exemplars...".format(n=len(kept)))
    try:
        texts = [e["question_text"] for e in kept]
        embeddings = _embed_texts(texts, model_name_or_path=getattr(args, "embedding_model", None) or None)
    except Exception as exc:
        print("")
        print(">>> Embedding step failed: " + str(type(exc).__name__) + ": " + str(exc))
        print(">>> The raw self-CoT pool is safe at " + str(raw_path) + ",")
        print(">>> so re-running this command after fixing the network/model issue")
        print(">>> will skip the LLM step and resume from embedding.")
        print(">>>")
        print(">>> If huggingface.co is blocked on this server, try ONE of:")
        print(">>>   (a) export HF_ENDPOINT=https://hf-mirror.com   # then re-run")
        print(">>>   (b) pre-download the model on a machine with internet access:")
        print(">>>         huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir /some/local/dir")
        print(">>>       then re-run with --embedding-model /some/local/dir")
        print(">>>   (c) use --embedding-model <local-path> to point at a model on disk.")
        raise
    return kept, embeddings


def _pool_paths(dataset):
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    return (POOL_DIR / (dataset + ".json"), POOL_DIR / (dataset + ".npy"))


def _save_pool(dataset, pool, embeddings):
    json_path, npy_path = _pool_paths(dataset)
    payload = {
        "dataset": dataset,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "k_neighbours": K_NEIGHBOURS,
        "n_ensemble": N_ENSEMBLE,
        "inference_temperature": INFERENCE_TEMPERATURE,
        "pool_cot_temperature": POOL_COT_TEMPERATURE,
        "size": len(pool),
        "pool": pool,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    np.save(npy_path, embeddings)
    print(">>> Saved exemplar pool: {j} ({n} entries) + {e}".format(
        j=json_path, n=len(pool), e=npy_path,
    ))


def _load_pool(dataset):
    json_path, npy_path = _pool_paths(dataset)
    if not json_path.exists() or not npy_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pool = payload.get("pool", [])
    embeddings = np.load(npy_path)
    if len(pool) != embeddings.shape[0]:
        raise RuntimeError(
            "Pool/embedding length mismatch for " + dataset + ": "
            + str(len(pool)) + " vs " + str(embeddings.shape[0])
        )
    print(">>> Reusing cached exemplar pool: {j} ({n} entries)".format(
        j=json_path, n=len(pool),
    ))
    return pool, embeddings


def _topk_indices(query_emb, pool_emb, k):
    sims = pool_emb @ query_emb
    if k >= sims.shape[0]:
        order = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        order = idx[np.argsort(-sims[idx])]
    return order.tolist()


async def _infer_one(idx, example, test_bench, llm, cfg, pool, pool_emb, test_emb, sem, k, n_samples):
    top_idx = _topk_indices(test_emb, pool_emb, k)
    selected = [pool[i] for i in top_idx]
    selected_in_order = list(reversed(selected))
    prompt_text = cfg["mp_prompt"](example, selected_in_order)
    raw_samples = []
    errors = []
    async with sem:
        for _ in range(n_samples):
            try:
                raw_samples.append(await _call_llm(llm, prompt_text))
            except Exception as exc:
                raw_samples.append(None)
                errors.append(str(type(exc).__name__) + ": " + str(exc))
    processed = []
    for r in raw_samples:
        if r is None:
            continue
        try:
            processed.append(cfg["post"](r))
        except Exception:
            processed.append(r)
    if not processed:
        return {
            "idx": idx, "id": test_bench._get_id(example),
            "prompt_preview": prompt_text[:200],
            "retrieved_pool_ids": [pool[i].get("id") for i in top_idx],
            "raw_samples": raw_samples, "processed_samples": processed,
            "vote_keys": [], "winner_key": None, "winner_count": 0,
            "winner_prediction": None, "metrics": {}, "primary_score": 0.0,
            "errors": errors,
        }
    vote_keys = [cfg["vote"](p) for p in processed]
    winner_key, winner_count = _majority(vote_keys)
    winner_pred = next(p for p, kk in zip(processed, vote_keys) if kk == winner_key)
    try:
        metrics, primary = cfg["score"](test_bench, winner_pred, example)
    except Exception as exc:
        metrics = {"_eval_error": str(type(exc).__name__) + ": " + str(exc)}
        primary = 0.0
    return {
        "idx": idx, "id": test_bench._get_id(example),
        "prompt_preview": prompt_text[:200],
        "retrieved_pool_ids": [pool[i].get("id") for i in top_idx],
        "raw_samples": raw_samples, "processed_samples": processed,
        "vote_keys": vote_keys, "winner_key": winner_key, "winner_count": winner_count,
        "winner_prediction": winner_pred, "metrics": metrics,
        "primary_score": float(primary), "errors": errors,
    }


async def _main_async(args):
    cfg = DATASETS[args.dataset]
    cached = None if args.rebuild_pool else _load_pool(args.dataset)
    if cached is None:
        pool, pool_emb = await _build_pool_async(args.dataset, cfg, args)
        _save_pool(args.dataset, pool, pool_emb)
    else:
        pool, pool_emb = cached

    test_bench = cfg["cls"](path=str(DATA_DIR), mode="test")
    test_data = test_bench.get_test_data() or []
    if args.limit is not None and args.limit > 0:
        test_data = test_data[: args.limit]
    total = len(test_data)
    if total == 0:
        raise RuntimeError("No test data loaded for dataset=" + args.dataset)
    metric_name = cfg["metric"]
    print(
        ">>> [Medprompt PHASE 2] dataset={d}  model={m}  total={n}  k={k}  N={N}  "
        "T_inf={t}  concurrency={c}  pool={p}  metric={k_}".format(
            d=args.dataset, m=args.model, n=total,
            k=args.k, N=args.n_samples, t=args.temperature,
            c=args.concurrency, p=len(pool), k_=metric_name,
        )
    )
    test_texts = [cfg["qtext"](ex) for ex in test_data]
    print(">>> Embedding {n} test questions...".format(n=total))
    test_emb = _embed_texts(test_texts, model_name_or_path=getattr(args, "embedding_model", None) or None)

    llm = _make_llm(temperature=args.temperature, max_tokens=args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_infer_one(
            i, ex, test_bench, llm, cfg, pool, pool_emb, test_emb[i], sem,
            k=args.k, n_samples=args.n_samples,
        ))
        for i, ex in enumerate(test_data)
    ]
    results = []
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
    num_errors = sum(1 for r in results if r.get("errors"))
    summary = {
        "dataset": args.dataset, "model": args.model,
        "k_neighbours": args.k, "n_ensemble": args.n_samples,
        "inference_temperature": args.temperature,
        "pool_cot_temperature": POOL_COT_TEMPERATURE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "pool_size": len(pool), "pool_threshold": cfg["pool_threshold"],
        "max_tokens": args.max_tokens, "metric": metric_name,
        "count": len(results), "num_errors": num_errors,
        "score": avg, "elapsed_sec": round(time.time() - start, 2),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    out_path = OUTPUT_DIR / (args.dataset + "_" + safe_model + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(">>> Medprompt baseline | {d} | {m}".format(d=args.dataset, m=args.model))
    print(">>>   {k:<10} = {v:.4f}".format(k=metric_name, v=avg))
    print(">>>   count      = {n} (with errors: {e})".format(n=len(results), e=num_errors))
    print(">>>   pool_size  = {p}".format(p=len(pool)))
    print(">>>   k / N / T  = {k} / {n} / {t}".format(k=args.k, n=args.n_samples, t=args.temperature))
    print(">>>   elapsed    = {e}s".format(e=summary["elapsed_sec"]))
    print(">>>   saved      = {p}".format(p=out_path))
    print("=" * 60)


def build_parser():
    p = argparse.ArgumentParser(description="Medprompt baseline (Nori et al. 2023)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--concurrency", type=int, default=40,
                   help="Max concurrent test problems (PHASE 2) and pool problems (PHASE 1).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on test samples (smoke runs).")
    p.add_argument("--pool-limit", dest="pool_limit", type=int, default=None,
                   help="Cap on dev/validate samples used to build the pool (smoke runs).")
    p.add_argument("--k", type=int, default=K_NEIGHBOURS,
                   help="Number of kNN exemplars per test problem (default 5; Nori et al. 2023).")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=N_ENSEMBLE,
                   help="Ensemble samples per test problem (default 5; Nori et al. 2023).")
    p.add_argument("--temperature", type=float, default=INFERENCE_TEMPERATURE,
                   help="Inference temperature (default 0.7; Plan B substitute for choice-shuffle).")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    p.add_argument("--rebuild-pool", action="store_true",
                   help="Force regeneration of the exemplar pool even if cached.")
    p.add_argument("--embedding-model", dest="embedding_model", default=None,
                   help="Override the sentence-transformers model name or local path "
                        "(useful when huggingface.co is unreachable; pass a directory "
                        "containing the pre-downloaded all-MiniLM-L6-v2 files).")
    return p


def main():
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
