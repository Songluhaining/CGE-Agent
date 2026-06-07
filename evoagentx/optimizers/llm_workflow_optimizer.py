import ast
import copy
import json
import math
import os
import random
import re
import networkx as nx
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import Field

from evoagentx.benchmark import AFlowHotPotQA, HotPotQA, AFlowMATH, MATH, HumanEval, AFlowHumanEval, MBPP, AFlowMBPP, GSM8K, AFlowGSM8K, AFlowDROP
from evoagentx.core.base_config import Parameter
from evoagentx.evaluators.evaluator import Evaluator
from evoagentx.workflow.workflow_graph import WorkFlowEdge, WorkFlowGraph, WorkFlowNode
# Domain guidance helper is reused from workflow_generator so the optimizer's
# prompt-edit rewriter can inject the same per-node, per-benchmark rules that
# _build_fallback_agent_for_node and _derive_node_agent_suggestion consume.
from ..workflow.workflow_generator import _derive_domain_guidance as _optimizer_domain_guidance

from .optimizer import Optimizer
from .rl import (
    EvaluationCache,
    FactorCalibrationProfile,
    ModificationHistory,
    ModificationRecord,
    PARALLEL_VOTING_SKELETON,
    PromptHistory,
    PromptRecord,
    RewardConfig,
    build_factor_calibration_profile,
    compute_workflow_utility,
    recommend_motifs,
    render_motifs_for_prompt,
    workflow_complexity_metrics,
    workflow_fingerprint,
)
from . import rl_workflow_optimizer as _legacy


# === Ablation switch: disable Root Cause Analysis (RCA / causal inference) ===
# Set DISABLE_RCA = True to run the "wo_rca" ablation. When True:
#   - Loopy belief propagation (_refresh_package_rca) is skipped at init,
#     after init-simplification, and on every iteration baseline refresh.
#   - The RCA target pool is replaced by a uniform pseudo-pool spanning every
#     node x {Prompt, Params} plus one global Structure target.
#   - planner_context["rca"] is emptied (keys preserved to avoid KeyError).
#   - The "no_evidence_or_root_causes" early-stop is relaxed to require only
#     evidence (root_causes is no longer required).
# All other optimizer knobs (planner_candidate_count, repair rounds, prompt
# history, modification history, eval budget, etc.) remain identical so the
# comparison is RCA-only.
DISABLE_RCA: bool = False


_RCA_DISABLED_STRENGTH: Dict[str, float] = {
    "top1": 0.0,
    "top2": 0.0,
    "margin": 0.0,
    "normalized_entropy": 1.0,
    "tie_count": 0.0,
    "saturated": 0.0,
    "strong": 0.0,
}


def _build_uniform_pseudo_pool(workflow_graph) -> List[Any]:
    """Pseudo target pool used only when DISABLE_RCA=True.

    Covers every node x {Prompt, Params} plus one global Structure target so
    the planner can freely propose prompt_edit / params_edit / structure_edit
    candidates against any node without RCA-based pruning. Failure probs are
    all 0.0; ranks are assigned in traversal order so the resulting rank_map
    in _sanitize_planner_candidates is non-empty.
    """
    from .rl import RcaTarget  # local import keeps top-level import block intact
    pool: List[Any] = []
    rank = 0
    nodes = list(getattr(workflow_graph, "nodes", []) or [])
    components = (("Prompt", "Prompt"), ("Params", "Parse"))
    for node in nodes:
        node_name = getattr(node, "name", "") or ""
        if not node_name:
            continue
        for comp, sub in components:
            rank += 1
            pool.append(
                RcaTarget(
                    health_name=f"{comp}:{node_name}",
                    component=comp,
                    subtype=sub,
                    node_name=node_name,
                    failure_prob=0.0,
                    source="no_rca",
                    target_rank=rank,
                    pool_mode="no_rca",
                )
            )
    rank += 1
    pool.append(
        RcaTarget(
            health_name="Structure:__STRUCTURE__",
            component="Structure",
            subtype="Ordering",
            node_name="__STRUCTURE__",
            failure_prob=0.0,
            source="no_rca",
            target_rank=rank,
            pool_mode="no_rca",
        )
    )
    for item in pool:
        item.target_pool_size = len(pool)
    return pool


WORKFLOW_GOAL = _legacy.WORKFLOW_GOAL
_ALLOWED_OP_FAMILIES = {"ADD", "DELETE", "MODIFY"}
_ALLOWED_PARAM_FIELDS = {"temperature", "max_tokens", "top_p", "parse_mode"}
_ALLOWED_PARSE_MODES = {"str", "json"}
# Prompt style vocabulary.
#
# Each style maps to a dict with:
#   - description: one-line LLM-facing summary shown in the planner context.
#
# The vocabulary is unified across all datasets: workflow-optimization logic
# is task-independent, and only the per-node output format is adapted per
# benchmark (see _extract_dataset_format_rules and _derive_node_agent_suggestion).
# The planner sees the full style list for every task; dataset-specific format
# requirements are injected through the node instruction / output schema layer,
# not by filtering which high-level editing styles are available.
_PROMPT_STYLES: Dict[str, Dict[str, Any]] = {
    # ---- Universal / QA-originated styles (unchanged in semantics) ----
    "BINDING_REPAIR": {
        "description": "Add explicit {{variable_name}} placeholders for every required input. Use when upstream variables are missing or unnamed in the Instructions section.",
    },
    "SCHEMA_HARDEN": {
        "description": "Clarify the output schema. List each required key / field / return-type with its expected shape. Use when outputs fail to parse or have wrong structure.",
    },
    "GROUNDING_HARDEN": {
        "description": "Force the node to use ONLY provided evidence / context and avoid external knowledge. Use when the node hallucinates or drifts off the retrieved passages. QA-specific - not useful for closed-form code or math tasks that have no retrieval step.",
    },
    "CHAIN_SYNTHESIS": {
        "description": "Add a numbered reasoning chain that explicitly links intermediate facts to the final answer. Use when the node jumps to conclusions or drops middle steps.",
    },
    "DEDUP_SIMPLIFY": {
        "description": "Remove redundant / contradictory instruction blocks; merge overlapping steps. Use when repeated MODIFY edits stop converging.",
    },
    "ANSWER_NORMALIZE": {
        "description": "Enforce a short, canonical answer format. For QA: strip articles, return only the entity / date / quantity. For math: return only the numeric or boxed value. Not applicable to code (function bodies cannot be normalised).",
    },
    # ---- Code-task-only styles ----
    "CODE_EDGE_CASES": {
        "description": "Instruct the node to handle empty / zero / negative / single-element / duplicate / boundary / invalid inputs explicitly, returning the appropriate type-consistent default (None, 0, [], ()) rather than raising exceptions. Use when hidden unit tests fail on edge conditions.",
    },
    "CODE_CONTRACT_STRICT": {
        "description": "Enforce the exact function signature (name, parameter order, parameter names) from the problem statement, and the exact return type / shape implied by the public asserts (tuple vs list, int vs float, sorted vs unsorted). Use when the node emits the right logic but wrong shape.",
    },
    "CODE_IO_PURITY": {
        "description": "Force raw-executable output: first non-whitespace characters MUST be 'def ' or a required 'import'; no markdown fences, no triple-backtick python, no natural-language preamble, no asserts, no check() wrappers, no if __name__ block, no trailing commentary. Use when exec() on the output fails.",
    },
    # ---- Math-task-only styles ----
    "STEP_DECOMPOSE_MATH": {
        "description": "Instruct the node to decompose the problem into numbered steps: (1) identify known quantities, (2) set up equations, (3) compute intermediates, (4) produce the final answer in the required format. Use when reasoning skips steps or drops units.",
    },
    "ANSWER_BOXING_MATH": {
        "description": "Enforce the dataset-specific final-answer format: \\boxed{...} on its own line for MATH, or a bare numeric token on the last line for GSM8K. Must be the VERY last line of the output, nothing after. Use when the answer is correct but the format fails the extractor.",
    },
}


def _infer_task_profile(workflow_goal: Optional[str]) -> str:
    """Infer a coarse task profile from the workflow goal text.

    The returned profile is used ONLY for logging / observability. The planner
    no longer branches on it: the prompt-style vocabulary and Step 3 decision
    tree are unified across all datasets, and per-dataset output-format needs
    are handled by the node instruction / output schema layer (see
    _extract_dataset_format_rules and _derive_node_agent_suggestion).

    Detection is keyword-based and deliberately conservative: when no signal
    is found, we fall back to 'general'. Priority on multi-match: code > math
    > qa > general.
    """
    if not workflow_goal:
        return "general"
    text = str(workflow_goal).lower()
    code_hits = sum(1 for kw in (
        "python function", "def ", "entry_point", "entrypoint",
        "code generation", "pass@1", "pass@k", "humaneval", "mbpp",
        "unit test", "unit tests", "function definition", "function body",
        "function signature", "exec(", "assert ",
    ) if kw in text)
    math_hits = sum(1 for kw in (
        "math problem", "\\\\boxed", "boxed{", "gsm8k", "math dataset",
        "numerical answer", "numeric answer",
        "word problem", "equation", "solve for",
        "mathematical reasoning", "arithmetic reasoning",
    ) if kw in text)
    qa_hits = sum(1 for kw in (
        "hotpotqa", "hotpot qa", "drop dataset", "multi-hop",
        "reference document", "reference documents", "context passage",
        "retrieved", "retrieval", "evidence", "supporting fact",
        "open-domain", "passage", "span of text", "multiple spans",
        "reading comprehension", "token-level f1", "extractive",
    ) if kw in text)
    # Priority: code beats math beats qa on STRICT majority. On ties involving
    # qa, prefer qa because its style vocabulary is the original 6-style set -
    # the most conservative, behaviourally bit-identical fallback. This keeps
    # benchmarks like DROP (which use a few math-ish words like "counting" or
    # "arithmetic" in passing) routed to qa as long as they also emit any of
    # the reading-comprehension markers.
    if code_hits >= 1 and code_hits >= math_hits and code_hits >= qa_hits:
        return "code"
    if math_hits >= 1 and math_hits > qa_hits:
        return "math"
    if qa_hits >= 1:
        return "qa"
    if math_hits >= 1:
        return "math"
    return "general"


def _styles_for_profile(profile: str) -> List[str]:
    """Return the sorted list of prompt style names applicable to a task profile.

    The round-3 redesign restores task-aware style filtering after a round-2
    experiment showed that exposing the full 11-style vocabulary for every task
    let the planner LLM pick benchmark-mismatched styles (e.g. ANSWER_BOXING_MATH
    on DROP) with large negative utility. The filter is driven entirely by the
    coarse profile inferred from the workflow goal (no per-dataset hardcoding)
    and has four cases:

      qa       - the 6 universal styles. QA benchmarks (HotpotQA, DROP) get
                 GROUNDING_HARDEN (retrieval/evidence) and ANSWER_NORMALIZE
                 (canonical token form) which are their bread-and-butter fixes.
      math     - universal minus GROUNDING_HARDEN (no retrieval step), plus
                 STEP_DECOMPOSE_MATH and ANSWER_BOXING_MATH which enforce
                 numbered-step reasoning and \\boxed{}/bare-number output.
      code     - universal minus GROUNDING_HARDEN (no retrieval) and minus
                 ANSWER_NORMALIZE (function bodies cannot be normalised), plus
                 CODE_EDGE_CASES, CODE_CONTRACT_STRICT, CODE_IO_PURITY for edge
                 handling, signature matching, and raw-executable output.
      general  - same as qa: the 6 universal styles. Conservative default when
                 the task type is unknown or the goal text is hybrid.

    Unknown or malformed profile strings fall back to the general set.
    """
    prof = str(profile or "general").strip().lower() or "general"
    universal = {
        "BINDING_REPAIR",
        "SCHEMA_HARDEN",
        "GROUNDING_HARDEN",
        "CHAIN_SYNTHESIS",
        "DEDUP_SIMPLIFY",
        "ANSWER_NORMALIZE",
    }
    if prof == "math":
        styles = (universal - {"GROUNDING_HARDEN"}) | {
            "STEP_DECOMPOSE_MATH",
            "ANSWER_BOXING_MATH",
        }
    elif prof == "code":
        styles = (universal - {"GROUNDING_HARDEN", "ANSWER_NORMALIZE"}) | {
            "CODE_EDGE_CASES",
            "CODE_CONTRACT_STRICT",
            "CODE_IO_PURITY",
        }
    elif prof == "qa":
        styles = set(universal)
    else:  # "general" and anything unrecognised
        styles = set(universal)
    # Intersect with _PROMPT_STYLES so we never emit a name that is not defined
    # (protects against vocabulary edits here getting out of sync with the
    # _PROMPT_STYLES dict). Return sorted for stable planner prompt ordering.
    return sorted(styles & set(_PROMPT_STYLES.keys()))


def _style_vocab_descriptions(style_names: Sequence[str]) -> Dict[str, str]:
    """Map style names to their one-line descriptions, preserving order."""
    out: Dict[str, str] = {}
    for name in style_names:
        spec = _PROMPT_STYLES.get(str(name).strip().upper())
        if spec is not None:
            out[str(name).strip().upper()] = str(spec.get("description") or "").strip()
    return out
_PARAM_STYLES = {"STRICT_JSON", "LOWER_TEMPERATURE", "LOWER_TOP_P", "MORE_TOKENS"}
_STRUCTURE_STYLE_VARIANTS = {
    "REWIRE_EDGE": ["linear_chain", "add_shortcut_edge", "swap_middle_stages"],
    "MERGE_NODE": ["merge_extract_organize", "merge_synthesize_answer"],
    "DELETE_NODE": ["delete_evidence_stage", "delete_redundant_synthesis"],
    "INSERT_NODE": ["insert_output_validator", "insert_evidence_organizer", "insert_reasoning_chain_stage"],
    "SPLIT_NODE": ["split_extract_and_organize", "split_reasoning_and_answer"],
    "LLM_PROPOSE": ["llm_open_structure"],
}

# ---- Per-dataset configuration ----
# Each entry describes the sample fields the planner LLM will see in
# failure_examples, the applicable prompt styles, and whether the
# gold_in_node_output substring match is reliable for fault localization.
_DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "hotpotqa": {
        "question_desc": "a natural-language multi-hop question (from the `question` field)",
        "gold_desc": "a short answer string (entity, date, or phrase)",
        "gold_match_reliable": True,
        "profile": "qa",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "GROUNDING_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "ANSWER_NORMALIZE"},
    },
    "drop": {
        "question_desc": "a passage with an embedded question (from the `context` field, concatenating Passage + Question)",
        "gold_desc": "a short answer string (number, entity, or date span)",
        "gold_match_reliable": True,
        "profile": "qa",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "GROUNDING_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "ANSWER_NORMALIZE"},
    },
    "gsm8k": {
        "question_desc": "a grade-school math word problem (from the `question` field)",
        "gold_desc": "a single number (the final numeric answer)",
        "gold_match_reliable": True,
        "profile": "math",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "ANSWER_NORMALIZE", "STEP_DECOMPOSE_MATH", "ANSWER_BOXING_MATH"},
    },
    "math": {
        "question_desc": "a competition-level math problem statement (from the `problem` field)",
        "gold_desc": "the full reference solution text including explanation and boxed answer (may be truncated, NOT just the final value)",
        "gold_match_reliable": False,
        "gold_unreliable_reason": "a long solution excerpt, not just the answer",
        "diagnosis_extra": "compare `predicted_answer` vs `gold_answer` directly and inspect `node_outputs` to trace reasoning errors. Focus on intermediate steps, final format (boxed / bare number), units, and signs",
        "profile": "math",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "ANSWER_NORMALIZE", "STEP_DECOMPOSE_MATH", "ANSWER_BOXING_MATH"},
    },
    "humaneval": {
        "question_desc": "a function signature with docstring (from the `prompt` field)",
        "gold_desc": "a serialized dict containing `canonical_solution`, `test`, and `entry_point` \u2014 NOT a simple answer string",
        "gold_match_reliable": False,
        "gold_unreliable_reason": "a serialized dict",
        "diagnosis_extra": "compare `predicted_answer` against `question` (the function spec) directly. Check syntax validity, signature match, edge case handling, and return type. Inspect `node_outputs` to trace where the code went wrong",
        "profile": "code",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "CODE_EDGE_CASES", "CODE_CONTRACT_STRICT", "CODE_IO_PURITY"},
    },
    "mbpp": {
        "question_desc": "a natural-language task description with function signature (from the `prompt` field)",
        "gold_desc": "a serialized dict containing `canonical_solution`, `test`, and `entry_point` \u2014 NOT a simple answer string",
        "gold_match_reliable": False,
        "gold_unreliable_reason": "a serialized dict",
        "diagnosis_extra": "compare `predicted_answer` against `question` (the function spec) directly. Check syntax validity, signature match, edge case handling, and return type. Inspect `node_outputs` to trace where the code went wrong",
        "profile": "code",
        "styles": {"BINDING_REPAIR", "SCHEMA_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY", "CODE_EDGE_CASES", "CODE_CONTRACT_STRICT", "CODE_IO_PURITY"},
    },
}

_BENCHMARK_NAME_MAP: Dict[str, str] = {
    "AFlowHotPotQA": "hotpotqa", "HotPotQA": "hotpotqa",
    "AFlowDROP": "drop",
    "AFlowMATH": "math", "MATH": "math",
    "AFlowHumanEval": "humaneval", "HumanEval": "humaneval",
    "AFlowMBPP": "mbpp", "MBPP": "mbpp",
    "AFlowGSM8K": "gsm8k", "GSM8K": "gsm8k",
}


def _resolve_dataset_name(benchmark) -> str:
    """Resolve benchmark object to a canonical dataset key in _DATASET_CONFIG."""
    cls_name = type(benchmark).__name__
    return _BENCHMARK_NAME_MAP.get(cls_name, cls_name.lower().replace("aflow", ""))


def _styles_for_dataset(dataset_name: str) -> List[str]:
    """Return sorted list of prompt style names for a specific dataset."""
    cfg = _DATASET_CONFIG.get(dataset_name)
    if cfg is None:
        cfg = _DATASET_CONFIG["hotpotqa"]
    return sorted(cfg["styles"] & set(_PROMPT_STYLES.keys()))


def _build_step2_for_dataset(dataset_name: str) -> str:
    """Build Step 2 prompt text. Each profile (qa / math / code) gets:
      - the fields it actually has in the failure example dict
      - a compact failure_mode -> style map keyed off those fields
    so the planner reads one consistent mapping instead of cross-
    referencing Step 2 fields against a Step 3 trigger tree.
    """
    cfg = _DATASET_CONFIG.get(dataset_name)
    if cfg is None:
        cfg = _DATASET_CONFIG["hotpotqa"]
    profile = _profile_for_dataset(dataset_name)
    q_desc = cfg["question_desc"]
    g_desc = cfg["gold_desc"]

    common = (
        "## Step 2 Localize the Fault\n"
        "Each failure example is a standardized dict with these fields:\n"
        f"- `question/problem`: {q_desc}.\n"
        f"- `gold_answer/solution`: {g_desc}.\n"
        "- `predicted_answer`: what the workflow ACTUALLY returned.\n"
        "- `node_outputs`: dict mapping each node name to its output text.\n"
        "- `failure_mode`: see the profile-specific list below.\n"
        "- `f1`, `em`: per-sample metric scores (0\u20131).\n"
    )

    if cfg.get("gold_match_reliable"):
        substr_block = (
            "- `gold_in_node_output[node]`: whether `gold_answer` appeared as a substring in that node's output (reliable here because the gold is a short answer string).\n"
            "- `first_fault_node`: the first node where the gold answer was lost.\n"
        )
    else:
        substr_block = ""

    if profile == "math":
        slice_block = (
            "- `predicted_boxed`: value extracted from `\\boxed{...}` in the prediction (null if absent).\n"
            "- `gold_boxed`: value extracted from `\\boxed{...}` in the reference solution (null if absent).\n"
            "- `numeric_match`: True iff predicted and gold answers match numerically.\n"
        )
        diagnosis = (
            "\nfailure_mode (math): empty_answer | format_error | close_value | wrong_value.\n"
            "Map symptom \u2192 style (apply on the indicated node):\n"
            "- format_error OR predicted_boxed = null \u2192 ANSWER_BOXING_MATH on the final answer node.\n"
            "- close_value \u2192 STEP_DECOMPOSE_MATH on the solving node (arithmetic slip).\n"
            "- wrong_value \u2192 STEP_DECOMPOSE_MATH on the solving node (reasoning error upstream)."
        )
    elif profile == "code":
        slice_block = (
            "- `code_block_extracted`: whether a `def ...:` block was recovered from the prediction.\n"
            "- `syntax_ok`: whether the extracted code parses with Python ast.\n"
            "- `signature_match`: whether the function signature matches the required entry_point (null if unknown).\n"
        )
        diagnosis = (
            "\nfailure_mode (code): empty_answer | syntax_error | signature_mismatch | wrong_logic.\n"
            "Map symptom \u2192 style (apply on the code-generating node):\n"
            "- syntax_error OR code_block_extracted = false \u2192 CODE_IO_PURITY.\n"
            "- signature_match = false \u2192 CODE_CONTRACT_STRICT.\n"
            "- wrong_logic \u2192 CODE_EDGE_CASES."
        )
    else:  # qa
        slice_block = ""
        diagnosis = (
            "\nfailure_mode (qa): empty_answer | too_long | partial_overlap | close_miss | wrong_entity.\n"
            "Map symptom \u2192 style:\n"
            "- first_fault_node = early/input node \u2192 GROUNDING_HARDEN on that node.\n"
            "- first_fault_node = middle/reasoning node \u2192 CHAIN_SYNTHESIS on that node.\n"
            "- too_long / close_miss / partial_overlap \u2192 ANSWER_NORMALIZE on the final answer node.\n"
            "- empty_answer \u2192 SCHEMA_HARDEN on the failing node."
        )

    return common + substr_block + slice_block + diagnosis


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_primary_metric(metrics_dict: dict, default: float = 0.0) -> float:
    """Extract primary score from metrics dict: f1 for QA, pass@1 for code, solve_rate/accuracy for math."""
    if not metrics_dict:
        return default
    for key in ("f1", "pass@1", "accuracy", "solve_rate"):
        if key in metrics_dict:
            try:
                return float(metrics_dict[key])
            except (TypeError, ValueError):
                return default
    return default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _round_nested(value: Any, digits: int = 4) -> Any:
    if isinstance(value, dict):
        return {str(k): _round_nested(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_nested(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def _llm_text(llm, prompt: str, system_message: Optional[str] = None) -> str:
    out = llm.generate(prompt=prompt, system_message=system_message, parse_mode="str")
    return str(getattr(out, "content", out)).strip()


def _normalize_component(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"prompt": "Prompt", "params": "Params", "structure": "Structure"}.get(text, str(value or "").strip())


def _normalize_target_subtype(component: Any, subtype: Any) -> str:
    component_text = str(component or "").strip()
    subtype_text = str(subtype or "").strip()
    if not subtype_text:
        return ""
    alias_map = {
        "Prompt": {
            "binding": "Binding",
            "contract": "Contract",
            "grounding": "Grounding",
            "prompt": "Prompt",
        },
        "Params": {
            "parse": "Parse",
            "length": "Length",
            "temperature": "Temperature",
            "sampling": "Sampling",
        },
        "Structure": {
            "ordering": "Ordering",
            "structure": "Structure",
            "coverage": "Coverage",
        },
        "Return": {
            "type": "Type",
            "task": "Task",
            "return": "Return",
            "content": "Content",
            "evidence": "Evidence",
        },
        "Edge": {
            "semantic": "Semantic",
            "ordering": "Ordering",
        },
    }
    normalized = alias_map.get(component_text, {}).get(subtype_text.lower())
    return normalized or subtype_text


def _target_signature(component: Any, subtype: Any, node_name: Any) -> Tuple[str, str, str]:
    return (
        _normalize_component(component).strip().lower(),
        str(subtype or "").strip().lower(),
        str(node_name or "").strip().lower(),
    )


def _project_rca_target_row(row: Dict[str, Any]) -> Dict[str, Any]:
    diagnostic_component = _normalize_component(row.get("component"))
    diagnostic_subtype = _normalize_target_subtype(diagnostic_component, row.get("subtype"))
    node_name = str(row.get("node_name") or "").strip()
    projection_reason = "same_component"
    fallback = None
    if diagnostic_component == "Prompt":
        component = "Prompt"
        subtype = diagnostic_subtype or "Prompt"
        node = node_name
    elif diagnostic_component == "Params":
        component = "Params"
        subtype = diagnostic_subtype or "Parse"
        node = node_name
    elif diagnostic_component == "Structure":
        component = "Structure"
        subtype = diagnostic_subtype or "Ordering"
        node = "__STRUCTURE__"
    elif diagnostic_component == "Return":
        if diagnostic_subtype in {"Type", "Task"}:
            component = "Prompt"
            subtype = "Contract"
            fallback = {"component": "Params", "subtype": "Parse", "node_name": node_name}
            projection_reason = f"map_{diagnostic_component.lower()}_{diagnostic_subtype.lower()}_to_prompt_contract"
        else:
            component = "Prompt"
            subtype = "Grounding"
            fallback = {"component": "Prompt", "subtype": "Binding", "node_name": node_name}
            projection_reason = f"map_{diagnostic_component.lower()}_{diagnostic_subtype.lower()}_to_prompt_grounding"
        node = node_name
    elif diagnostic_component == "Edge":
        component = "Structure"
        subtype = "Ordering"
        node = "__STRUCTURE__"
        projection_reason = "map_edge_to_structure_ordering"
    else:
        component = "Structure"
        subtype = "Ordering"
        node = "__STRUCTURE__"
        projection_reason = "unknown_component_fallback_to_structure"
    return {
        "rca_rank": int(row.get("rca_rank", 0) or 0),
        "component": component,
        "subtype": subtype,
        "node_name": node,
        "failure_prob": round(_safe_rate(row.get("failure_prob", 0.0)), 6),
        "source": row.get("source"),
        "edge_source": row.get("edge_source"),
        "edge_target": row.get("edge_target"),
        "diagnostic_component": diagnostic_component,
        "diagnostic_subtype": diagnostic_subtype,
        "diagnostic_node_name": node_name,
        "projection_reason": projection_reason,
        "fallback_target": fallback,
    }


def _project_rca_targets_to_supported_targets(target_pool: Sequence[Any]) -> List[Dict[str, Any]]:
    return [_project_rca_target_row(row) for row in _planner_targets(target_pool)]


def _workflow_summary(workflow_graph: WorkFlowGraph) -> Dict[str, Any]:
    structure_valid, structure_reasons, structure_meta = _legacy._validate_workflow_structure_for_evolution(workflow_graph)
    complexity = workflow_complexity_metrics(workflow_graph)
    return {
        "node_count": int(complexity.get("node_count", 0.0)),
        "edge_count": int(complexity.get("edge_count", 0.0)),
        "dag_depth": int(complexity.get("dag_depth", 0.0)),
        "role_meta": structure_meta.get("role_meta", {}),
        "consumed_intermediate_outputs": int(structure_meta.get("consumed_intermediate_outputs", 0)),
        "interface_ok": bool(structure_meta.get("interface_ok", False)),
        "structure_valid": bool(structure_valid),
        "structure_reasons": list(structure_reasons),
        "workflow_description": workflow_graph.get_workflow_description(),
    }


def _best_workflow_summary(best_workflow: WorkFlowGraph, best_results: Dict[str, float]) -> Any:
    raw = _legacy._best_workflow_summary(best_workflow, best_results)
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    if isinstance(parsed, dict):
        parsed["estimated_full_f1"] = round(_safe_rate((best_results or {}).get("estimated_full_f1", 0.0)), 4)
    return parsed


def _planner_targets(target_pool: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target in target_pool or []:
        rows.append(
            {
                "rca_rank": int(target.target_rank),
                "component": target.component,
                "subtype": target.subtype,
                "node_name": target.node_name,
                "failure_prob": round(float(target.failure_prob), 6),
                "source": target.source,
                "edge_source": target.edge_source,
                "edge_target": target.edge_target,
            }
        )
    return rows


def _is_successful_eval_record(record: Dict[str, Any], success_f1_threshold: float) -> bool:
    metrics = (record or {}).get("metrics") or {}
    f1 = _safe_rate(_get_primary_metric(metrics))
    threshold = min(1.0, max(0.0, float(success_f1_threshold)))
    if threshold >= 1.0 - 1e-12:
        return f1 >= 1.0 - 1e-12
    return f1 >= threshold



def _build_next_eval_subset(
    *,
    benchmark: HotPotQA,
    eval_mode: str,
    base_eval_indices: Sequence[int],
    current_eval_indices: Sequence[int],
    evaluation_package,
    eval_seed: int,
    iteration: int,
    success_keep_ratio: float,
    success_f1_threshold: float,
    accumulated_success_indices: Optional[Set[int]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Build next eval subset using accumulated success pool across ALL base indices.

    The accumulated_success_indices set is MUTATED in place: newly successful
    indices are added, newly failed indices are removed.  The next subset is
    built as: all current failures + all never-evaluated base indices + 20%
    of the full accumulated success pool.
    """
    base_indices = [int(idx) for idx in (base_eval_indices or [])]
    current_indices = [int(idx) for idx in (current_eval_indices or [])]
    if accumulated_success_indices is None:
        accumulated_success_indices = set()

    if not current_indices:
        stats = {
            "current_size": 0,
            "next_size": len(base_indices),
            "success_count": 0,
            "failure_count": 0,
            "missing_record_count": 0,
            "invalid_example_count": 0,
            "retained_success_count": 0,
            "accumulated_success_pool_size": len(accumulated_success_indices),
            "never_evaluated_count": len(base_indices),
            "success_keep_ratio": float(success_keep_ratio),
            "success_f1_threshold": float(success_f1_threshold),
            "first10_current": [],
            "first10_next": base_indices[:10],
        }
        return base_indices, stats

    records = getattr(evaluation_package, "evaluation_records", {}) or {}
    success_indices: List[int] = []
    failure_indices: List[int] = []
    missing_record_indices: List[int] = []
    invalid_example_indices: List[int] = []

    for idx in current_indices:
        example = benchmark.get_example_by_index(index=int(idx), mode=eval_mode)
        if example is None:
            failure_indices.append(int(idx))
            invalid_example_indices.append(int(idx))
            accumulated_success_indices.discard(int(idx))
            continue
        raw_example_id = benchmark.get_id(example=example)
        record = records.get(raw_example_id)
        if record is None:
            record = records.get(str(raw_example_id))
        if not isinstance(record, dict):
            failure_indices.append(int(idx))
            missing_record_indices.append(int(idx))
            accumulated_success_indices.discard(int(idx))
            continue
        if _is_successful_eval_record(record, success_f1_threshold):
            success_indices.append(int(idx))
            accumulated_success_indices.add(int(idx))
        else:
            failure_indices.append(int(idx))
            accumulated_success_indices.discard(int(idx))

    keep_ratio = min(1.0, max(0.0, float(success_keep_ratio)))
    # Identify base indices never evaluated in any iteration
    evaluated_ever = set(current_indices) | accumulated_success_indices
    never_evaluated = [idx for idx in base_indices if idx not in evaluated_ever]

    # Sample 20% from the FULL accumulated success pool
    full_success_pool = sorted(accumulated_success_indices)
    keep_count = int(len(full_success_pool) * keep_ratio)
    if full_success_pool and keep_ratio > 0.0 and keep_count == 0:
        keep_count = 1
    keep_count = min(len(full_success_pool), keep_count)

    rng = random.Random((int(eval_seed) + 1) * 1009 + int(iteration))
    retained_success_indices = sorted(rng.sample(full_success_pool, k=keep_count)) if keep_count > 0 else []
    next_indices = sorted(dict.fromkeys(list(failure_indices) + never_evaluated + retained_success_indices))
    if not next_indices:
        next_indices = base_indices

    stats = {
        "current_size": len(current_indices),
        "next_size": len(next_indices),
        "success_count": len(success_indices),
        "failure_count": len(failure_indices),
        "missing_record_count": len(missing_record_indices),
        "invalid_example_count": len(invalid_example_indices),
        "retained_success_count": len(retained_success_indices),
        "accumulated_success_pool_size": len(accumulated_success_indices),
        "never_evaluated_count": len(never_evaluated),
        "success_keep_ratio": round(keep_ratio, 4),
        "success_f1_threshold": float(success_f1_threshold),
        "first10_current": current_indices[:10],
        "first10_next": next_indices[:10],
    }
    return next_indices, stats


def _update_sample_f1_tracker(
    tracker: Dict[int, float],
    benchmark: "HotPotQA",
    eval_mode: str,
    eval_indices: Sequence[int],
    evaluation_package,
) -> None:
    """Update per-sample F1 tracker from evaluation records (mutates tracker in place)."""
    records = getattr(evaluation_package, "evaluation_records", {}) or {}
    for idx in (eval_indices or []):
        example = benchmark.get_example_by_index(index=int(idx), mode=eval_mode)
        if example is None:
            tracker[int(idx)] = 0.0
            continue
        raw_id = benchmark.get_id(example=example)
        record = records.get(raw_id) or records.get(str(raw_id))
        if isinstance(record, dict):
            tracker[int(idx)] = _get_primary_metric(record.get("metrics") or {})
        else:
            tracker[int(idx)] = 0.0


def _compute_estimated_full_f1(
    tracker: Dict[int, float],
    base_eval_indices: Sequence[int],
) -> float:
    """Compute estimated full-set F1 using per-sample tracker.

    Samples that have been evaluated keep their last known F1.
    Samples never evaluated are treated as F1=0.
    This makes F1 comparable across iterations with different subsets.
    """
    base_count = len(base_eval_indices or [])
    if base_count == 0:
        return 0.0
    total = sum(tracker.get(int(idx), 0.0) for idx in base_eval_indices)
    return total / base_count


def _evaluation_execution_error_stats(
    *,
    benchmark: HotPotQA,
    eval_mode: str,
    eval_indices: Sequence[int],
    evaluation_package,
) -> Dict[str, Any]:
    eval_indices_list = [int(idx) for idx in (eval_indices or [])]
    records = getattr(evaluation_package, "evaluation_records", {}) or {}
    error_indices: List[int] = []
    error_example_ids: List[str] = []
    missing_record_count = 0
    invalid_example_count = 0
    invalid_record_count = 0

    for idx in eval_indices_list:
        example = benchmark.get_example_by_index(index=int(idx), mode=eval_mode)
        if example is None:
            invalid_example_count += 1
            error_indices.append(int(idx))
            error_example_ids.append(f"idx:{idx}")
            continue

        raw_example_id = benchmark.get_id(example=example)
        record = records.get(raw_example_id)
        if record is None:
            record = records.get(str(raw_example_id))
        metrics = record.get("metrics") if isinstance(record, dict) else None
        if not isinstance(record, dict):
            missing_record_count += 1
            error_indices.append(int(idx))
            error_example_ids.append(str(raw_example_id))
            continue
        if not isinstance(metrics, dict):
            invalid_record_count += 1
            error_indices.append(int(idx))
            error_example_ids.append(str(raw_example_id))
            continue

    error_count = len(error_indices)
    return {
        "subset_size": len(eval_indices_list),
        "error_count": error_count,
        "error_rate": float(error_count / max(1, len(eval_indices_list))),
        "missing_record_count": missing_record_count,
        "invalid_example_count": invalid_example_count,
        "invalid_record_count": invalid_record_count,
        "error_indices": error_indices,
        "error_example_ids": error_example_ids,
        "first5_error_example_ids": error_example_ids[:5],
    }


def _candidate_execution_error_threshold(subset_size: int) -> int:
    return max(3, int(math.ceil(0.05 * max(1, int(subset_size)))))


def _complexity_key(workflow_graph: WorkFlowGraph) -> Tuple[float, float, float]:
    metrics = workflow_complexity_metrics(workflow_graph)
    return (
        float(metrics.get("node_count", 0.0)),
        float(metrics.get("edge_count", 0.0)),
        float(metrics.get("dag_depth", 0.0)),
    )



def _candidate_priority_key(
    workflow_graph: WorkFlowGraph,
    results: Dict[str, Any],
    score_override: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    node_count, edge_count, dag_depth = _complexity_key(workflow_graph)
    primary_score = (
        _safe_rate(score_override)
        if score_override is not None
        else _get_primary_metric(results or {})
    )
    return (
        primary_score,
        -node_count,
        -edge_count,
        -dag_depth,
    )



def _aflow_f1_first_acceptance(
    *,
    candidate_workflow: WorkFlowGraph,
    candidate_results: Dict[str, Any],
    incumbent_workflow: WorkFlowGraph,
    incumbent_results: Dict[str, Any],
    candidate_score: Optional[float] = None,
    incumbent_score: Optional[float] = None,
    score_label: str = "f1",
    tol: float = 1e-12,
) -> Tuple[bool, str]:
    cand_f1 = (
        _safe_rate(candidate_score)
        if candidate_score is not None
        else _get_primary_metric(candidate_results or {})
    )
    inc_f1 = (
        _safe_rate(incumbent_score)
        if incumbent_score is not None
        else _get_primary_metric(incumbent_results or {})
    )
    reason_prefix = str(score_label or "f1").replace(" ", "_")
    if cand_f1 > inc_f1 + tol:
        return True, f"{reason_prefix}_improved"
    if abs(cand_f1 - inc_f1) <= tol and _complexity_key(candidate_workflow) < _complexity_key(incumbent_workflow):
        return True, f"{reason_prefix}_tie_broken_by_complexity"
    return False, "rejected"



def _stringify_context_value(value: Any, max_chars: int = 240) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)
    except Exception:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _prompt_excerpt_for_planner(prompt_text: str, max_head: int = 420, max_tail: int = 420) -> str:
    text = str(prompt_text or "").strip()
    if not text:
        return ""
    if len(text) <= max_head + max_tail + 48:
        return text
    omitted = max(0, len(text) - max_head - max_tail)
    head = text[:max_head].rstrip()
    tail = text[-max_tail:].lstrip()
    return (
        f"{head}\n\n"
        f"[... omitted middle {omitted} chars to preserve both the prompt body and the tail contract ...]\n\n"
        f"{tail}"
    )



def _profile_for_dataset(dataset_name: str) -> str:
    """Return the coarse task profile (qa / math / code) for a dataset key.

    Looks up _DATASET_CONFIG; unknown names fall back to qa, the most
    conservative default (matches the historical hotpotqa behavior).
    """
    cfg = _DATASET_CONFIG.get(str(dataset_name or "").strip().lower())
    if cfg is None:
        return "qa"
    return str(cfg.get("profile") or "qa").strip().lower() or "qa"


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort numeric parse. Strips $, commas, backslashes, whitespace.
    Returns None on failure."""
    if value is None:
        return None
    text = str(value).strip()
    for ch in ("$", ",", "\\", "\u200b"):
        text = text.replace(ch, "")
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _extract_boxed_value(text: Any) -> Optional[str]:
    """Pull the contents of the LAST \\boxed{...} occurrence. Handles nested
    braces. Returns None when no \\boxed{} is present."""
    if not text:
        return None
    s = str(text)
    last = s.rfind("\\boxed{")
    if last < 0:
        return None
    i = last + len("\\boxed{")
    depth = 1
    out = []
    while i < len(s) and depth > 0:
        ch = s[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def _extract_code_block(text: Any) -> str:
    """Return the most likely Python code body. Tries, in order:
    1. a fenced ``` python block, 2. the substring starting at the first
    `def ` line, 3. the raw text. Returns "" only when input is empty."""
    if not text:
        return ""
    s = str(text)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    m = re.search(r"^def\s+\w+", s, re.MULTILINE)
    if m:
        return s[m.start():].strip()
    return s.strip()


def _extract_last_number(text: Any) -> Optional[str]:
    """Return the LAST numeric token (possibly signed / decimal) in `text`,
    or None if none exists. Used as a fallback for math outputs that
    state the answer in prose rather than inside \\boxed{}."""
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return matches[-1] if matches else None


def _check_code_syntax(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _check_signature_match(code: str, entry_point: str) -> Optional[bool]:
    """True iff the code defines `def <entry_point>(...)` at any indent
    level. Returns None when the entry_point is unknown so the planner
    can distinguish 'wrong' from 'unmeasurable'."""
    if not code or not entry_point:
        return None
    pattern = rf"^\s*def\s+{re.escape(str(entry_point).strip())}\s*\("
    return bool(re.search(pattern, code, re.MULTILINE))


def _classify_failure_mode(gold: str, predicted: str, dataset_name: str = "") -> str:
    """Dataset-aware failure label. Profiles emit disjoint vocabularies:

      qa   - empty_answer | too_long | partial_overlap | close_miss | wrong_entity
      math - empty_answer | format_error | close_value | wrong_value
      code - empty_answer | syntax_error | signature_mismatch | wrong_logic

    The vocabulary is intentionally narrow per profile so Step 2 of the
    planner prompt can map each label to a concrete style suggestion.
    """
    p = (predicted or "").strip()
    if not p:
        return "empty_answer"
    profile = _profile_for_dataset(dataset_name)
    if profile == "code":
        code = _extract_code_block(p)
        if not _check_code_syntax(code):
            return "syntax_error"
        # We cannot tell signature_mismatch vs wrong_logic without the
        # entry_point; the diagnostic slice surfaces signature_match so
        # the planner can distinguish in-prompt. Default to wrong_logic.
        return "wrong_logic"
    if profile == "math":
        pb = _extract_boxed_value(p)
        gb = _extract_boxed_value(gold or "")
        # Fall back to last-number extraction for GSM8K-style prose answers.
        pf = _safe_float(pb if pb is not None else _extract_last_number(p))
        gf = _safe_float(gb if gb is not None else _extract_last_number(gold or ""))
        if pf is None and gf is not None:
            return "format_error"
        if pf is not None and gf is not None:
            if abs(pf - gf) < 1e-6:
                # Defensive: this shouldn't be in failure_examples.
                return "wrong_value"
            denom = max(abs(gf), 1e-9)
            if abs(pf - gf) / denom < 0.05:
                return "close_value"
            return "wrong_value"
        return "wrong_value"
    # qa profile (and any unknown -> qa, matching legacy behaviour)
    g = (gold or "").strip().lower()
    pl = p.lower()
    if len(pl.split()) > max(5, len(g.split()) * 3):
        return "too_long"
    if g and g in pl and pl not in g:
        return "partial_overlap"
    if g and (g in pl or pl in g):
        return "close_miss"
    return "wrong_entity"


def _diagnostic_slice(example: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Per-profile diagnostic fields that REPLACE substring-based fault
    localization when gold_match_reliable is False (math, humaneval, mbpp).

    Returns an empty dict for qa profile so the caller can blindly merge.
    """
    profile = _profile_for_dataset(dataset_name)
    predicted = str(example.get("predicted_answer") or "").strip()
    gold = str(example.get("gold_answer") or "").strip()
    if profile == "math":
        pb = _extract_boxed_value(predicted)
        gb = _extract_boxed_value(gold)
        pf = _safe_float(pb if pb is not None else predicted)
        gf = _safe_float(gb if gb is not None else gold)
        return {
            "predicted_boxed": pb,
            "gold_boxed": gb,
            "numeric_match": bool(
                pf is not None and gf is not None and abs(pf - gf) < 1e-6
            ),
        }
    if profile == "code":
        entry = str(example.get("entry_point") or "").strip()
        if not entry and gold.startswith("{"):
            try:
                gold_obj = json.loads(gold)
                if isinstance(gold_obj, dict):
                    entry = str(gold_obj.get("entry_point") or "").strip()
            except Exception:
                entry = ""
        code = _extract_code_block(predicted)
        return {
            "code_block_extracted": bool(code),
            "syntax_ok": _check_code_syntax(code),
            "signature_match": _check_signature_match(code, entry),
        }
    return {}


def _truncate_failure_examples(failure_examples: Sequence[Dict[str, Any]], dataset_name: str = "", limit: int = 12, per_mode_cap: int = 4) -> List[Dict[str, Any]]:
    """Standardize failure examples for the planner prompt.

    Two responsibilities:
      1. Bucket-sample by failure_mode so the planner sees a *distribution*
         of failure types rather than the worst-F1 cases (which usually
         collapse onto one mode on a saturated incumbent). Each mode
         contributes up to per_mode_cap worst-first cases; the total
         is capped at limit.
      2. Project per-example fields the planner needs. Behaviour depends
         on the dataset's gold_match_reliable flag:
           reliable=True  (qa, gsm8k) - keep substring-based
             gold_in_node_output / first_fault_node fields, since gold
             is a short answer string that may legitimately appear inside
             node output text.
           reliable=False (math, humaneval, mbpp) - drop those fields
             and emit the profile-specific _diagnostic_slice instead,
             because substring matching against a long solution /
             serialized dict produces noise that misleads the planner.
    """
    cfg = _DATASET_CONFIG.get(str(dataset_name or "").strip().lower()) or _DATASET_CONFIG["hotpotqa"]
    reliable = bool(cfg.get("gold_match_reliable", False))
    # Group worst-first inputs by failure_mode and round-robin across
    # buckets so a single dominant mode cannot crowd out the long tail.
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    bucket_order: List[str] = []
    for example in list(failure_examples or []):
        gold = str(example.get("gold_answer", "") or "").strip()
        predicted = str(example.get("predicted_answer", "") or "").strip()
        mode = _classify_failure_mode(gold, predicted, dataset_name)
        bucket = buckets.setdefault(mode, [])
        if mode not in bucket_order:
            bucket_order.append(mode)
        if len(bucket) < per_mode_cap:
            bucket.append(example)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < limit and any(buckets[m] for m in bucket_order):
        mode = bucket_order[cursor % len(bucket_order)]
        if buckets[mode]:
            selected.append(buckets[mode].pop(0))
        cursor += 1
    rows: List[Dict[str, Any]] = []
    for example in selected:
        gold = str(example.get("gold_answer", "") or "").strip()
        gold_lower = gold.lower()
        node_outputs: Dict[str, str] = {}
        for name, value in (example.get("node_outputs") or {}).items():
            text = _stringify_context_value(value, max_chars=300)
            if text:
                node_outputs[str(name)] = text
        predicted = str(example.get("predicted_answer", "") or "").strip()
        sample: Dict[str, Any] = {
            "question": _stringify_context_value(example.get("question") or example.get("problem") or example.get("context") or example.get("prompt") or "", max_chars=200),
            "gold_answer": _stringify_context_value(example.get("gold_answer", ""), max_chars=100),
            "predicted_answer": _stringify_context_value(example.get("predicted_answer", ""), max_chars=100),
            "node_outputs": node_outputs,
            "failure_mode": _classify_failure_mode(gold, predicted, dataset_name),
            "f1": round(_get_primary_metric(example), 4),
            "em": round(_safe_rate(example.get("em", 0.0)), 4),
        }
        if reliable:
            gold_in_node_output: Dict[str, bool] = {
                name: bool(gold_lower and gold_lower in text.lower())
                for name, text in node_outputs.items()
            }
            first_fault_node = ""
            prev_had_gold = False
            for name, had_gold in gold_in_node_output.items():
                if prev_had_gold and not had_gold:
                    first_fault_node = name
                    break
                prev_had_gold = had_gold
            if not first_fault_node and gold_lower:
                for name, had_gold in gold_in_node_output.items():
                    if not had_gold:
                        first_fault_node = name
                        break
            sample["gold_in_node_output"] = gold_in_node_output
            sample["first_fault_node"] = first_fault_node
        else:
            sample.update(_diagnostic_slice(example, dataset_name))
        rows.append(sample)
    return rows


def _top_rca_snapshot(target_pool: Sequence[Any]) -> Optional[Dict[str, Any]]:
    if not target_pool:
        return None
    ranked = sorted(target_pool, key=lambda item: int(getattr(item, "target_rank", 10**6)))
    top = ranked[0]
    return {
        "component": getattr(top, "component", ""),
        "subtype": getattr(top, "subtype", ""),
        "node_name": getattr(top, "node_name", ""),
        "failure_prob": round(float(getattr(top, "failure_prob", 0.0)), 6),
        "rca_rank": int(getattr(top, "target_rank", 0)),
    }


def _describe_rca_reliability(pool_mode: str, rca_strength: Dict[str, float]) -> str:
    top1 = _safe_rate((rca_strength or {}).get("top1", 0.0))
    margin = _safe_rate((rca_strength or {}).get("margin", 0.0))
    entropy = _safe_rate((rca_strength or {}).get("norm_entropy", 1.0))
    if str(pool_mode) == "strong_rca":
        return (
            f"RCA signal is currently strong: top1={top1:.4f}, margin={margin:.4f}, entropy={entropy:.4f}. "
"Large margin and lower entropy mean the top diagnosis is relatively reliable."
        )
    return (
        f"RCA signal is currently weak/noisy: top1={top1:.4f}, margin={margin:.4f}, entropy={entropy:.4f}. "
"Prefer conservative edits and lean more on recent history and failure examples."
    )


def _describe_rca_trend(current_snapshot: Optional[Dict[str, Any]], previous_snapshot: Optional[Dict[str, Any]]) -> str:
    if not current_snapshot:
        return "No current RCA target is available."
    current_label = (
        f"{current_snapshot.get('component', '')}.{current_snapshot.get('subtype', '')}"
        f"@{current_snapshot.get('node_name', '')}"
    )
    current_prob = _safe_rate(current_snapshot.get("failure_prob", 0.0))
    if not previous_snapshot:
        return f"No previous RCA snapshot is available. Current top RCA is {current_label} ({current_prob:.4f})."
    previous_label = (
        f"{previous_snapshot.get('component', '')}.{previous_snapshot.get('subtype', '')}"
        f"@{previous_snapshot.get('node_name', '')}"
    )
    previous_prob = _safe_rate(previous_snapshot.get("failure_prob", 0.0))
    if current_label != previous_label:
        return (
            f"Top RCA shifted from {previous_label} ({previous_prob:.4f}) to "
            f"{current_label} ({current_prob:.4f})."
        )
    delta = current_prob - previous_prob
    if delta > 0.02:
        trend = "worsening"
    elif delta < -0.02:
        trend = "improving"
    else:
        trend = "stable"
    return (
        f"Top RCA remains {current_label}; failure probability is {trend} "
        f"({previous_prob:.4f} -> {current_prob:.4f})."
    )


def _compute_planner_directives(
    *,
    modification_history: ModificationHistory,
    workflow_graph: WorkFlowGraph,
    no_improve_count: int,
    blocked_styles: Sequence[str],
    failure_examples_processed: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Promote the few decision-critical signals to top-level directives.

    Without this, the same signals exist deep inside ``modification_history``
    (recent_records, narrative_summary, plain_prohibitions) but get diluted
    by ~5k tokens of surrounding context, so the planner LLM keeps repeating
    a saturated edit family. We surface three things:

      - ``ACTION_REQUIRED``: short imperative sentences (read first by the
        planner prompt). Currently triggers on prompt-edit saturation,
        kind-imbalance, and weak-RCA + repeated-failure combos.
      - ``PROHIBITED_THIS_ITER``: explicit (kind, node) and style bans
        derived from recent failure runs; complements the existing
        ``blocked_styles`` (which is global / per-style only).
      - ``recommended_motifs``: trigger-driven projection of the motif
        library, only populated when run-time symptoms match (see
        ``recommend_motifs``).

    Returns a dict ready to be merged into ``planner_context``.
    """
    records = list(getattr(modification_history, "records", []) or [])
    recent_failures = [r for r in records[-6:] if not getattr(r, "accepted", False)]

    # Per-node consecutive prompt_edit failures from the tail of the history.
    per_node_consecutive: Dict[str, int] = {}
    for record in reversed(records):
        if record.edit_kind != "prompt_edit":
            continue
        node = str(record.target_node_name or "")
        if not node:
            continue
        if record.accepted:
            per_node_consecutive[node] = 0
            break
        per_node_consecutive[node] = per_node_consecutive.get(node, 0) + 1

    # Per (kind) attempts/accepts across the whole run.
    kind_attempts: Dict[str, int] = {}
    kind_accepts: Dict[str, int] = {}
    for record in records:
        kind = record.edit_kind or "unknown"
        kind_attempts[kind] = kind_attempts.get(kind, 0) + 1
        if record.accepted:
            kind_accepts[kind] = kind_accepts.get(kind, 0) + 1

    action_required: List[str] = []
    prohibited: List[Dict[str, Any]] = []

    # 1) Same-node prompt_edit saturation. 3+ failures in a row on the same
    #    node almost always means dedup is squeezing out further variants;
    #    next attempts on this node are wasted slots.
    for node, count in per_node_consecutive.items():
        if count >= 3:
            action_required.append(
                f"Recent {count} consecutive prompt_edit attempts on `{node}` all failed - "
                "do NOT propose another prompt_edit on this node this iteration; "
                "switch the target node OR change edit.kind."
            )
            prohibited.append({"kind": "prompt_edit", "node": node, "reason": f"{count} consecutive failures"})

    # 2) Kind-imbalance: if we have spent most of our budget on prompt_edit
    #    and structure_edit / params_edit are essentially untried, push the
    #    planner to diversify - especially after 2+ stalled iterations.
    prompt_attempts = kind_attempts.get("prompt_edit", 0)
    struct_attempts = kind_attempts.get("structure_edit", 0)
    params_attempts = kind_attempts.get("params_edit", 0)
    if no_improve_count >= 2 and prompt_attempts >= 6 and struct_attempts <= 1 and params_attempts <= 1:
        untried = []
        if struct_attempts <= 1:
            untried.append("structure_edit")
        if params_attempts <= 1:
            untried.append("params_edit")
        action_required.append(
            f"Kinds attempted: prompt_edit={prompt_attempts}, structure_edit={struct_attempts}, "
            f"params_edit={params_attempts}. Reserve at least one slot this iteration for "
            f"{' or '.join(untried)} to break the prompt-only plateau."
        )

    # 2b) Cross-node prompt_edit deadlock. The kind-imbalance check above
    #     only fires after >=6 prompt_attempts, which is too late when the
    #     RCA top is stuck on a Prompt.* projection (e.g. Return.Type ->
    #     Prompt.Contract) and every iteration sends another rejected
    #     prompt_edit. As soon as the *recent tail* shows 3+ consecutive
    #     cross-node prompt_edit rejections AND we have stalled for 2+
    #     iterations, force the planner to allocate at least one slot to
    #     params_edit/structure_edit -- this is the missing consumer of the
    #     fallback_target the RCA projection already computes.
    recent_prompt_edits = [r for r in records[-6:] if r.edit_kind == "prompt_edit"]
    recent_prompt_failures = [r for r in recent_prompt_edits if not getattr(r, "accepted", False)]
    if (
        len(recent_prompt_edits) >= 3
        and len(recent_prompt_failures) == len(recent_prompt_edits)
        and no_improve_count >= 2
    ):
        action_required.append(
            f"Recent {len(recent_prompt_edits)} prompt_edit attempts (across all nodes) "
            "all failed while no_improve_count is high; the RCA top is likely overfit "
            "to a Prompt.* projection. REQUIRED this iteration: at least one params_edit "
            "or structure_edit candidate -- consider the RCA fallback_target "
            "(typically Params.Parse) instead of another prompt_edit on the same dimension."
        )
        prohibited.append({
            "kind": "prompt_edit_only_batch",
            "reason": (
                f"{len(recent_prompt_edits)} consecutive cross-node prompt_edit failures; "
                "must include at least one non-prompt edit this iteration"
            ),
        })

    # 3) Echo blocked_styles into prohibitions so the LLM sees a single
    #    place to look. Cheap; helps when the planner ignores the deep
    #    blocked_styles field.
    for style in blocked_styles or []:
        prohibited.append({"style": str(style), "reason": "blocked due to repeated failure"})

    # 4) Failure-mode distribution feeds motif recommendation.
    failure_mode_distribution: Dict[str, int] = {}
    for sample in failure_examples_processed or []:
        mode = str(sample.get("failure_mode", "") or "")
        if mode:
            failure_mode_distribution[mode] = failure_mode_distribution.get(mode, 0) + 1

    node_count = len(getattr(workflow_graph, "nodes", []) or [])
    recommended_names = recommend_motifs(
        no_improve_count=int(no_improve_count),
        node_count=int(node_count),
        edit_kind_attempts=kind_attempts,
        edit_kind_accepts=kind_accepts,
        failure_mode_distribution=failure_mode_distribution,
    )
    recommended_motifs = render_motifs_for_prompt(recommended_names)

    if recommended_motifs and not action_required:
        action_required.append(
            "Consider one of the recommended_motifs below; current trajectory matches its symptom pattern."
        )

    return {
        "ACTION_REQUIRED": action_required,
        "PROHIBITED_THIS_ITER": prohibited,
        "recommended_motifs": recommended_motifs,
        "edit_kind_stats": {
            "attempts": kind_attempts,
            "accepts": kind_accepts,
        },
    }


def _node_edge_context(workflow_graph: WorkFlowGraph, node_name: str) -> Dict[str, List[str]]:
    upstream: List[str] = []
    downstream: List[str] = []
    for edge in getattr(workflow_graph, "edges", []) or []:
        source = getattr(edge, "source", None) if not isinstance(edge, dict) else edge.get("source")
        target = getattr(edge, "target", None) if not isinstance(edge, dict) else edge.get("target")
        if target == node_name and source:
            upstream.append(str(source))
        if source == node_name and target:
            downstream.append(str(target))
    return {
        "upstream": sorted(set(upstream)),
        "downstream": sorted(set(downstream)),
    }


def _workflow_graph_view(workflow_graph: WorkFlowGraph) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    for node in list(getattr(workflow_graph, "nodes", []) or []):
        edge_context = _node_edge_context(workflow_graph, node.name)
        nodes.append(
            {
                "name": node.name,
                "description": getattr(node, "description", ""),
                "inputs": [
                    {
                        "name": getattr(inp, "name", ""),
                        "type": getattr(inp, "type", ""),
                        "description": getattr(inp, "description", ""),
                    }
                    for inp in list(getattr(node, "inputs", []) or [])
                ],
                "outputs": [
                    {
                        "name": getattr(out, "name", ""),
                        "type": getattr(out, "type", ""),
                        "description": getattr(out, "description", ""),
                    }
                    for out in list(getattr(node, "outputs", []) or [])
                ],
                "upstream": edge_context.get("upstream", []),
                "downstream": edge_context.get("downstream", []),
            }
        )
    edges: List[Dict[str, str]] = []
    for edge in list(getattr(workflow_graph, "edges", []) or []):
        source = getattr(edge, "source", None) if not isinstance(edge, dict) else edge.get("source")
        target = getattr(edge, "target", None) if not isinstance(edge, dict) else edge.get("target")
        if source and target:
            edges.append({"source": str(source), "target": str(target)})
    edges.sort(key=lambda item: (item.get("source", ""), item.get("target", "")))
    return {"nodes": nodes, "edges": edges}


def _recent_modification_lines(modification_history: ModificationHistory, node_name: str, max_records: int = 3) -> List[str]:
    lines: List[str] = []
    for record in reversed(getattr(modification_history, "records", [])):
        if (record.target_node_name or "") != node_name:
            continue
        lines.append(
            f"- Iter {record.iteration}: {record.edit_kind}/{record.style}/{record.op_family or '-'} -> "
            f"accepted={bool(record.accepted)}, estimated_full_f1_delta={float(record.estimated_full_f1_delta):+.4f}, "
            f"utility_delta={float(record.utility_delta):+.4f}, "
            f"status={record.materialization_status or record.validation_status or 'unknown'}"
        )
        if len(lines) >= max_records:
            break
    if not lines:
        return ["- No recent modification outcomes for this node."]
    return list(reversed(lines))

def _planner_context(
    workflow_graph: WorkFlowGraph,
    baseline_package,
    baseline_utility: float,
    target_pool: Sequence[Any],
    pool_mode: str,
    rca_strength: Dict[str, float],
    best_workflow: WorkFlowGraph,
    best_results: Dict[str, float],
    prompt_history: PromptHistory,
    modification_history: ModificationHistory,
    previous_rca_snapshot: Optional[Dict[str, Any]] = None,
    baseline_estimated_full_f1: Optional[float] = None,
    workflow_goal: Optional[str] = None,
    dataset_name: str = "",
    no_improve_count: int = 0,
) -> Dict[str, Any]:
    root_cause_summary = _legacy._summarize_root_cause_distribution(baseline_package.root_causes)
    evidence_support = _legacy._summarize_evidence_support(baseline_package.evidences)
    prompt_history_summary = {
        # node_name: prompt_history.format_history_for_llm(node_name, max_records=3)
        # for node_name in prompt_history.get_all_node_names()
    }
    modification_summary = ""#modification_history.summarize_for_llm()
    current_rca_snapshot = _top_rca_snapshot(target_pool)
    current_node_prompts = {}
    node_domain_guidance = {}
    dataset_format_rules = {}
    _wf_goal = getattr(workflow_graph, "goal", "") or ""
    try:
        _end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        _end_nodes = set()
    for node in (workflow_graph.nodes or []):
        prompt_text = _legacy._get_node_primary_prompt(node) or ""
        if prompt_text:
            current_node_prompts[node.name] = prompt_text
        try:
            _bullets = _optimizer_domain_guidance(_wf_goal, node, dataset_name=dataset_name) or []
        except Exception:
            _bullets = []
        if _bullets:
            node_domain_guidance[node.name] = _bullets
        if node.name in _end_nodes:
            _rules = _legacy._extract_dataset_format_rules(_wf_goal, is_end_node=True, dataset_name=dataset_name)
            if _rules:
                dataset_format_rules[node.name] = _rules
    style_summary = modification_history.style_summary()
    # Track worst effective_delta per prompt style so we can rank blockables by severity.
    _blocked_scores = {}
    for _key, _bucket in style_summary.items():
        _edit_kind, _style, _op_family, _sv = _key
        if _edit_kind != "prompt_edit":
            continue
        _attempts = float(_bucket.get("attempts", 0.0))
        _mean_score_delta = float(_bucket.get("mean_estimated_full_f1_delta", 0.0))
        _mean_utility_delta = float(_bucket.get("mean_utility_delta", 0.0))
        _effective_delta = _mean_score_delta if abs(_mean_score_delta) > 1e-12 else _mean_utility_delta
        # A1: faster style-block trigger. Original (>=3 attempts AND
        # delta<-0.01) let a regressing style burn 3 iterations before being
        # shelved; on single-node workflows that meant the optimizer never
        # got to try a different style. Block as soon as one attempt produces
        # a clear regression OR two attempts tie the incumbent. The
        # downstream _max_blockable cap still guarantees at least
        # _MIN_AVAILABLE_PROMPT_STYLES (3) styles remain usable.
        _block = False
        if _attempts >= 1 and _effective_delta < -0.005:
            _block = True
        elif _attempts >= 2 and abs(_effective_delta) < 0.005:
            _block = True
        if _block:
            _prev = _blocked_scores.get(_style)
            _eff_for_rank = _effective_delta if _effective_delta < 0 else -1e-6
            if _prev is None or _eff_for_rank < _prev:
                _blocked_scores[_style] = _eff_for_rank

    # Use the explicit dataset_name to select applicable prompt styles.
    _dataset_styles = set(_styles_for_dataset(dataset_name))

    # Cap blocked_styles so the planner always retains a usable prompt-style vocabulary.
    # Blocking too many causes the LLM planner to either repeat a blocked style or
    # hallucinate invalid style names, which manifests as planner_failure iterations.
    # Keep at least _MIN_AVAILABLE_PROMPT_STYLES usable at all times within the
    # current profile vocabulary; when more would be blocked, retain only the worst
    # offenders.
    _MIN_AVAILABLE_PROMPT_STYLES = 3
    _pool_size = max(len(_dataset_styles), _MIN_AVAILABLE_PROMPT_STYLES)
    _max_blockable = max(0, _pool_size - _MIN_AVAILABLE_PROMPT_STYLES)
    # Only block styles that are inside the current profile vocabulary; styles
    # outside the profile are already unavailable and must not occupy a block slot.
    _ranked = sorted(
        ((style, delta) for style, delta in _blocked_scores.items() if style in _dataset_styles),
        key=lambda item: item[1],
    )  # most negative first
    _blocked = [style for style, _ in _ranked[:_max_blockable]]
    _allowed_prompt_styles = sorted(s for s in _dataset_styles if s not in set(_blocked))
    _allowed_prompt_style_help = _style_vocab_descriptions(_allowed_prompt_styles)

    failure_examples_processed = _truncate_failure_examples(
        getattr(baseline_package, "failure_examples", []),
        dataset_name=dataset_name,
        limit=12,
    )
    directives = _compute_planner_directives(
        modification_history=modification_history,
        workflow_graph=workflow_graph,
        no_improve_count=no_improve_count,
        blocked_styles=_blocked,
        failure_examples_processed=failure_examples_processed,
    )

    # ---- C4: saturation escalation -------------------------------------------
    # When the planner has plateaued for 2+ iterations and the projected RCA
    # targets are Prompt-only (i.e. no Structure/Params signal), the LLM has
    # no real choice but to keep proposing prompt_edit candidates, which is
    # exactly the failure mode we observed on HumanEval. Append a synthetic
    # Structure target so the planner can legitimately propose a structure_edit.
    # The fallback target points at __STRUCTURE__ with subtype=Coverage which
    # the sanitizer accepts for INSERT_NODE/DELETE_NODE structure variants.
    _supported_targets = _project_rca_targets_to_supported_targets(target_pool)
    _saturation_escalation = False
    if no_improve_count >= 2:
        _component_set = {str(t.get("component", "")).strip() for t in _supported_targets}
        if _component_set and _component_set <= {"Prompt"}:
            _next_rank = max([int(t.get("rca_rank", 0) or 0) for t in _supported_targets] or [0]) + 1
            _supported_targets.append({
                "rca_rank": _next_rank,
                "component": "Structure",
                "subtype": "Coverage",
                "node_name": "__STRUCTURE__",
                "failure_prob": 0.0,
                "source": "saturation_escalation",
                "edge_source": None,
                "edge_target": None,
                "diagnostic_component": "Structure",
                "diagnostic_subtype": "Coverage",
                "diagnostic_node_name": "__STRUCTURE__",
                "projection_reason": "saturation_escalation_inject_structure",
                "fallback_target": None,
            })
            _saturation_escalation = True

    return {
        # ---- Decision-critical directives (read these FIRST) -------------
        "ACTION_REQUIRED": directives["ACTION_REQUIRED"],
        "PROHIBITED_THIS_ITER": directives["PROHIBITED_THIS_ITER"],
        "recommended_motifs": directives["recommended_motifs"],
        "edit_kind_stats": directives["edit_kind_stats"],
        # ---- Vocabulary -------------------------------------------------
        "dataset_name": dataset_name,
        "blocked_styles": _blocked,
        "allowed_prompt_styles": _allowed_prompt_styles,
        "allowed_prompt_style_help": _allowed_prompt_style_help,
        "baseline": {
            "metrics": _round_nested(dict(baseline_package.results)),
            "estimated_full_f1": round(_safe_rate(baseline_estimated_full_f1), 6),
            "selection_metric": "estimated_full_f1",
            "utility": round(float(baseline_utility), 6),
            "obs_coverage": round(float(getattr(baseline_package, "obs_coverage", 0.0)), 6),
        },
        "workflow_summary": _workflow_summary(workflow_graph),
        "workflow_graph_view": _workflow_graph_view(workflow_graph),
        "rca": (
            {
                "pool_mode": "no_rca",
                "strength": {},
                "targets": [],
                "component_mass": {},
                "subtype_mass": {},
                "top_rows": [],
                "reliability_summary": "RCA disabled for ablation (wo_rca). Pick any node freely.",
                "trend_summary": "",
                "top_snapshot": {},
            }
            if DISABLE_RCA
            else {
                "pool_mode": pool_mode,
                "strength": _round_nested(rca_strength),
                "targets": _planner_targets(target_pool),
                "component_mass": _round_nested(root_cause_summary.get("component_mass", {})),
                "subtype_mass": _round_nested(dict(list(root_cause_summary.get("subtype_mass", {}).items())[:10])),
                "top_rows": _round_nested(root_cause_summary.get("top_rows", [])[:10]),
                "reliability_summary": _describe_rca_reliability(pool_mode, rca_strength),
                "trend_summary": _describe_rca_trend(current_rca_snapshot, previous_rca_snapshot),
                "top_snapshot": _round_nested(current_rca_snapshot or {}),
            }
        ),
        "supported_targets": _supported_targets,
        "saturation_escalation": bool(_saturation_escalation),
        "evidence_support": _round_nested(evidence_support),
        "node_stats": _round_nested(baseline_package.node_stats),
        "failure_examples": failure_examples_processed,
        "best_workflow": _best_workflow_summary(best_workflow, best_results),
        "prompt_history": prompt_history_summary,
        "modification_history": modification_summary,
        "current_node_prompts": current_node_prompts,
        "node_domain_guidance": node_domain_guidance,
        "dataset_format_rules": dataset_format_rules,
        "allowed_param_fields": sorted(_ALLOWED_PARAM_FIELDS),
    }


def _workflow_safety_status(workflow_graph: WorkFlowGraph) -> Dict[str, Any]:
    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow_graph)
    prompt_errors = list(_legacy._validate_workflow_prompt_templates(workflow_graph)[:10])
    structure_valid, structure_reasons, _ = _legacy._validate_workflow_structure_for_evolution(workflow_graph)
    return {
        "ok": (not prompt_errors) and bool(structure_valid),
        "prompts_repaired": int(repaired_cnt),
        "parse_mode_changed": int(mode_cnt),
        "prompt_errors": prompt_errors,
        "structure_valid": bool(structure_valid),
        "structure_reasons": list(structure_reasons),
    }


def _planner_prompt(planner_context: Dict[str, Any], planner_candidate_count: int) -> str:
    _allowed_styles_ctx = planner_context.get("allowed_prompt_styles") or sorted(_PROMPT_STYLES)
    _allowed_styles_help = planner_context.get("allowed_prompt_style_help") or {}
    _dataset_name = str(planner_context.get("dataset_name", "")).strip().lower()
    # C5: when the optimizer has plateaued (saturation_escalation=True), force
    # the planner to diversify away from prompt_edit so we can break out of
    # the local optimum. The bonus INSERT injection in optimize() runs in
    # parallel as a safety net, but having the LLM generate at least one
    # non-prompt candidate yields more diverse trial workflows.
    _saturation_escalation = bool(planner_context.get("saturation_escalation"))
    if _saturation_escalation:
        _saturation_banner = (
            '' + chr(10) + '## SATURATION ESCALATION (this iteration)' + chr(10) + 'The optimizer has not improved for 2+ consecutive iterations and the' + chr(10) + 'current RCA targets are Prompt-only. Repeated prompt_edit candidates' + chr(10) + 'are unlikely to escape this plateau. AT LEAST ONE candidate this batch' + chr(10) + 'MUST be `edit.kind = structure_edit` OR `edit.kind = params_edit`.' + chr(10) + 'Use the synthetic Structure target appended to `supported_targets`' + chr(10) + '(node_name = "__STRUCTURE__", subtype = "Coverage") as the target' + chr(10) + 'for that candidate. The other candidate may still be a prompt_edit.' + chr(10) + ''
        )
    else:
        _saturation_banner = ""

    # ---- Step 2: dataset-specific field descriptions ----
    _step2 = _build_step2_for_dataset(_dataset_name)

    # ---- Step 3: cross-profile rules. Profile-specific symptoms are
    # already mapped to styles in Step 2, so this block only handles
    # the three rules that fire across qa / math / code uniformly.
    _allowed_set = set(_allowed_styles_ctx)
    _step3_lines = []
    if "SCHEMA_HARDEN" in _allowed_set:
        _step3_lines.append(
            "SCHEMA_HARDEN \u2014 failure_mode = empty_answer, OR return.type_ok < 0.6, OR params.format_parseable < 0.6. Target: the node with low format_parseable.")
    if "BINDING_REPAIR" in _allowed_set:
        _step3_lines.append(
            "BINDING_REPAIR \u2014 prompt.input_binding < 0.7, OR a required input variable is not referenced by `{placeholder}`. Target: the node with low input_binding.")
    if "DEDUP_SIMPLIFY" in _allowed_set:
        _step3_lines.append(
            "DEDUP_SIMPLIFY \u2014 recent repeated MODIFY edits on the same node all failed with no improvement. Target: the most-edited node.")
    _step3_numbered = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(_step3_lines))
    _step3 = f"""## Step 3 Choose a Style
Step 2 already maps each dataset-specific failure_mode (and diagnostic fields) to a concrete style \u2014 follow that mapping first.

For symptoms not covered there, apply these cross-profile rules:

{_step3_numbered}

Every `edit.style` for prompt_edit MUST be one of `allowed_prompt_styles` (exact spelling). If no rule fires, pick the most relevant style from `allowed_prompt_styles` based on the failure examples and target the highest-ranked RCA target node."""

    # ---- Allowed styles display ----
    _styles_sorted = sorted(str(s) for s in _allowed_styles_ctx)
    if _allowed_styles_help:
        _allowed_styles_display = "\n".join(
            f"  - {name}: {_allowed_styles_help.get(name, '').strip()}" if _allowed_styles_help.get(name) else f"  - {name}"
            for name in _styles_sorted
        )
    else:
        _allowed_styles_display = "  " + ", ".join(_styles_sorted)

    schema_example = {
        "candidates": [
            {
                "candidate_id": "cand_1",
                "target": {"component": "Prompt", "subtype": "Binding", "node_name": "organize_evidence", "rca_rank": 1},
                "edit": {
                    "kind": "prompt_edit",
                    "style": "BINDING_REPAIR",
                    "op_family": "MODIFY",
                    "new_prompt": "(the FULL rewritten prompt text for this node, ready to replace current_node_prompts[node_name])",
                    "instructions": ["Changed: strengthened explicit use of upstream evidence variables."],
                    "param_changes": {},
                    "structure_variant": "",
                },
                "rationale": "Prompt binding is a current RCA focus.",
                "history_reference": "Avoid repeating the most recent failed DELETE attempt.",
                "expected_effect": "Improve evidence usage and downstream answer quality.",
            },
            {
                "candidate_id": "cand_2",
                "target": {"component": "Structure", "subtype": "Topology", "node_name": "__STRUCTURE__", "rca_rank": 2},
                "edit": {
                    "kind": "structure_edit",
                    "new_workflow": {
                        "nodes": [
                            {
                                "name": "problem_framing",
                                "description": "Parse the goal and restate the core question in structured form.",
                                "inputs": [{"name": "goal", "type": "string", "description": "The task goal / problem statement.", "required": True}],
                                "outputs": [{"name": "structured_question", "type": "string", "description": "Parsed problem specification.", "required": True}]
                            },
                            {
                                "name": "solution_derivation",
                                "description": "Derive the solution step by step from the structured question.",
                                "inputs": [{"name": "structured_question", "type": "string", "description": "From problem_framing.", "required": True}],
                                "outputs": [{"name": "solution_trace", "type": "string", "description": "Numbered derivation with intermediate results.", "required": True}]
                            },
                            {
                                "name": "answer_finalization",
                                "description": "Produce the final answer in the exact required output format.",
                                "inputs": [{"name": "solution_trace", "type": "string", "description": "From solution_derivation.", "required": True}],
                                "outputs": [{"name": "answer", "type": "string", "description": "Final answer for the workflow.", "required": True}]
                            }
                        ],
                        "edges": [
                            {"source": "problem_framing", "target": "solution_derivation"},
                            {"source": "solution_derivation", "target": "answer_finalization"}
                        ]
                    },
                    "instructions": ["Split the monolithic solver into framing, derivation, and finalization stages to reduce error propagation."],
                    "param_changes": {},
                },
                "rationale": "Separating framing/derivation/finalization localizes failures and lets each stage be prompt-tuned independently.",
                "history_reference": "Previous monolithic attempt conflated derivation with final formatting.",
                "expected_effect": "Cleaner data flow and better answer-format adherence.",
            },
        ]
    }
    return f"""
## Role
You are the workflow optimization planner. Diagnose failures from diagnostic evidence and propose targeted, well-reasoned edits to the workflow.
Primary objective: improve performance of the workflow on the fixed evaluation set.

{_saturation_banner}## Output Format
Return ONLY a valid JSON object with exactly one key: `candidates` (a list of exactly {planner_candidate_count} objects).
Each candidate must have: `candidate_id`, `target`, `edit`, `rationale`, `history_reference`, `expected_effect`.
- `target.component` must be one of: Prompt, Params, Structure. Never Return or Edge (those are diagnosis-only).
- `target.component`, `target.subtype`, `target.node_name` must come from `supported_targets`.
- `target.rca_rank` references the corresponding row in `rca.targets`.

## Step 1 Read optimization history
Read these fields BEFORE making any decisions:
- `baseline.estimated_full_f1`: performance score for the current workflow.
- `modification_history.narrative_summary`: recent edit quality and trajectory. 
- `modification_history.risky_patterns` + `failure_patterns`: patterns to avoid.
- `workflow_graph_view.nodes` + `workflow_graph_view.edges`: explicit node I/O and current topology.
- `current_node_prompts`: the FULL current prompt text for each workflow node. For prompt_edit candidates, rewrite the target node prompt directly based on this.
  Do not propose adding something that is already present in the prompt.

{_step2}

{_step3}

For Params edits: use MORE_TOKENS when not_truncated < 0.8; use LOWER_TEMPERATURE when output is unstable.
When rca.pool_mode = weak_rca: use the RCA as a reference; you should decide on the specific changes yourself.
When rca.pool_mode = strong_rca: trust the top RCA target.

## Step 4 Provide two candidate workflows
- Candidate 1: highest-confidence fix addressing the root fault node.
- Candidate 2: A node that differs from Candidate 1 but that you believe should be modified.
- Do NOT repeat a recently failed edit unless your rationale clearly explains what is different this time.
- `allowed_prompt_styles` lists the prompt styles you may use this iteration. 
This is the authoritative vocabulary: every `edit.style` for prompt_edit candidates MUST be one of these strings, exactly spelled.
  Do NOT invent styles such as OUTPUT_PRECISION, OUTPUT_SCOPE_*, or any string not in `allowed_prompt_styles`; candidates with unrecognized styles will be rejected.

### prompt_edit: Direct Prompt Rewriting
For every prompt_edit candidate, you MUST provide `edit.new_prompt`: the COMPLETE rewritten prompt text for the target node.
- Read the current prompt from `current_node_prompts[target_node_name]`.
- Rewrite it according to your chosen style and diagnosis. The new prompt must be complete and ready to use as-is.
- Preserve ALL input variable placeholders (e.g. {{goal}}, {{evidence}}) and output variable names from the original prompt.
- If `node_domain_guidance` contains entries for this node, incorporate those rules.
- If `dataset_format_rules` contains entries for this node (end nodes), preserve the Output Format section verbatim.
- `edit.instructions` should briefly summarize what you changed (1-2 sentences), for logging purposes.

### structure_edit: Direct Workflow Rewriting
For every structure_edit candidate, you MUST provide `edit.new_workflow` containing the COMPLETE modified workflow:
- Read the current topology from `workflow_graph_view.nodes` and `workflow_graph_view.edges`.
- Modify the structure as needed: add, remove, merge, split, or rewire nodes and edges.
- Output ALL nodes (including unchanged ones) and ALL edges in `new_workflow`.
- For unchanged nodes, keep the exact same name, description, inputs, and outputs.
- For new or modified nodes, provide clear descriptions and properly typed inputs/outputs.
- Ensure input/output names chain correctly across edges (upstream node output name = downstream node input name).
- Topology (node count, role distribution, degree of specialization) is
  YOUR decision. Pick whatever shape best solves the benchmark. The only
  HARD constraints the validator enforces are:
    * at least one initial node has a required input named `goal`,
    * at least one terminal node has a required output named `answer`,
    * every edge's source output name matches one of the target's input names,
    * the graph is a connected DAG (>= node_count - 1 edges) and has <= 12 nodes,
    * the workflow solves the runtime task, not meta-plans another workflow.
  Anything else (2 nodes or 7, linear or branching, whether you include a
  verifier stage) is up to your judgment and will be scored empirically.
- `edit.instructions` should briefly summarize what structural change was made.

## Available Edit Options
- Prompt styles (must pick from this list):\n{_allowed_styles_display}
- Params styles: {sorted(_PARAM_STYLES)}
- Prompt op families: {sorted(_ALLOWED_OP_FAMILIES)}
- Structure edit: provide `edit.new_workflow` with the complete modified workflow (nodes + edges). See schema example.
- Param fields: {sorted(_ALLOWED_PARAM_FIELDS)}

## Schema Example
```json
{_json(schema_example)}
```

## Planner Context
```json
{_json(planner_context)}
```

Return ONLY the JSON object with the `candidates` key.
""".strip()
def _planner_repair_prompt(original_prompt: str, raw_output: str, error_message: str) -> str:
    return f"""
The previous planner output was invalid.
Validation error:
{error_message}

Invalid output:
```text
{raw_output}
```

Repair it. Follow the original prompt exactly and return ONLY one valid JSON object.
Original prompt:
```text
{original_prompt}
```
""".strip()


#{_step3} - Every additional candidate: an independent backup with a different target node OR a clearly different style.  prefer conservative Prompt or Params edits; avoid Structure edits.

## Observation Scores (0-1, higher is better)
# - prompt.input_binding: upstream variable placeholders correctly used
# - prompt.output_contract: output variable names clearly specified
# - prompt.grounded: prompt stays tied to provided evidence
# - params.not_truncated: output not cut off by token limit
# - params.format_parseable: output parseable in expected format
# - return.type_ok: output matches declared type
# - return.content_ok: output non-empty and meaningful
# - return.task_ok: output correctly solves the node task (primary success signal)


def _normalize_param_changes(param_changes: Any, candidate_id: str) -> Dict[str, Any]:
    if not isinstance(param_changes, dict) or not param_changes:
        raise ValueError(f"candidate {candidate_id}: params_edit requires non-empty param_changes")
    normalized: Dict[str, Any] = {}
    bad_fields = [str(key) for key in param_changes if str(key) not in _ALLOWED_PARAM_FIELDS]
    if bad_fields:
        raise ValueError(f"candidate {candidate_id}: invalid param fields {bad_fields}")
    for name, raw_value in param_changes.items():
        field = str(name)
        if field == "parse_mode":
            parse_mode = str(raw_value or "").strip().lower()
            if parse_mode == "strict_json":
                parse_mode = "json"
            if parse_mode not in _ALLOWED_PARSE_MODES:
                raise ValueError(f"candidate {candidate_id}: unsupported parse_mode={raw_value}")
            normalized[field] = parse_mode
        elif field == "temperature":
            normalized[field] = float(raw_value)
        elif field == "top_p":
            normalized[field] = float(raw_value)
        elif field == "max_tokens":
            normalized[field] = int(raw_value)
        else:
            normalized[field] = raw_value
    return normalized


def _normalize_candidate(
    item: Dict[str, Any],
    index: int,
    valid_ranks: Sequence[int],
    forced_target: Optional[Dict[str, Any]] = None,
    blocked_styles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    candidate_id = str(item.get("candidate_id") or f"cand_{index + 1}").strip() or f"cand_{index + 1}"
    target = item.get("target") or {}
    edit = item.get("edit") or {}
    if not isinstance(target, dict) or not isinstance(edit, dict):
        raise ValueError(f"candidate {candidate_id}: target/edit must be objects")

    if forced_target is not None:
        component = _normalize_component(forced_target.get("component"))
        subtype = _normalize_target_subtype(component, forced_target.get("subtype"))
        node_name = str(forced_target.get("node_name") or "").strip() or ("__STRUCTURE__" if component == "Structure" else "")
        rca_rank = int(forced_target.get("rca_rank", 0) or 0)
    else:
        component = _normalize_component(target.get("component"))
        subtype = _normalize_target_subtype(component, target.get("subtype"))
        node_name = str(target.get("node_name") or "").strip() or ("__STRUCTURE__" if component == "Structure" else "")
        try:
            rca_rank = int(target.get("rca_rank"))
        except Exception as exc:
            raise ValueError(f"candidate {candidate_id}: target.rca_rank must be integer") from exc
    if valid_ranks and rca_rank not in set(int(x) for x in valid_ranks):
        raise ValueError(f"candidate {candidate_id}: target.rca_rank={rca_rank} is not in the current RCA target pool")
    if component not in {"Prompt", "Params", "Structure"}:
        raise ValueError(f"candidate {candidate_id}: unsupported target.component={component}")
    if not subtype:
        raise ValueError(f"candidate {candidate_id}: target.subtype is required")
    if component != "Structure" and not node_name:
        raise ValueError(f"candidate {candidate_id}: target.node_name is required")

    kind = str(edit.get("kind") or "").strip().lower()
    style = str(edit.get("style") or "").strip().upper()
    op_family = str(edit.get("op_family") or "").strip().upper()
    instructions = [str(x).strip() for x in (edit.get("instructions") or []) if str(x or "").strip()]
    structure_variant = str(edit.get("structure_variant") or "").strip()
    param_changes = edit.get("param_changes") or {}

    expected_kind = {"Prompt": "prompt_edit", "Params": "params_edit", "Structure": "structure_edit"}[component]
    if kind != expected_kind:
        # Auto-correct kind: planner LLMs (especially mid-tier) frequently emit
        # the wrong `edit.kind` label even though the target.component is
        # pinned by RCA / slot-refill. Keep component fixed and salvage the
        # candidate when the payload itself is substantively consistent
        # with the expected kind. Only reject when both label AND payload
        # are unusable for this component.
        _new_workflow = edit.get("new_workflow") if isinstance(edit, dict) else None
        _has_struct_payload = (
            (isinstance(_new_workflow, dict) and bool(_new_workflow.get("nodes")))
            or (style in _STRUCTURE_STYLE_VARIANTS and structure_variant in _STRUCTURE_STYLE_VARIANTS.get(style, set()))
        )
        _has_prompt_payload = bool(str(edit.get("new_prompt") or "").strip()) or bool(instructions)
        _has_params_payload = bool(param_changes)
        _payload_ok = {
            "structure_edit": _has_struct_payload,
            "prompt_edit": _has_prompt_payload,
            "params_edit": _has_params_payload,
        }[expected_kind]
        if not _payload_ok:
            raise ValueError(
                f"candidate {candidate_id}: {component} target requires edit.kind={expected_kind}, "
                f"got {kind} and edit payload also lacks the fields needed for {expected_kind}"
            )
        kind = expected_kind

    new_prompt = ""
    if kind == "prompt_edit":
        new_prompt = str(edit.get("new_prompt") or "").strip()
        if style not in _PROMPT_STYLES:
            raise ValueError(
                f"candidate {candidate_id}: invalid prompt style={style}. "
                f"Must be one of {sorted(_PROMPT_STYLES)}"
            )
        _blocked_styles = {str(item).strip().upper() for item in (blocked_styles or []) if str(item).strip()}
        if style in _blocked_styles:
            raise ValueError(
                f"candidate {candidate_id}: style={style} is in blocked_styles (confirmed negative primary-score impact). "
                f"Choose a different style."
            )
        if op_family not in _ALLOWED_OP_FAMILIES:
            raise ValueError(f"candidate {candidate_id}: prompt_edit requires valid op_family")
        if not new_prompt and not instructions:
            raise ValueError(f"candidate {candidate_id}: prompt_edit requires new_prompt or instructions")
        normalized_param_changes = {}
    elif kind == "params_edit":
        if style not in _PARAM_STYLES:
            raise ValueError(f"candidate {candidate_id}: invalid params style={style}")
        normalized_param_changes = _normalize_param_changes(param_changes, candidate_id)
        op_family = ""
        instructions = []
        structure_variant = ""
    elif kind == "structure_edit":
        new_workflow = edit.get("new_workflow")
        if isinstance(new_workflow, dict) and new_workflow.get("nodes"):
            # New path: planner provides complete workflow spec
            instructions = [str(x).strip() for x in (edit.get("instructions") or []) if str(x or "").strip()]
        else:
            # Legacy path: style+variant validation
            new_workflow = None
            if style not in _STRUCTURE_STYLE_VARIANTS:
                raise ValueError(f"candidate {candidate_id}: invalid structure style={style}")
            if structure_variant not in _STRUCTURE_STYLE_VARIANTS[style]:
                raise ValueError(f"candidate {candidate_id}: invalid structure_variant={structure_variant} for style={style}")
        op_family = ""
        normalized_param_changes = {}
    else:
        raise ValueError(f"candidate {candidate_id}: unsupported edit.kind={kind}")
    return {
        "candidate_id": candidate_id,
        "target": {"component": component, "subtype": subtype, "node_name": node_name, "rca_rank": rca_rank},
        "edit": {
            "kind": kind,
            "style": style,
            "op_family": op_family,
            "new_prompt": new_prompt,
            "instructions": instructions,
            "param_changes": normalized_param_changes,
            "structure_variant": structure_variant,
            "new_workflow": new_workflow if kind == "structure_edit" else None,
        },
        "rationale": str(item.get("rationale") or "").strip(),
        "history_reference": str(item.get("history_reference") or "").strip(),
        "expected_effect": str(item.get("expected_effect") or "").strip(),
    }


_CANDIDATE_LIST_KEYS = (
    "candidates",
    "plan",
    "proposals",
    "edits",
    "recommendations",
    "actions",
    "items",
    "results",
)


def _coerce_candidate_list(parsed: Any) -> Optional[List[Dict[str, Any]]]:
    """Best-effort extraction of a candidate list from parsed JSON.

    The legacy planner contract was a strict ``{\"candidates\": [...]}`` shape
    but smaller / cheaper LLMs frequently produce alternative envelopes:
    ``{\"plan\": [...]}``, ``{\"edits\": [...]}``, a bare list, or even a
    single candidate object. This helper accepts all of them so the same
    iteration is not silently failed for a syntactic quirk.
    """
    if isinstance(parsed, list):
        items = [item for item in parsed if isinstance(item, dict)]
        return items or None
    if not isinstance(parsed, dict):
        return None
    for key in _CANDIDATE_LIST_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, dict)]
            if items:
                return items
        if isinstance(value, dict):
            return [value]
    # Single-candidate response: dict with edit/target keys at the top level.
    if any(k in parsed for k in ("edit", "target", "candidate_id")):
        return [parsed]
    # Last resort: scan for any list-of-dicts value.
    for value in parsed.values():
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, dict) and ("edit" in item or "target" in item)]
            if items:
                return items
    return None


def _parse_candidate_items(raw_text: str, minimum_count: int = 1) -> List[Dict[str, Any]]:
    parsed = _legacy._extract_first_json_value(raw_text, required_keys=["candidates"])
    if parsed is None:
        # Try without the required-keys hint -- captures bare lists, alt keys,
        # and single-object envelopes that the strict pass would reject.
        parsed = _legacy._extract_first_json_value(raw_text)
    candidates = _coerce_candidate_list(parsed)
    if candidates is None:
        raise ValueError("planner output does not contain a recognisable candidate list (tried `candidates`, `plan`, `proposals`, `edits`, `recommendations`, `actions`, bare list, single object)")
    if len(candidates) < minimum_count:
        raise ValueError(f"planner output must contain at least {minimum_count} candidates")
    return list(candidates)


def _dump_planner_debug(stage: str, prompt: str, raw_output: str, error: str) -> Optional[str]:
    """Persist the full LLM exchange so future parser failures are 1-minute-debuggable.

    Writes to data/planner_debug/<ts>_<stage>.txt with prompt + raw output
    + error message. Failure-tolerant: any I/O error is swallowed (we never
    want diagnostics to break the optimisation loop).
    """
    try:
        import os, time
        debug_dir = os.path.join("data", "planner_debug")
        os.makedirs(debug_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(debug_dir, f"{ts}_{stage}.txt")
        nl = chr(10)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== STAGE: {stage} ==={nl}")
            f.write(f"=== ERROR: {error} ==={nl}")
            raw_text = raw_output or ""
            f.write(f"=== RAW_OUTPUT (len={len(raw_text)}) ==={nl}")
            f.write(raw_text)
            prompt_text = prompt or ""
            f.write(f"{nl}{nl}=== PROMPT (len={len(prompt_text)}) ==={nl}")
            f.write(prompt_text)
        return path
    except Exception:
        return None


def _request_candidate_batch_with_repair(llm, prompt: str, minimum_count: int, planner_repair_rounds: int) -> Tuple[List[Dict[str, Any]], str, int]:
    raw_output = _llm_text(llm, prompt)
    parse_failures = 0
    try:
        return _parse_candidate_items(raw_output, minimum_count), raw_output, parse_failures
    except Exception as first_error:
        parse_failures += 1
        if planner_repair_rounds <= 0:
            preview = (raw_output or "")[:200].replace(chr(10), ' ')
            dump_path = _dump_planner_debug("initial", prompt, raw_output or "", str(first_error))
            raise ValueError(f"planner output invalid: {first_error}; raw_head={preview!r}; dump={dump_path}") from first_error
        repaired_output = _llm_text(llm, _planner_repair_prompt(prompt, raw_output, str(first_error)))
        try:
            return _parse_candidate_items(repaired_output, minimum_count), repaired_output, parse_failures
        except Exception as second_error:
            raw_preview = (raw_output or "")[:200].replace(chr(10), ' ')
            repaired_preview = (repaired_output or "")[:200].replace(chr(10), ' ')
            dump_initial = _dump_planner_debug("initial", prompt, raw_output or "", str(first_error))
            dump_repair = _dump_planner_debug("after_repair", prompt, repaired_output or "", str(second_error))
            raise ValueError(
                f"planner output invalid after repair: {first_error}; {second_error}; "
                f"raw_head={raw_preview!r}; repair_head={repaired_preview!r}; "
                f"dump_initial={dump_initial}; dump_repair={dump_repair}"
            ) from second_error


_FALLBACK_STYLE_BY_COMPONENT = {
    "Prompt": ("prompt_edit", "SCHEMA_HARDEN", "MODIFY"),
    "Params": ("params_edit", "LOWER_TEMPERATURE", ""),
}


_FALLBACK_PROMPT_INSTRUCTIONS = (
    (
        "Re-emphasise the output contract literally as stated in the goal: required output names, exact format, and any verbatim sentences. Do not invent new sections.",
        "Before producing the answer, restate the entry-point or terminal-output name from the goal in one short reasoning line so the model anchors on it.",
    ),
    (
        "Add an explicit self-check step: before emitting the final answer, confirm every required output field has a value of the correct type and length.",
        "If any required field is missing, regenerate with that field filled rather than skipping it.",
    ),
    (
        "Reduce verbosity: produce only the fields named in the contract; do NOT include reasoning prose, prefixes, or markdown decoration around the final answer.",
        "When the goal asks for a single token / number / boolean, output exactly that token with no surrounding text.",
    ),
    (
        "Cite the upstream evidence (the previous node's output) verbatim before producing the answer; do not invent new facts.",
        "If the upstream evidence is empty or malformed, return the explicit fallback value defined by the goal instead of guessing.",
    ),
)


def _synthesize_fallback_candidates(
    planner_context: Dict[str, Any],
    minimum_count: int,
) -> List[Dict[str, Any]]:
    """Build deterministic candidates when the LLM planner fails.

    Multi-tier policy designed for small/cheap planner LLMs and tight search
    spaces (e.g. single-node code workflows where every prompt rewrite quickly
    fingerprints the same workflow):

    * Tier A -- ``prompt_edit`` with a style drawn from
      ``allowed_prompt_styles - blocked_styles``, rotated by iteration. Never
      emits a blocked style.
    * Tier B -- ``params_edit`` with rotated temperature; always semantically
      valid because ``LOWER_TEMPERATURE`` style is permanently available.
    * Tier C -- ``structure_edit`` (legacy ``INSERT_NODE`` /
      ``insert_reasoning_chain_stage``). Adds a node so the post-canonical
      workflow fingerprint is materially different from any prompt-only edit.

    The tier selection cycles by ``iteration_salt`` so iter 1/2/3/... emit
    different families, and skips a tier if its preconditions cannot be met
    (e.g. all prompt styles blocked, no Params target, etc.). Goal: ALWAYS
    return >=1 semantically valid candidate so the iter is never wasted.
    """
    targets = list(planner_context.get("supported_targets") or [])
    blocked_styles = {str(s).strip().upper() for s in (planner_context.get("blocked_styles") or []) if str(s).strip()}
    # The upstream context uses ``allowed_prompt_styles`` (see line ~1552).
    # ``available_prompt_styles`` is kept as a compat fallback only.
    allowed_styles_raw = list(
        planner_context.get("allowed_prompt_styles")
        or planner_context.get("available_prompt_styles")
        or []
    )
    allowed_styles = [str(s).strip().upper() for s in allowed_styles_raw if str(s or "").strip()]
    if not allowed_styles:
        # Last-ditch: use the global vocab so we never end up with an empty pool.
        try:
            allowed_styles = sorted(_PROMPT_STYLES.keys())
        except Exception:
            allowed_styles = ["SCHEMA_HARDEN", "BINDING_REPAIR", "GROUNDING_HARDEN", "CHAIN_SYNTHESIS", "DEDUP_SIMPLIFY"]
    effective_styles = [s for s in allowed_styles if s and s not in blocked_styles]
    iteration_salt = int(planner_context.get("iteration", 0) or 0)
    exhausted_nodes = {str(n).strip() for n in (planner_context.get("exhausted_prompt_nodes") or []) if str(n or "").strip()}

    # Partition targets by component so each tier picks the right anchor.
    prompt_targets = [t for t in targets if _normalize_component(t.get("component")) == "Prompt"]
    params_targets = [t for t in targets if _normalize_component(t.get("component")) == "Params"]
    structure_targets = [t for t in targets if _normalize_component(t.get("component")) == "Structure"]

    def _pick_target(pool: List[Dict[str, Any]], skip_exhausted: bool) -> Optional[Dict[str, Any]]:
        if not pool:
            return None
        if skip_exhausted and exhausted_nodes:
            usable = [t for t in pool if str(t.get("node_name") or "").strip() not in exhausted_nodes]
            if usable:
                return usable[iteration_salt % len(usable)]
        return pool[iteration_salt % len(pool)]

    candidates: List[Dict[str, Any]] = []
    fallback_idx = 0

    # ---- Build the tier ordering. Cycle through (A, B, C) by iteration so
    # consecutive iters propose different families even when the LLM fails
    # consistently. The first tier whose preconditions fire wins; the rest are
    # tried in order if it cannot produce a candidate.
    tier_order = ["A", "B", "C"]
    tier_offset = iteration_salt % 3
    rotated_tiers = tier_order[tier_offset:] + tier_order[:tier_offset]

    def _build_prompt_edit() -> Optional[Dict[str, Any]]:
        if not effective_styles:
            return None
        target = _pick_target(prompt_targets, skip_exhausted=True)
        if target is None:
            return None
        node_name = str(target.get("node_name") or "").strip()
        rca_rank = int(target.get("rca_rank", 0) or 0)
        chosen_style = effective_styles[iteration_salt % len(effective_styles)]
        instr_pool = _FALLBACK_PROMPT_INSTRUCTIONS
        instructions = list(instr_pool[iteration_salt % len(instr_pool)])
        return {
            "candidate_id": f"fallback_synth_iter{iteration_salt}_p{fallback_idx + 1}",
            "target": {
                "component": "Prompt",
                "subtype": _normalize_target_subtype("Prompt", target.get("subtype")) or "Contract",
                "node_name": node_name,
                "rca_rank": rca_rank,
            },
            "edit": {
                "kind": "prompt_edit",
                "style": chosen_style,
                "op_family": "MODIFY",
                "instructions": instructions,
            },
            "rationale": f"fallback prompt_edit (iter={iteration_salt}, style={chosen_style})",
            "history_reference": "",
            "expected_effect": "rotate prompt style on failing terminal node",
        }

    def _build_params_edit() -> Optional[Dict[str, Any]]:
        # params_edit only when RCA actually surfaced a Params target.
        # If we synth a Params edit on a Prompt target, the downstream
        # sanitize step (line ~2624) projects target.component back to
        # Prompt via supported_targets/rank_map, then the validator rejects
        # the candidate ("Prompt target requires edit.kind=prompt_edit").
        # No fallback -- let the tier rotation move on to A or C.
        target = _pick_target(params_targets, skip_exhausted=False)
        if target is None:
            return None
        node_name = str(target.get("node_name") or "").strip()
        rca_rank = int(target.get("rca_rank", 0) or 0)
        temp_steps = (0.0, 0.15, 0.3, 0.5)
        return {
            "candidate_id": f"fallback_synth_iter{iteration_salt}_q{fallback_idx + 1}",
            "target": {
                "component": "Params",
                "subtype": _normalize_target_subtype("Params", target.get("subtype")) or "Parse",
                "node_name": node_name,
                "rca_rank": rca_rank,
            },
            "edit": {
                "kind": "params_edit",
                "style": "LOWER_TEMPERATURE",
                "op_family": "",
                "param_changes": {"temperature": temp_steps[iteration_salt % len(temp_steps)]},
            },
            "rationale": f"fallback params_edit (iter={iteration_salt})",
            "history_reference": "",
            "expected_effect": "reduce sampling variance on the failing node",
        }

    def _build_structure_edit() -> Optional[Dict[str, Any]]:
        # Last resort: legacy structure_edit using the INSERT_NODE style.
        # Variants: insert_evidence_organizer, insert_reasoning_chain_stage.
        try:
            variants = list(_STRUCTURE_STYLE_VARIANTS.get("INSERT_NODE") or [])
        except Exception:
            variants = []
        if not variants:
            return None
        chosen_variant = variants[iteration_salt % len(variants)]
        # Same anti-projection rule as Params: only emit structure_edit when
        # a Structure target actually exists in the rank_map. Otherwise
        # sanitize will project rca_rank back onto a Prompt target and the
        # validator will reject the kind/component mismatch.
        if not structure_targets:
            return None
        target = structure_targets[iteration_salt % len(structure_targets)]
        rca_rank = int(target.get("rca_rank", 0) or 0)
        return {
            "candidate_id": f"fallback_synth_iter{iteration_salt}_s{fallback_idx + 1}",
            "target": {
                "component": "Structure",
                "subtype": "Topology",
                "node_name": "__STRUCTURE__",
                "rca_rank": rca_rank,
            },
            "edit": {
                "kind": "structure_edit",
                "style": "INSERT_NODE",
                "op_family": "",
                "structure_variant": chosen_variant,
            },
            "rationale": f"fallback structure_edit (iter={iteration_salt}, variant={chosen_variant})",
            "history_reference": "",
            "expected_effect": "add a stage to break a converged single-node fingerprint",
        }

    builders = {"A": _build_prompt_edit, "B": _build_params_edit, "C": _build_structure_edit}
    target_min = max(1, minimum_count)
    for tier in rotated_tiers:
        if len(candidates) >= target_min:
            break
        entry = builders[tier]()
        if entry is not None:
            candidates.append(entry)
            fallback_idx += 1

    # Final guarantee: if everything above somehow produced nothing, force a
    # structure_edit with the first available variant. This should be
    # unreachable in practice but provides an absolute floor.
    if not candidates:
        # Absolute floor: a prompt_edit on the highest-rank target always
        # survives sanitize/validate because the projection is identity
        # (rank already maps to whatever supported_targets say it is).
        floor_target = None
        if prompt_targets:
            floor_target = prompt_targets[0]
        elif targets:
            floor_target = targets[0]
        if floor_target is not None and effective_styles:
            floor_node = str(floor_target.get("node_name") or "").strip()
            floor_rank = int(floor_target.get("rca_rank", 0) or 0)
            floor_subtype = _normalize_target_subtype("Prompt", floor_target.get("subtype")) or "Contract"
            floor_style = effective_styles[iteration_salt % len(effective_styles)]
            floor_instr = list(_FALLBACK_PROMPT_INSTRUCTIONS[iteration_salt % len(_FALLBACK_PROMPT_INSTRUCTIONS)])
            candidates.append({
                "candidate_id": f"fallback_synth_iter{iteration_salt}_floor",
                "target": {"component": "Prompt", "subtype": floor_subtype, "node_name": floor_node, "rca_rank": floor_rank},
                "edit": {"kind": "prompt_edit", "style": floor_style, "op_family": "MODIFY", "instructions": floor_instr},
                "rationale": f"absolute floor prompt_edit (iter={iteration_salt}, style={floor_style})",
                "history_reference": "",
                "expected_effect": "guarantee at least one materialisable candidate",
            })
    return candidates


def _plan_candidates_with_llm(llm, planner_context: Dict[str, Any], planner_candidate_count: int, planner_repair_rounds: int) -> Tuple[List[Dict[str, Any]], str, int]:
    prompt = _planner_prompt(planner_context, planner_candidate_count)
    try:
        return _request_candidate_batch_with_repair(llm, prompt, minimum_count=1, planner_repair_rounds=planner_repair_rounds)
    except Exception as exc:
        # LLM-agnostic safety net: when the planner LLM repeatedly produces
        # unparseable output (small/cheap models often refuse to emit JSON
        # when baseline accuracy is already high), synthesise candidates
        # locally from the RCA target pool so the iter is not wasted.
        synthetic = _synthesize_fallback_candidates(planner_context, minimum_count=1)
        if not synthetic:
            raise
        try:
            from ..core.logging import logger as _planner_logger
            _planner_logger.warning(
                f"planner LLM failed ({exc}); falling back to {len(synthetic)} "
                f"deterministic candidate(s) synthesised from RCA targets."
            )
        except Exception:
            pass
        synth_payload = json.dumps({"candidates": synthetic, "_synthetic_fallback": True}, ensure_ascii=False)
        return synthetic, synth_payload, 2


def _planner_candidate_brief(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "target": dict((candidate.get("target") or {})),
        "edit": {
            "kind": ((candidate.get("edit") or {}).get("kind") or ""),
            "style": ((candidate.get("edit") or {}).get("style") or ""),
            "op_family": ((candidate.get("edit") or {}).get("op_family") or ""),
            "structure_variant": ((candidate.get("edit") or {}).get("structure_variant") or ""),
        },
    }


def _planner_slot_refill_prompt(planner_context: Dict[str, Any], accepted_candidates: Sequence[Dict[str, Any]], invalid_slots: Sequence[Dict[str, Any]], missing_count: int) -> str:
    accepted_view = [_planner_candidate_brief(candidate) for candidate in accepted_candidates]
    # Surface ALL prior rejections (do not truncate to 8) so the LLM sees every
    # mistake it has already made this iteration and can avoid repeating them.
    invalid_list = list(invalid_slots or [])
    invalid_view = [
        {
            "slot_index": int(item.get("slot_index", 0) or 0),
            "candidate_id": item.get("candidate_id", ""),
            "reason": item.get("reason", ""),
            "rejection_type": item.get("rejection_type", ""),
        }
        for item in invalid_list
    ]
    # Diversification rules surfaced from rejection history. Mid-tier LLMs
    # ignore `blocked_styles` when it is buried inside the planner context
    # JSON, so we lift it (and per-slot duplicate fingerprints) into
    # explicit MUST-NOT rules at the top of the prompt.
    blocked_styles_raw = planner_context.get("blocked_styles") or []
    blocked_styles_list = sorted({str(s).strip().upper() for s in blocked_styles_raw if str(s or "").strip()})
    duplicate_fps: List[str] = []
    for item in invalid_list:
        if item.get("rejection_type") == "duplicate_workflow":
            reason = str(item.get("reason") or "")
            if "fp=" in reason:
                fp = reason.split("fp=", 1)[1].rstrip(") ").strip()
                if fp:
                    duplicate_fps.append(fp)
    duplicate_seen = bool(duplicate_fps)
    diversification_rules = []
    if blocked_styles_list:
        diversification_rules.append(
            "- BLOCKED prompt styles (DO NOT use any of these as edit.style for prompt_edit candidates, "
            "they were already shown to lower the primary score): " + ", ".join(blocked_styles_list)
        )
    if duplicate_fps:
        diversification_rules.append(
            "- The following candidate workflows fingerprinted IDENTICAL to previously-tried workflows "
            "and were rejected as duplicates: " + ", ".join(sorted(set(duplicate_fps))) + ". "
            "Your replacement candidates MUST introduce a materially different change "
            "(different node added/removed, different edge topology, different prompt rewrite, "
            "or a different param value) so the resulting workflow fingerprint is NEW."
        )
    exhausted_nodes_raw = planner_context.get("exhausted_prompt_nodes") or []
    exhausted_nodes_list = sorted({str(n).strip() for n in exhausted_nodes_raw if str(n or "").strip()})
    if exhausted_nodes_list:
        diversification_rules.append(
            "- EXHAUSTED prompt-edit target nodes (multiple style rewrites on each of these "
            "converged to the SAME post-repair workflow, so another prompt_edit on them would "
            "be rejected again as duplicate): " + ", ".join(exhausted_nodes_list) + ". "
            "DO NOT propose prompt_edit targeting any of these nodes. Choose a DIFFERENT "
            "target node, OR switch the candidate to structure_edit / params_edit."
        )
    diversification_block = (
        "\nDiversification constraints (HIGHEST PRIORITY):\n" + "\n".join(diversification_rules) + "\n"
        if diversification_rules else ""
    )
    schema_example = {
        "candidates": [
            {
                "candidate_id": "cand_refill_1",
                "target": {"component": "Prompt", "subtype": "Grounding", "node_name": "answer_generation", "rca_rank": 1},
                "edit": {
                    "kind": "prompt_edit",
                    "style": "GROUNDING_HARDEN",
                    "op_family": "MODIFY",
                    "new_prompt": "(the FULL rewritten prompt text for this node)",
                    "instructions": ["Changed: require the node to answer only with evidence already extracted upstream."],
                    "param_changes": {},
                    "structure_variant": "",
                },
                "rationale": "Replacement candidate for an invalid or failed slot.",
                "history_reference": "Avoid the rejected slot patterns listed below.",
                "expected_effect": "Produce a legal, materially different candidate.",
            }
        ]
    }
    return f"""
You are refilling missing planner candidate slots.
Return ONLY one JSON object.
{diversification_block}
Rules:
- Return exactly {missing_count} replacement candidates.
- Only generate the missing slots; do not repeat or modify the already accepted candidates.
- Every candidate MUST choose its executable target from `supported_targets`.
- Do NOT output `Return` or `Edge` as `target.component`.
- Candidate ids must be unique and must not reuse any already accepted candidate_id.

REQUIRED FIELDS PER edit.kind (failing any rule below = candidate WILL be rejected again):

A. prompt_edit:
   - `edit.style` MUST be one of the strings in `allowed_prompt_styles` (see planner context).
     Do NOT invent new style names. Do NOT leave it empty.
   - `edit.style` MUST NOT be in `blocked_styles` (see planner context).
   - `edit.new_prompt` MUST contain the COMPLETE rewritten prompt text (not a diff,
     not "(unchanged)", not a placeholder). Read the current prompt from
     `current_node_prompts` and produce a materially different rewrite.
   - `edit.op_family` MUST be one of the values listed in planner context.

B. structure_edit (preferred path):
   - `edit.new_workflow` MUST be an object with two keys:
       * `nodes`: a list of >=1 distinct nodes; each node MUST have `name`,
         `description`, `inputs` (list of {{name,type,description,required}}),
         `outputs` (list of {{name,type,description,required}}). Up to 12 nodes.
       * `edges`: a list of {{source,target}} pairs forming a connected DAG.
   - Required execution-layer invariants (enforced by validator):
       * at least one initial (in-degree=0) node has a required input named `goal`,
       * at least one terminal (out-degree=0) node has a required output named `answer`,
       * every edge's source output name matches one of the target's input names,
       * the graph is connected (>= node_count - 1 edges).
   - Topology shape (node count, stage responsibilities, whether to add a verifier
     or parallel branches) is YOUR decision and will be judged by empirical score.
   - When you use this path, `edit.style` and `edit.structure_variant` MAY be empty.

C. structure_edit (legacy style+variant path, only if you do NOT provide new_workflow):
   - `edit.style` MUST be a non-empty string from the allowed structure styles
     listed in planner context (`allowed_structure_styles` if present).
   - `edit.structure_variant` MUST be a non-empty variant valid for that style.

D. params_edit:
   - `edit.style` MUST be one of the strings in `allowed_params_styles`.
   - `edit.param_changes` MUST be a non-empty mapping of {{field: new_value}} pairs.

After producing a candidate, MENTALLY check it against the rejection reasons below.
If your new candidate would trigger the same reason again, REWRITE it before emitting.

Already accepted candidates:
```json
{_json(accepted_view)}
```

Rejected slots to replace:
```json
{_json(invalid_view)}
```

Schema example:
```json
{_json(schema_example)}
```

Planner context:
```json
{_json(planner_context)}
```

Return ONLY the JSON object.
""".strip()


def _refill_planner_candidates_with_llm(
    llm,
    planner_context: Dict[str, Any],
    accepted_candidates: Sequence[Dict[str, Any]],
    invalid_slots: Sequence[Dict[str, Any]],
    missing_count: int,
    planner_repair_rounds: int,
) -> Tuple[List[Dict[str, Any]], str, int]:
    prompt = _planner_slot_refill_prompt(planner_context, accepted_candidates, invalid_slots, missing_count)
    try:
        return _request_candidate_batch_with_repair(llm, prompt, minimum_count=1, planner_repair_rounds=planner_repair_rounds)
    except Exception as exc:
        # LLM-agnostic safety net: when the refill LLM call fails to emit a
        # parseable candidate list, return an empty result so the caller's
        # `if not refill_valid: break` clause exits the refill loop cleanly
        # rather than letting the exception kill the whole iteration. The
        # initial planner call already tries deterministic synthesis on
        # failure, so reaching here means we already have at least one
        # synthetic candidate to evaluate this iter.
        try:
            from ..core.logging import logger as _refill_logger
            _refill_logger.warning(
                f"refill planner LLM failed ({exc}); returning empty refill batch "
                f"so the iter can proceed with already-accepted candidates."
            )
        except Exception:
            pass
        empty_payload = json.dumps({"candidates": [], "_refill_llm_failed": True, "_error": str(exc)}, ensure_ascii=False)
        return [], empty_payload, 1


def _sanitize_planner_candidates(
    raw_candidates: Sequence[Dict[str, Any]],
    planner_context: Dict[str, Any],
    planner_candidate_count: int,
    existing_candidate_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    supported_targets = list(planner_context.get("supported_targets") or [])
    rank_map = {int(item.get("rca_rank", 0) or 0): dict(item) for item in supported_targets if int(item.get("rca_rank", 0) or 0) > 0}
    signature_map: Dict[Tuple[str, str, str], List[int]] = {}
    for item in supported_targets:
        signature_map.setdefault(_target_signature(item.get("component"), item.get("subtype"), item.get("node_name")), []).append(int(item.get("rca_rank", 0) or 0))

    valid_candidates: List[Dict[str, Any]] = []
    invalid_slots: List[Dict[str, Any]] = []
    projection_events: List[Dict[str, Any]] = []
    used_ids = {str(item).strip() for item in (existing_candidate_ids or []) if str(item).strip()}

    for index, item in enumerate(list(raw_candidates or [])):
        candidate_id = ""
        raw_component = ""
        raw_subtype = ""
        raw_node_name = ""
        raw_rank: Optional[int] = None
        if isinstance(item, dict):
            candidate_id = str(item.get("candidate_id") or f"cand_{index + 1}").strip() or f"cand_{index + 1}"
            target = item.get("target") or {}
            if isinstance(target, dict):
                raw_component = _normalize_component(target.get("component"))
                raw_subtype = _normalize_target_subtype(raw_component, target.get("subtype"))
                raw_node_name = str(target.get("node_name") or "").strip() or ("__STRUCTURE__" if raw_component == "Structure" else "")
                try:
                    raw_rank = int(target.get("rca_rank"))
                except Exception:
                    raw_rank = None
        else:
            candidate_id = f"cand_{index + 1}"

        if candidate_id in used_ids:
            invalid_slots.append(
                {
                    "slot_index": index + 1,
                    "candidate_id": candidate_id,
                    "reason": f"candidate_id={candidate_id} duplicates an existing accepted/generated candidate",
                    "rejection_type": "duplicate_candidate_id",
                }
            )
            continue
        if not isinstance(item, dict):
            invalid_slots.append(
                {
                    "slot_index": index + 1,
                    "candidate_id": candidate_id,
                    "reason": "candidate must be a JSON object",
                    "rejection_type": "invalid_candidate_object",
                }
            )
            continue

        supported_target = rank_map.get(int(raw_rank or 0)) if raw_rank is not None else None
        recovered_by = ""
        if supported_target is None:
            # Strategy 1: exact signature match
            signature = _target_signature(raw_component, raw_subtype, raw_node_name)
            candidate_ranks = [rank for rank in signature_map.get(signature, []) if rank in rank_map]
            if len(candidate_ranks) == 1:
                raw_rank = candidate_ranks[0]
                supported_target = rank_map[raw_rank]
                recovered_by = "supported_target_signature"
            elif len(candidate_ranks) > 1:
                # Multiple matches with same signature: pick the highest-ranked
                raw_rank = min(candidate_ranks)  # rank 1 is highest priority
                supported_target = rank_map[raw_rank]
                recovered_by = "supported_target_signature_best_rank"
            else:
                # Strategy 2: match by component alone (relaxed recovery)
                component_ranks = [
                    rank for rank, target in rank_map.items()
                    if _normalize_component(target.get("component")) == raw_component
                ]
                if len(component_ranks) >= 1:
                    raw_rank = min(component_ranks)
                    supported_target = rank_map[raw_rank]
                    recovered_by = "component_fallback"
                else:
                    # Strategy 3: just pick the first available target
                    if rank_map:
                        raw_rank = min(rank_map.keys())
                        supported_target = rank_map[raw_rank]
                        recovered_by = "any_target_fallback"
                    else:
                        invalid_slots.append(
                            {
                                "slot_index": index + 1,
                                "candidate_id": candidate_id,
                                "reason": f"candidate {candidate_id}: target.rca_rank={raw_rank} is invalid and no recovery possible (empty rank_map)",
                                "rejection_type": "invalid_rca_rank",
                            }
                        )
                        continue

        try:
            normalized = _normalize_candidate(
                item,
                index,
                [int(supported_target.get("rca_rank", 0) or 0)],
                forced_target=supported_target,
                blocked_styles=planner_context.get("blocked_styles", []),
            )
        except Exception as exc:
            invalid_slots.append(
                {
                    "slot_index": index + 1,
                    "candidate_id": candidate_id,
                    "reason": str(exc),
                    "rejection_type": "semantic_invalid",
                }
            )
            continue

        projection_needed = (
            raw_component != normalized["target"]["component"]
            or raw_subtype != normalized["target"]["subtype"]
            or raw_node_name != normalized["target"]["node_name"]
            or (raw_rank is None or raw_rank != normalized["target"]["rca_rank"])
            or bool(recovered_by)
        )
        if projection_needed:
            projection = {
                "candidate_id": normalized["candidate_id"],
                "rca_rank": int(normalized["target"]["rca_rank"]),
                "from": {
                    "component": raw_component,
                    "subtype": raw_subtype,
                    "node_name": raw_node_name,
                    "rca_rank": raw_rank,
                },
                "to": dict(normalized["target"]),
                "projection_reason": supported_target.get("projection_reason", ""),
                "recovered_by": recovered_by,
            }
            normalized["planner_original_target"] = projection["from"]
            normalized["target_projection"] = projection
            projection_events.append(projection)

        used_ids.add(normalized["candidate_id"])
        valid_candidates.append(normalized)
        if len(valid_candidates) >= planner_candidate_count:
            break

    info = {
        "projection_events": projection_events,
        "supported_target_count": len(supported_targets),
    }
    return valid_candidates, invalid_slots, info

def _rewrite_prompt_instruction(
    node,
    old_prompt: str,
    candidate: Dict[str, Any],
    node_stats: Dict[str, Any],
    prompt_history: PromptHistory,
    modification_history: ModificationHistory,
    anchor_workflow: WorkFlowGraph,
    failure_examples: Optional[List[Dict[str, Any]]] = None,
    workflow_goal: str = "",
    is_end_node: bool = False,
    dataset_name: str = "",
) -> str:
    input_desc = "\n".join(
        f"- {inp.name} ({inp.type}): {inp.description}" for inp in getattr(node, "inputs", []) or []
    ) or "- None"
    output_desc = "\n".join(
        f"- {out.name} ({out.type}): {out.description}" for out in getattr(node, "outputs", []) or []
    ) or "- None"
    instructions = "\n".join(f"- {line}" for line in candidate["edit"].get("instructions", [])) or "- None"
    node_level_stats = _round_nested((node_stats or {}).get(node.name, {})) if isinstance(node_stats, dict) else {}
    prompt_obs = node_level_stats.get("prompt", {})
    params_obs = node_level_stats.get("params", {})
    return_obs = node_level_stats.get("return", {})
    history_text = prompt_history.format_history_for_llm(node.name, max_records=3)
    modification_text = "\n".join(_recent_modification_lines(modification_history, node.name, max_records=3))
    edge_context = _node_edge_context(anchor_workflow, node.name)
    upstream = ", ".join(edge_context.get("upstream", [])) or "None"
    downstream = ", ".join(edge_context.get("downstream", [])) or "None"
    # Dataset-specific final-answer format rules extracted from the workflow
    # goal. Only populated for end nodes tied to a known benchmark; otherwise
    # "". When non-empty, we embed it in the rewriter prompt so the LLM cannot
    # strip dataset-critical rules (\boxed{}, bare number, raw Python, etc.)
    # during iteration. _repair_prompt_contract applies the same recovery as a
    # safety net for the rewritten output.
    _format_rules_block = _legacy._extract_dataset_format_rules(
        workflow_goal, is_end_node=is_end_node, dataset_name=dataset_name
    )
    if _format_rules_block:
        _dataset_rules_section = (
            "Dataset-specific final-answer format rules (the rewritten prompt MUST\n"
            "contain a '# Output Format' section enforcing every line below verbatim;\n"
            "do not paraphrase, do not drop any bullet):\n"
            + _format_rules_block.rstrip()
            + "\n\n"
        )
        _dataset_constraint_line = (
            "- Dataset format rules are provided above. The rewritten prompt MUST"
            " contain a '# Output Format' section that reproduces every bullet"
            " verbatim. Do not paraphrase, drop, or reorder bullets.\n"
        )
    else:
        _dataset_rules_section = ""
        _dataset_constraint_line = ""

    # Per-node domain guidance: imperative, role-aware, benchmark-specific rules
    # that tell the rewriter LLM what this node's output really needs to contain
    # (e.g. DROP evidence extraction = verbatim spans, both endpoint dates; MATH
    # answer extraction = \boxed{}, reduced fraction; HotpotQA answer = minimal
    # noun phrase). Without this, the rewriter only sees generic prompt-style
    # names (GROUNDING_HARDEN, ANSWER_NORMALIZE) and produces generic edits that
    # get rejected during evaluation, which manifests as the optimizer thrashing
    # on one node with many versions and no F1 improvement.
    try:
        _domain_bullets = _optimizer_domain_guidance(workflow_goal, node, dataset_name=dataset_name) or []
    except Exception:
        _domain_bullets = []
    if _domain_bullets:
        _domain_lines = "\n".join(f"- {b}" for b in _domain_bullets)
        _domain_section = (
            "Domain-specific guidance for this node (imperative rules derived from\n"
            "the workflow goal and the node's role; the rewritten prompt MUST\n"
            "incorporate these rules in its '### Instructions' section, either\n"
            "verbatim or closely paraphrased while preserving every numbered step\n"
            "and example):\n"
            + _domain_lines
            + "\n\n"
        )
        _domain_constraint_line = (
            "- Domain-specific guidance is provided above. The rewritten prompt MUST"
            " incorporate every bullet in its '### Instructions' section; do not"
            " drop numbered steps or worked examples.\n"
        )
    else:
        _domain_section = ""
        _domain_constraint_line = ""
    return f"""
You are rewriting a workflow node prompt according to a planner decision.

Node name: {node.name}
Node description: {getattr(node, 'description', '')}
Inputs:
{input_desc}
Outputs:
{output_desc}
Direct neighbors:
- upstream: {upstream}
- downstream: {downstream}

Current prompt:
```text
{old_prompt}
```

Diagnostic evidence for this node:
- prompt observations: {_json(prompt_obs)}
- params observations: {_json(params_obs)}
- return observations: {_json(return_obs)}

Score interpretation (0-1, higher is better):
- prompt.input_binding: whether upstream variables are referenced correctly.
- prompt.output_contract: whether the output variable names are clearly specified.
- prompt.grounded: whether the prompt stays tied to provided evidence.
- params.not_truncated: whether the node is avoiding token-cutoff failures.
- params.format_parseable: whether outputs can be parsed reliably.
- return.type_ok: whether the output matches the declared type.
- return.content_ok: whether the output is non-empty and meaningful.
- return.task_ok: whether the output is actually solving the node task.

Prompt history for this node:
{history_text}

Recent modification outcomes for this node:
{modification_text}

Planner target:
- component: {candidate['target']['component']}
- subtype: {candidate['target']['subtype']}
- rca_rank: {candidate['target']['rca_rank']}

Planner reasoning:
- rationale: {candidate.get('rationale', '')}
- history_reference: {candidate.get('history_reference', '')}
- expected_effect: {candidate.get('expected_effect', '')}

Planner edit:
- style: {candidate['edit']['style']}
- op_family: {candidate['edit']['op_family']}
- instructions:
{instructions}

{_domain_section}{_dataset_rules_section}Constraints:
- Preserve required placeholders for all inputs.
- Preserve the declared output contract and output variable names.
- Keep the prompt concise, executable, and evidence-grounded.
- Do not add markdown fences.
- If the issue is binding, make variable usage explicit.
- If the issue is grounding, emphasize using only provided evidence.
- If the issue is contract, make the output format and variable names explicit.
- Avoid repeating recently failed prompt strategies unless the planner rationale clearly justifies it.
{_dataset_constraint_line}{_domain_constraint_line}
Return ONLY a JSON object like {{"prompt": "..."}}.
""".strip()

def _extract_prompt_text(raw_output: str) -> str:
    prompt_value = _legacy._extract_json_field_value(raw_output, "prompt")
    if isinstance(prompt_value, str) and prompt_value.strip():
        return prompt_value.strip()
    parsed = _legacy._extract_first_json_value(raw_output)
    if isinstance(parsed, dict):
        for key in ("new_prompt", "rewritten_prompt", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    cleaned = re.sub(r"^```(?:json|text)?", "", raw_output.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()


def _heuristic_add_prompt(old_prompt: str, instructions: Sequence[str]) -> str:
    additions = [f"- {str(line).strip()}" for line in instructions if str(line).strip()]
    if not additions:
        return old_prompt
    return (old_prompt.rstrip() + "\n\n# Planner Revision\n" + "\n".join(additions)).strip()


def _materialize_prompt_edit(
    llm,
    anchor_workflow: WorkFlowGraph,
    candidate: Dict[str, Any],
    baseline_package,
    prompt_history: PromptHistory,
    modification_history: ModificationHistory,
    prompt_retry_per_node: int,
    dataset_name: str = "",
) -> Dict[str, Any]:
    workflow = _legacy._clone_workflow_graph(anchor_workflow)
    node_name = candidate["target"]["node_name"]
    try:
        node = workflow.get_node(node_name)
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": f"prompt_edit node lookup failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": candidate["edit"].get("op_family", "")}
    old_prompt = _legacy._get_node_primary_prompt(node)
    if not old_prompt:
        return {"workflow": None, "status": "materialization_failed", "message": "prompt_edit target node has no primary prompt", "prompt_ops_by_node": {}, "selected_op_family": candidate["edit"].get("op_family", "")}
    try:
        _end_nodes = set(workflow.find_end_nodes() or [])
    except Exception:
        _end_nodes = set()
    _is_end = node_name in _end_nodes
    _workflow_goal = getattr(anchor_workflow, "goal", "") or ""
    prompt_ops = [f"{candidate['edit'].get('op_family', '')}:{line}" for line in candidate["edit"].get("instructions", [])]

    # ---- Path A: planner provided new_prompt directly (no rewriter LLM call) ----
    new_prompt_text = str((candidate.get("edit") or {}).get("new_prompt") or "").strip()
    if new_prompt_text:
        repaired = _legacy._repair_prompt_contract(
            node, new_prompt_text, is_end_node=_is_end, workflow_goal=_workflow_goal, dataset_name=dataset_name,
        )
        changed = _legacy._set_node_prompt(node=node, new_prompt=repaired)
        if changed:
            repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow)
            return {
                "workflow": workflow,
                "status": "materialized",
                "message": f"prompt_edit applied directly to {node_name}: style={candidate['edit']['style']}, op_family={candidate['edit'].get('op_family', '')}, prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}",
                "prompt_ops_by_node": {node_name: prompt_ops},
                "selected_op_family": candidate["edit"].get("op_family", ""),
            }
        # new_prompt identical to old after repair; fall through to rewriter

    # ---- Path B: fallback rewriter LLM call (when new_prompt missing or identical) ----
    raw_output = ""
    changed = False
    try:
        raw_output = _llm_text(
            llm,
            _rewrite_prompt_instruction(
                node=node,
                old_prompt=old_prompt,
                candidate=candidate,
                node_stats=baseline_package.node_stats,
                prompt_history=prompt_history,
                modification_history=modification_history,
                anchor_workflow=anchor_workflow,
                failure_examples=list(getattr(baseline_package, "failure_examples", []) or []),
                workflow_goal=_workflow_goal,
                is_end_node=_is_end,
                dataset_name=dataset_name,
            ),
        )
        rewritten_prompt = _extract_prompt_text(raw_output)
        if rewritten_prompt:
            repaired_prompt = _legacy._repair_prompt_contract(
                node,
                rewritten_prompt,
                is_end_node=_is_end,
                workflow_goal=_workflow_goal,
                dataset_name=dataset_name,
            )
            changed = _legacy._set_node_prompt(node=node, new_prompt=repaired_prompt)
    except Exception as exc:
        raw_output = f"llm_error: {exc}"
    if not changed and candidate["edit"].get("op_family") == "ADD":
        fallback_prompt = _legacy._repair_prompt_contract(
            node,
            _heuristic_add_prompt(old_prompt, candidate["edit"].get("instructions", [])),
            is_end_node=_is_end,
            workflow_goal=_workflow_goal,
            dataset_name=dataset_name,
        )
        changed = _legacy._set_node_prompt(node=node, new_prompt=fallback_prompt)
    if not changed:
        return {"workflow": None, "status": "materialization_failed", "message": f"prompt_edit produced no usable change: raw={raw_output[:300]}", "prompt_ops_by_node": {}, "selected_op_family": candidate["edit"].get("op_family", "")}
    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow)
    return {
        "workflow": workflow,
        "status": "materialized",
        "message": f"prompt_edit applied via rewriter to {node_name}: style={candidate['edit']['style']}, op_family={candidate['edit'].get('op_family', '')}, prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}",
        "prompt_ops_by_node": {node_name: prompt_ops},
        "selected_op_family": candidate["edit"].get("op_family", ""),
    }

def _coerce_param_value(name: str, value: Any) -> Any:
    if name == "temperature":
        return max(0.0, min(1.0, float(value)))
    if name == "top_p":
        return max(0.0, min(1.0, float(value)))
    if name == "max_tokens":
        return max(64, min(4096, int(value)))
    if name == "parse_mode":
        parse_mode = str(value).strip().lower()
        if parse_mode not in _ALLOWED_PARSE_MODES:
            raise ValueError(f"unsupported parse_mode={value}")
        return parse_mode
    raise ValueError(f"unsupported field={name}")


def _materialize_params_edit(anchor_workflow: WorkFlowGraph, candidate: Dict[str, Any]) -> Dict[str, Any]:
    workflow = _legacy._clone_workflow_graph(anchor_workflow)
    node_name = candidate["target"]["node_name"]
    try:
        node = workflow.get_node(node_name)
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": f"params_edit node lookup failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}
    changes: List[str] = []
    for name, raw_value in (candidate["edit"].get("param_changes") or {}).items():
        field_name = str(name)
        if field_name not in _ALLOWED_PARAM_FIELDS:
            return {"workflow": None, "status": "materialization_failed", "message": f"params_edit invalid field={field_name}", "prompt_ops_by_node": {}, "selected_op_family": ""}
        try:
            value = _coerce_param_value(field_name, raw_value)
        except Exception as exc:
            return {"workflow": None, "status": "materialization_failed", "message": f"params_edit bad value for {field_name}: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}
        if field_name == "parse_mode":
            if _legacy._set_node_parse_mode(node=node, parse_mode=value):
                changes.append(f"parse_mode->{value}")
        else:
            old_value = _legacy._get_node_generation_param(node, field_name, None)
            if _legacy._set_node_generation_param(node=node, name=field_name, value=value):
                changes.append(f"{field_name}:{old_value}->{value}")
    if not changes:
        return {"workflow": None, "status": "materialization_failed", "message": "params_edit produced no effective change", "prompt_ops_by_node": {}, "selected_op_family": ""}
    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow)
    return {
        "workflow": workflow,
        "status": "materialized",
        "message": f"params_edit applied to {node_name}: style={candidate['edit']['style']}, changes={changes}, prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}",
        "prompt_ops_by_node": {},
        "selected_op_family": "",
    }


def _param_to_spec(param: Any) -> Dict[str, Any]:
    if isinstance(param, dict):
        spec = dict(param)
    else:
        spec = {
            "name": getattr(param, "name", ""),
            "type": getattr(param, "type", "string"),
            "description": getattr(param, "description", ""),
            "required": getattr(param, "required", True),
        }
    spec["name"] = str(spec.get("name", "") or "")
    spec["type"] = str(spec.get("type", "string") or "string")
    spec["description"] = str(spec.get("description", "") or "")
    spec["required"] = bool(spec.get("required", True))
    return spec



def _dedupe_param_specs(param_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for raw_spec in param_specs or []:
        spec = _param_to_spec(raw_spec)
        key = spec["name"].lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped



def _instantiate_param_from_spec(spec: Dict[str, Any]) -> Parameter:
    payload = copy.deepcopy(spec)
    try:
        return Parameter.from_dict(payload)
    except Exception:
        return Parameter(**payload)



def _instantiate_node_from_spec(spec: Dict[str, Any]) -> WorkFlowNode:
    payload = copy.deepcopy(spec)
    payload.setdefault("agents", [])
    try:
        return WorkFlowNode.from_dict(payload)
    except Exception:
        return WorkFlowNode(**payload)



def _instantiate_edge(source: str, target: str) -> WorkFlowEdge:
    try:
        return WorkFlowEdge.from_dict({"source": source, "target": target})
    except Exception:
        return WorkFlowEdge(edge_tuple=(source, target))



def _node_input_specs(node: WorkFlowNode) -> List[Dict[str, Any]]:
    return [_param_to_spec(param) for param in getattr(node, "inputs", []) or []]



def _node_output_specs(node: WorkFlowNode) -> List[Dict[str, Any]]:
    return [_param_to_spec(param) for param in getattr(node, "outputs", []) or []]



def _node_output_names(node: WorkFlowNode) -> List[str]:
    return [spec["name"] for spec in _node_output_specs(node)]



def _set_node_inputs_from_specs(node: WorkFlowNode, param_specs: Sequence[Dict[str, Any]]):
    node.inputs = [_instantiate_param_from_spec(spec) for spec in _dedupe_param_specs(param_specs)]




def _replace_node_inputs(node: WorkFlowNode, remove_names: Sequence[str], add_specs: Sequence[Dict[str, Any]]):
    remove_lower = {str(name).lower().strip() for name in (remove_names or []) if str(name).strip()}
    kept = [spec for spec in _node_input_specs(node) if spec["name"].lower().strip() not in remove_lower]
    _set_node_inputs_from_specs(node, kept + list(add_specs or []))



def _goal_input_spec(node: WorkFlowNode) -> Dict[str, Any]:
    for spec in _node_input_specs(node):
        if spec["name"].lower().strip() == "goal":
            return spec
    return _legacy._io_spec("goal", "string", "The original question and reference context.")



def _first_consumable_output_spec(node: WorkFlowNode) -> Optional[Dict[str, Any]]:
    for spec in _node_output_specs(node):
        if spec["name"].lower().strip() != "answer":
            return spec
    output_specs = _node_output_specs(node)
    return output_specs[0] if output_specs else None



def _topological_node_names(workflow: WorkFlowGraph) -> List[str]:
    try:
        return [str(name) for name in nx.topological_sort(workflow.graph)]
    except Exception:
        return [node.name for node in workflow.nodes]



def _find_node_name_by_exact_names(
    workflow: WorkFlowGraph,
    names: Sequence[str],
    exclude_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    exclude = {str(name).lower().strip() for name in (exclude_names or []) if str(name).strip()}
    lookup = {node.name.lower().strip(): node.name for node in workflow.nodes}
    for raw_name in names or []:
        key = str(raw_name).lower().strip()
        if not key or key in exclude:
            continue
        resolved = lookup.get(key)
        if resolved:
            return resolved
    return None



# Mapping from domain-specific terms used by structure-edit helpers to
# the abstract role vocabulary returned by _infer_roles_from_text_parts.
_ROLE_ALIAS_MAP: Dict[str, Set[str]] = {
    "answer":     {"produce"},
    "extract":    {"gather"},
    "evidence":   {"gather"},
    "synthesize": {"reason"},
    "synthesise": {"reason"},
    "produce":    {"produce"},
    "gather":     {"gather"},
    "transform":  {"transform"},
    "reason":     {"reason"},
    "verify":     {"verify"},
    "understand": {"understand"},
}


def _expand_role_aliases(roles: Sequence[str]) -> Set[str]:
    """Expand domain-specific role terms to the abstract vocabulary."""
    expanded: Set[str] = set()
    for role in roles:
        key = str(role).strip().lower()
        if key in _ROLE_ALIAS_MAP:
            expanded.update(_ROLE_ALIAS_MAP[key])
        expanded.add(key)  # always keep original for direct match
    return expanded


def _find_node_name_by_roles(
    workflow: WorkFlowGraph,
    roles: Sequence[str],
    exclude_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    exclude = {str(name).lower().strip() for name in (exclude_names or []) if str(name).strip()}
    wanted = _expand_role_aliases(roles)
    for node_name in _topological_node_names(workflow):
        if node_name.lower().strip() in exclude:
            continue
        node = workflow.get_node(node_name)
        node_roles = set(_legacy._infer_workflow_node_roles(node))
        if node_roles & wanted:
            return node_name
        # Fallback: check if any wanted term appears in node name, description, or output names
        text_parts = [
            getattr(node, "name", ""),
            getattr(node, "description", ""),
        ]
        text_parts.extend(getattr(out, "name", "") for out in getattr(node, "outputs", []) or [])
        combined = " ".join(str(x) for x in text_parts if x).lower()
        if any(term in combined for term in roles if str(term).strip()):
            return node_name
    return None



def _find_child_name_by_roles(
    workflow: WorkFlowGraph,
    source_name: str,
    roles: Sequence[str],
    fallback_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    children = workflow.get_node_children(source_name)
    wanted = _expand_role_aliases(roles)
    for child_name in children:
        node = workflow.get_node(child_name)
        node_roles = set(_legacy._infer_workflow_node_roles(node))
        if node_roles & wanted:
            return child_name
    # Fallback: check if any wanted term appears in child node text
    for child_name in children:
        node = workflow.get_node(child_name)
        text_parts = [getattr(node, "name", ""), getattr(node, "description", "")]
        text_parts.extend(getattr(out, "name", "") for out in getattr(node, "outputs", []) or [])
        combined = " ".join(str(x) for x in text_parts if x).lower()
        if any(term in combined for term in roles if str(term).strip()):
            return child_name
    if fallback_names:
        child_lookup = {name.lower().strip(): name for name in children}
        for raw_name in fallback_names:
            resolved = child_lookup.get(str(raw_name).lower().strip())
            if resolved:
                return resolved
    return children[0] if children else None



def _make_unique_node_name(workflow: WorkFlowGraph, base_name: str) -> str:
    if not workflow.node_exists(base_name):
        return base_name
    suffix = 2
    while workflow.node_exists(f"{base_name}_{suffix}"):
        suffix += 1
    return f"{base_name}_{suffix}"

def _safe_add_edge(workflow: WorkFlowGraph, source: str, target: str):
    if source == target:
        return
    if workflow.edge_exists((source, target)):
        return
    workflow.add_edge(_instantiate_edge(source, target), update_graph=False)



def _safe_remove_edge(workflow: WorkFlowGraph, source: str, target: str):
    if workflow.edge_exists((source, target)):
        workflow.remove_edge((source, target), update_graph=False)



def _merge_external_input_specs(first_node: WorkFlowNode, second_node: WorkFlowNode) -> List[Dict[str, Any]]:
    internal_names = {spec["name"].lower().strip() for spec in _node_output_specs(first_node)}
    merged_specs: List[Dict[str, Any]] = []
    for spec in _node_input_specs(first_node) + _node_input_specs(second_node):
        if spec["name"].lower().strip() in internal_names and spec["name"].lower().strip() != "goal":
            continue
        merged_specs.append(spec)
    return _dedupe_param_specs(merged_specs)



def _apply_delete_target(workflow: WorkFlowGraph, target_name: str, reason: str) -> Tuple[Optional[List[str]], str]:
    if not target_name or not workflow.node_exists(target_name):
        return None, f"{reason}: target node missing"
    target_node = workflow.get_node(target_name)
    predecessor_names = workflow.get_node_predecessors(target_name)
    successor_names = workflow.get_node_children(target_name)
    if not predecessor_names or not successor_names:
        return None, f"{reason}: target node must have both predecessors and successors"

    target_output_names = _node_output_names(target_node)
    upstream_specs: List[Dict[str, Any]] = []
    for predecessor_name in predecessor_names:
        predecessor_output = _first_consumable_output_spec(workflow.get_node(predecessor_name))
        if predecessor_output is not None:
            upstream_specs.append(predecessor_output)

    for successor_name in successor_names:
        successor_node = workflow.get_node(successor_name)
        _replace_node_inputs(successor_node, target_output_names, upstream_specs)

    for predecessor_name in predecessor_names:
        _safe_remove_edge(workflow, predecessor_name, target_name)
    for successor_name in successor_names:
        _safe_remove_edge(workflow, target_name, successor_name)
    workflow.remove_node(target_name, update_graph=False)
    for predecessor_name in predecessor_names:
        for successor_name in successor_names:
            _safe_add_edge(workflow, predecessor_name, successor_name)
    workflow.update_graph()
    return sorted(set(successor_names)), f"{reason}: deleted {target_name} and reconnected {predecessor_names} -> {successor_names}"


def _select_removal_candidates(workflow: "WorkFlowGraph", root_causes: Optional[List[Tuple[str, float]]] = None) -> List[str]:
    """Rank intermediate nodes (have both predecessors and successors) for removal.

    Priority order:
      1. Return-component failure signals from RCA (highest failure prob first).
      2. Remaining intermediate nodes in topological order.
    A node is removable only when it has >=1 predecessor AND >=1 successor so rewiring is safe.
    """
    intermediate: List[str] = []
    for node in (workflow.nodes or []):
        preds = workflow.get_node_predecessors(node.name)
        succs = workflow.get_node_children(node.name)
        if preds and succs:
            intermediate.append(node.name)
    if not intermediate:
        return []
    rc_scores: Dict[str, float] = {}
    parser = getattr(_legacy, "_parse_health_node_name", None)
    for rc_item in (root_causes or []):
        try:
            rc_name, rc_prob = rc_item[0], rc_item[1]
        except Exception:
            continue
        parsed = parser(rc_name) if parser is not None else None
        if not parsed:
            continue
        component, _subtype, node_name = parsed
        if component != "Return" or not node_name or node_name not in intermediate:
            continue
        rc_scores[node_name] = max(rc_scores.get(node_name, 0.0), float(rc_prob or 0.0))
    topo = [n for n in _topological_node_names(workflow) if n in intermediate]
    ranked_by_rc = sorted(
        (name for name in intermediate if name in rc_scores),
        key=lambda n: rc_scores.get(n, 0.0),
        reverse=True,
    )
    seen = set(ranked_by_rc)
    tail = [n for n in topo if n not in seen]
    return ranked_by_rc + tail


def _build_delete_node_new_workflow_spec(workflow: "WorkFlowGraph", target_node_name: str) -> Optional[Dict[str, Any]]:
    """Build a new_workflow dict (nodes+edges spec) representing workflow with target_node_name removed.

    Successor nodes that referenced the deleted node's outputs are rewritten to consume the predecessor's
    first-consumable output instead, so the materializer regenerates their agents/prompts for the new input set.
    Returns None when the target is missing or lacks a valid predecessor-successor surround.
    """
    if not target_node_name or not workflow.node_exists(target_node_name):
        return None
    target_node = workflow.get_node(target_node_name)
    predecessor_names = list(workflow.get_node_predecessors(target_node_name) or [])
    successor_names = list(workflow.get_node_children(target_node_name) or [])
    if not predecessor_names or not successor_names:
        return None

    target_output_names_lower = {s["name"].lower().strip() for s in _node_output_specs(target_node)}
    upstream_specs: List[Dict[str, Any]] = []
    seen_upstream: set = set()
    for pred_name in predecessor_names:
        pred_output = _first_consumable_output_spec(workflow.get_node(pred_name))
        if pred_output is None:
            continue
        key = pred_output["name"].lower().strip()
        if key and key not in seen_upstream:
            seen_upstream.add(key)
            upstream_specs.append(pred_output)
    if not upstream_specs:
        return None

    nodes_spec: List[Dict[str, Any]] = []
    successor_set = set(successor_names)
    for node in (workflow.nodes or []):
        if node.name == target_node_name:
            continue
        node_spec: Dict[str, Any] = {
            "name": node.name,
            "description": getattr(node, "description", "") or "",
            "inputs": _node_input_specs(node),
            "outputs": _node_output_specs(node),
            "reason": getattr(node, "reason", "") or "",
        }
        if node.name in successor_set:
            kept_inputs = [s for s in node_spec["inputs"] if s["name"].lower().strip() not in target_output_names_lower]
            node_spec["inputs"] = _dedupe_param_specs(kept_inputs + upstream_specs)
        nodes_spec.append(node_spec)

    edges_spec: List[Dict[str, Any]] = []
    edge_set: set = set()
    for edge in (workflow.edges or []):
        try:
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            if src is None or tgt is None:
                tup = getattr(edge, "edge_tuple", None) or (None, None)
                src = src or (tup[0] if len(tup) >= 1 else None)
                tgt = tgt or (tup[1] if len(tup) >= 2 else None)
        except Exception:
            continue
        if not src or not tgt:
            continue
        src = str(src)
        tgt = str(tgt)
        if src == target_node_name or tgt == target_node_name:
            continue
        if (src, tgt) in edge_set:
            continue
        edge_set.add((src, tgt))
        edges_spec.append({"source": src, "target": tgt})
    for pred in predecessor_names:
        for succ in successor_names:
            if pred == succ or (pred, succ) in edge_set:
                continue
            edge_set.add((pred, succ))
            edges_spec.append({"source": pred, "target": succ})

    return {"nodes": nodes_spec, "edges": edges_spec}




def _build_insert_output_validator_new_workflow_spec(workflow: "WorkFlowGraph") -> Optional[Dict[str, Any]]:
    """Build a new_workflow spec that appends a validator node after the terminal.

    Why: the bonus DELETE injection only fires for workflows with >2 nodes. Tiny
    workflows (HumanEval-style single-stage code generation) instead need a
    structural ADD when prompt_edit saturates. This builder mirrors the local
    apply variant `_apply_local_insert_output_validator` but emits a spec that
    `_materialize_structure_rewrite` can consume.

    Returns None if the workflow has no terminal, the terminal already has a
    validator downstream, or the terminal has no outputs.
    """
    if not isinstance(workflow.nodes, list) or not workflow.nodes:
        return None
    try:
        end_nodes = list(workflow.find_end_nodes() or [])
    except Exception:
        return None
    if not end_nodes:
        return None

    # Names containing FORBIDDEN keywords (verify, validate, audit, run_test,
    # execute_code, ...) are rejected by _check_goal_forbidden_node_names
    # because such nodes have no oracle and tend to hallucinate. We use
    # "refine_output" + a refine/polish description so the LLM at this node
    # treats it as rewriting the draft for clarity, not as a binary
    # pass/fail validator against ground truth.
    if _find_node_name_by_exact_names(workflow, ["refine_output", "polish_output", "output_refiner", "validate_output"]):
        return None

    terminal_name = end_nodes[0]
    terminal_node = workflow.get_node(terminal_name)
    terminal_outputs = _node_output_specs(terminal_node)
    if not terminal_outputs:
        return None

    goal_spec = _goal_input_spec(terminal_node)
    draft_specs: List[Dict[str, Any]] = []
    for spec in terminal_outputs:
        draft = dict(spec)
        draft["name"] = f"{spec['name']}_draft"
        draft["description"] = f"Initial draft of {spec['name']} (will be refined by the downstream stage)."
        draft_specs.append(draft)

    refiner_name = _make_unique_node_name(workflow, "refine_output")
    refiner_inputs = _dedupe_param_specs([goal_spec] + draft_specs)
    refiner_outputs = copy.deepcopy(terminal_outputs)

    nodes_spec: List[Dict[str, Any]] = []
    for node in (workflow.nodes or []):
        node_spec: Dict[str, Any] = {
            "name": node.name,
            "description": getattr(node, "description", "") or "",
            "inputs": _node_input_specs(node),
            "outputs": _node_output_specs(node),
            "reason": getattr(node, "reason", "") or "",
        }
        if node.name == terminal_name:
            node_spec["outputs"] = draft_specs
        nodes_spec.append(node_spec)
    nodes_spec.append({
        "name": refiner_name,
        "description": (
            "Refine the upstream draft into a cleaner, more accurate final "
            "version. Re-read the original goal, then rewrite the draft to "
            "fix any obvious issues (clarity, missing fields, format "
            "violations, inconsistencies). Do NOT claim correctness against "
            "any external oracle and do NOT invent test cases; only improve "
            "what is already in the draft. Preserve the exact output schema "
            "and field names of the original task."
        ),
        "inputs": refiner_inputs,
        "outputs": refiner_outputs,
        "reason": "Bonus stagnation INSERT: append a refiner stage to escape plateau on small workflows.",
    })

    edges_spec: List[Dict[str, Any]] = []
    edge_set: set = set()
    for edge in (workflow.edges or []):
        try:
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
        except Exception:
            continue
        if not src or not tgt:
            continue
        if (src, tgt) in edge_set:
            continue
        edge_set.add((src, tgt))
        edges_spec.append({"source": src, "target": tgt})
    if (terminal_name, refiner_name) not in edge_set:
        edges_spec.append({"source": terminal_name, "target": refiner_name})

    return {"nodes": nodes_spec, "edges": edges_spec}




def _apply_local_insert_output_validator(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    """Universal INSERT_NODE variant that works on ANY workflow shape.

    Why: the existing INSERT variants (insert_evidence_organizer,
    insert_reasoning_chain_stage) require multi-node QA-style workflows with
    explicit extract/answer roles, and silently no-op on single-node code-gen
    workflows like HumanEval. That left the optimizer with no usable
    structure_edit option once prompt_edit saturated, so the population
    plateaued.

    Strategy: pick a terminal node, rename each of its outputs X -> X_draft,
    then append a validator node that takes [goal, *draft] -> X (original
    names, original types) so the workflow contract for downstream consumers
    is preserved. The validator's role is to critique the draft against the
    goal and emit a corrected version when needed. This is domain-agnostic
    and applies on any DAG with at least one end node.
    """
    # NOTE: the node name must NOT contain any FORBIDDEN keyword (verify,
    # validate, audit, run_test, execute_code, ...) — those names are
    # rejected by _check_goal_forbidden_node_names because such nodes have
    # no oracle and tend to hallucinate. We use "refine_output" + a
    # refine/polish-style description so the LLM at this node treats it as
    # rewriting the draft for clarity/correctness, not as a binary
    # pass/fail validator against ground truth.
    if _find_node_name_by_exact_names(workflow, ["refine_output", "polish_output", "output_refiner", "validate_output"]):
        return None, "insert_output_validator: refiner already exists"

    end_nodes = []
    try:
        end_nodes = list(workflow.find_end_nodes() or [])
    except Exception:
        end_nodes = []
    if not end_nodes:
        return None, "insert_output_validator: no terminal node found"

    terminal_name = end_nodes[0]
    terminal_node = workflow.get_node(terminal_name)
    terminal_outputs = _node_output_specs(terminal_node)
    if not terminal_outputs:
        return None, "insert_output_validator: terminal node has no outputs"

    goal_spec = _goal_input_spec(terminal_node)

    draft_specs: List[Dict[str, Any]] = []
    for spec in terminal_outputs:
        orig_name = spec["name"]
        draft_name = f"{orig_name}_draft"
        # Avoid collision with any existing input name.
        suffix = 2
        existing = {s["name"] for s in _node_input_specs(terminal_node)} | {s["name"] for s in terminal_outputs}
        while draft_name in existing and draft_name != f"{orig_name}_draft":
            draft_name = f"{orig_name}_draft_{suffix}"
            suffix += 1
        draft_spec = dict(spec)
        draft_spec["name"] = draft_name
        draft_spec["description"] = f"Initial draft of {orig_name} (will be refined by the downstream stage)."
        draft_specs.append(draft_spec)

    # Mutate terminal node's outputs in place so its draft names propagate to
    # the refiner. (Pydantic permits attribute assignment; the existing
    # _replace_node_inputs helper does the same for inputs.)
    terminal_node.outputs = [_instantiate_param_from_spec(spec) for spec in draft_specs]

    refiner_inputs = _dedupe_param_specs([goal_spec] + draft_specs)
    refiner_outputs = copy.deepcopy(terminal_outputs)

    refiner_name = _make_unique_node_name(workflow, "refine_output")
    refiner_spec = _legacy._node_spec(
        refiner_name,
        (
            "Refine the upstream draft into a cleaner, more accurate final "
            "version. Re-read the original goal, then rewrite the draft to "
            "fix any obvious issues (clarity, missing fields, format "
            "violations, inconsistencies). Do NOT claim correctness against "
            "any external oracle and do NOT invent test cases; only improve "
            "what is already in the draft. Preserve the exact output schema "
            "and field names of the original task."
        ),
        refiner_inputs,
        refiner_outputs,
    )
    refiner_node = _instantiate_node_from_spec(refiner_spec)
    workflow.add_node(refiner_node, update_graph=False)
    _safe_add_edge(workflow, terminal_name, refiner_node.name)
    workflow.update_graph()
    return (
        [refiner_node.name],
        f"insert_output_validator: appended refiner {refiner_node.name} after terminal node {terminal_name}",
    )


def _apply_local_insert_evidence_organizer(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    if _find_node_name_by_exact_names(workflow, ["organize_evidence"]):
        return None, "insert_evidence_organizer: organize_evidence already exists"

    extract_name = _find_node_name_by_exact_names(
        workflow,
        ["evidence_extraction", "extract_facts", "extract_supporting_facts"],
    )
    if extract_name is None:
        extract_name = _find_node_name_by_roles(
            workflow,
            ["extract", "evidence"],
            exclude_names=["query_decomposition", "answer_generation", "multi_hop_synthesis", "multi_hop_reasoning", "reasoning_and_answer"],
        )
    if extract_name is None:
        return None, "insert_evidence_organizer: no extraction-like node found"

    downstream_name = _find_child_name_by_roles(
        workflow,
        extract_name,
        ["answer", "synthesize"],
        fallback_names=["answer_generation", "multi_hop_synthesis", "multi_hop_reasoning", "reasoning_and_answer"],
    )
    if downstream_name is None:
        return None, "insert_evidence_organizer: no downstream synthesis/answer node found"

    extract_node = workflow.get_node(extract_name)
    downstream_node = workflow.get_node(downstream_name)
    goal_spec = _goal_input_spec(extract_node)
    extract_output = _first_consumable_output_spec(extract_node)
    if extract_output is None:
        return None, "insert_evidence_organizer: extraction node has no consumable output"

    organize_spec = _legacy._node_spec(
        _make_unique_node_name(workflow, "organize_evidence"),
        "Organize the extracted facts into traceable bundles that prepare downstream reasoning.",
        [goal_spec, extract_output],
        [_legacy._io_spec("organized_evidence", "list", "Traceable evidence bundles ready for downstream use.")],
    )
    organize_node = _instantiate_node_from_spec(organize_spec)
    workflow.add_node(organize_node, update_graph=False)
    _safe_remove_edge(workflow, extract_name, downstream_name)
    _safe_add_edge(workflow, extract_name, organize_node.name)
    _safe_add_edge(workflow, organize_node.name, downstream_name)
    _replace_node_inputs(
        downstream_node,
        [extract_output["name"], "organized_evidence"],
        [goal_spec, _legacy._io_spec("organized_evidence", "list", "Traceable evidence bundles ready for downstream use.")],
    )
    workflow.update_graph()
    return [organize_node.name, downstream_name], f"insert_evidence_organizer: inserted {organize_node.name} between {extract_name} and {downstream_name}"

def _apply_local_insert_reasoning_chain_stage(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    answer_name = _find_node_name_by_exact_names(workflow, ["answer_generation", "reasoning_and_answer"])
    if answer_name is None:
        answer_name = _find_node_name_by_roles(workflow, ["answer"])
    if answer_name is None:
        return None, "insert_reasoning_chain_stage: no answer node found"

    source_name = None
    predecessor_names = workflow.get_node_predecessors(answer_name)
    for predecessor_name in predecessor_names:
        predecessor_roles = set(_legacy._infer_workflow_node_roles(workflow.get_node(predecessor_name)))
        if predecessor_roles & {"extract", "evidence", "synthesize"}:
            source_name = predecessor_name
            break
    if source_name is None:
        source_name = _find_node_name_by_roles(
            workflow,
            ["extract", "evidence", "synthesize"],
            exclude_names=[answer_name, "query_decomposition"],
        )
    if source_name is None:
        return None, "insert_reasoning_chain_stage: no upstream evidence/reasoning node found"

    source_node = workflow.get_node(source_name)
    answer_node = workflow.get_node(answer_name)
    goal_spec = _goal_input_spec(answer_node)
    source_output = _first_consumable_output_spec(source_node)
    if source_output is None:
        return None, "insert_reasoning_chain_stage: source node has no consumable output"

    reasoning_name = _make_unique_node_name(workflow, "multi_hop_synthesis")
    reasoning_spec = _legacy._node_spec(
        reasoning_name,
        "Synthesize the upstream evidence into a concise reasoning chain before final answer generation.",
        [goal_spec, source_output],
        [_legacy._io_spec("reasoning_chain", "string", "A concise reasoning chain grounded in the upstream evidence.")],
    )
    reasoning_node = _instantiate_node_from_spec(reasoning_spec)
    workflow.add_node(reasoning_node, update_graph=False)
    _safe_remove_edge(workflow, source_name, answer_name)
    _safe_add_edge(workflow, source_name, reasoning_node.name)
    _safe_add_edge(workflow, reasoning_node.name, answer_name)
    _replace_node_inputs(
        answer_node,
        [source_output["name"], "reasoning_chain"],
        [goal_spec, _legacy._io_spec("reasoning_chain", "string", "A concise reasoning chain grounded in the upstream evidence.")],
    )
    workflow.update_graph()
    return [reasoning_node.name, answer_name], f"insert_reasoning_chain_stage: inserted {reasoning_node.name} between {source_name} and {answer_name}"



def _apply_local_merge_extract_organize(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    first_name = _find_node_name_by_exact_names(workflow, ["evidence_extraction", "extract_facts", "extract_supporting_facts"])
    if first_name is None:
        first_name = _find_node_name_by_roles(workflow, ["extract"], exclude_names=["answer_generation"])
    if first_name is None:
        return None, "merge_extract_organize: no extraction node found"

    second_name = _find_child_name_by_roles(
        workflow,
        first_name,
        ["evidence"],
        fallback_names=["organize_evidence", "organize_evidence_plan"],
    )
    if second_name is None or second_name == first_name:
        return None, "merge_extract_organize: no downstream organizer node found"

    first_node = workflow.get_node(first_name)
    second_node = workflow.get_node(second_name)
    merged_name = _make_unique_node_name(workflow, "evidence_extraction_and_organization")
    merged_spec = _legacy._node_spec(
        merged_name,
        "Extract grounded evidence and organize it into traceable bundles aligned to each sub-question.",
        _merge_external_input_specs(first_node, second_node),
        _dedupe_param_specs(_node_output_specs(second_node) + _node_output_specs(first_node)),
    )
    merged_node = _instantiate_node_from_spec(merged_spec)

    predecessor_names = sorted(
        set(
            workflow.get_node_predecessors(first_name)
            + [name for name in workflow.get_node_predecessors(second_name) if name not in {first_name, second_name}]
        )
    )
    successor_names = sorted(
        set(
            [name for name in workflow.get_node_children(first_name) if name not in {first_name, second_name}]
            + [name for name in workflow.get_node_children(second_name) if name not in {first_name, second_name}]
        )
    )

    workflow.add_node(merged_node, update_graph=False)
    workflow.remove_node(second_name, update_graph=False)
    workflow.remove_node(first_name, update_graph=False)
    for predecessor_name in predecessor_names:
        _safe_add_edge(workflow, predecessor_name, merged_node.name)
    for successor_name in successor_names:
        _safe_add_edge(workflow, merged_node.name, successor_name)
    workflow.update_graph()
    return [merged_node.name], f"merge_extract_organize: merged {first_name} + {second_name} into {merged_node.name}"



def _apply_local_merge_synthesize_answer(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    first_name = _find_node_name_by_exact_names(workflow, ["multi_hop_synthesis", "multi_hop_reasoning"])
    if first_name is None:
        first_name = _find_node_name_by_roles(workflow, ["synthesize"], exclude_names=["answer_generation"])
    if first_name is None:
        return None, "merge_synthesize_answer: no reasoning node found"

    second_name = _find_child_name_by_roles(workflow, first_name, ["answer"], fallback_names=["answer_generation", "reasoning_and_answer"])
    if second_name is None or second_name == first_name:
        second_name = _find_node_name_by_roles(workflow, ["answer"], exclude_names=[first_name])
    if second_name is None:
        return None, "merge_synthesize_answer: no answer node found"

    first_node = workflow.get_node(first_name)
    second_node = workflow.get_node(second_name)
    merged_name = _make_unique_node_name(workflow, "reasoning_and_answer")
    merged_spec = _legacy._node_spec(
        merged_name,
        "Synthesize the upstream evidence into a concise reasoning chain and produce the final answer.",
        _merge_external_input_specs(first_node, second_node),
        _node_output_specs(second_node),
    )
    merged_node = _instantiate_node_from_spec(merged_spec)

    predecessor_names = sorted(
        set(
            workflow.get_node_predecessors(first_name)
            + [name for name in workflow.get_node_predecessors(second_name) if name not in {first_name, second_name}]
        )
    )
    successor_names = sorted(
        set(
            [name for name in workflow.get_node_children(first_name) if name not in {first_name, second_name}]
            + [name for name in workflow.get_node_children(second_name) if name not in {first_name, second_name}]
        )
    )

    workflow.add_node(merged_node, update_graph=False)
    workflow.remove_node(second_name, update_graph=False)
    workflow.remove_node(first_name, update_graph=False)
    for predecessor_name in predecessor_names:
        _safe_add_edge(workflow, predecessor_name, merged_node.name)
    for successor_name in successor_names:
        _safe_add_edge(workflow, merged_node.name, successor_name)
    workflow.update_graph()
    return [merged_node.name], f"merge_synthesize_answer: merged {first_name} + {second_name} into {merged_node.name}"

def _apply_local_split_extract_and_organize(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    target_name = _find_node_name_by_exact_names(workflow, ["evidence_extraction_and_organization"])
    if target_name is None:
        return _apply_local_insert_evidence_organizer(workflow)

    target_node = workflow.get_node(target_name)
    predecessor_names = workflow.get_node_predecessors(target_name)
    successor_names = workflow.get_node_children(target_name)
    if not predecessor_names:
        return None, "split_extract_and_organize: target node has no predecessors"

    extract_name = _make_unique_node_name(workflow, "evidence_extraction")
    organize_name = _make_unique_node_name(workflow, "organize_evidence")
    goal_spec = _goal_input_spec(target_node)
    extract_output = _legacy._io_spec("extracted_facts", "list", "Grounded facts aligned to the decomposed queries.")
    organize_outputs = _node_output_specs(target_node) or [_legacy._io_spec("organized_evidence", "list", "Traceable evidence bundles ready for downstream use.")]

    extract_node = _instantiate_node_from_spec(
        _legacy._node_spec(
            extract_name,
            "Extract grounded facts for each decomposed query without synthesizing across hops.",
            _node_input_specs(target_node),
            [extract_output],
        )
    )
    organize_node = _instantiate_node_from_spec(
        _legacy._node_spec(
            organize_name,
            "Organize the extracted facts into traceable bundles that prepare downstream reasoning.",
            [goal_spec, extract_output],
            organize_outputs,
        )
    )

    workflow.add_node(extract_node, update_graph=False)
    workflow.add_node(organize_node, update_graph=False)
    workflow.remove_node(target_name, update_graph=False)
    for predecessor_name in predecessor_names:
        _safe_add_edge(workflow, predecessor_name, extract_node.name)
    _safe_add_edge(workflow, extract_node.name, organize_node.name)
    for successor_name in successor_names:
        _safe_add_edge(workflow, organize_node.name, successor_name)
        _replace_node_inputs(workflow.get_node(successor_name), _node_output_names(target_node), organize_outputs)
    workflow.update_graph()
    return [extract_node.name, organize_node.name] + successor_names, f"split_extract_and_organize: split {target_name} into {extract_node.name} -> {organize_node.name}"



def _apply_local_split_reasoning_and_answer(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    target_name = _find_node_name_by_exact_names(workflow, ["reasoning_and_answer"])
    if target_name is None:
        return _apply_local_insert_reasoning_chain_stage(workflow)

    target_node = workflow.get_node(target_name)
    predecessor_names = workflow.get_node_predecessors(target_name)
    successor_names = workflow.get_node_children(target_name)
    if not predecessor_names:
        return None, "split_reasoning_and_answer: target node has no predecessors"

    reasoning_name = _make_unique_node_name(workflow, "multi_hop_synthesis")
    answer_name = _make_unique_node_name(workflow, "answer_generation")
    goal_spec = _goal_input_spec(target_node)
    reasoning_output = _legacy._io_spec("reasoning_chain", "string", "A concise reasoning chain grounded in the upstream evidence.")

    reasoning_node = _instantiate_node_from_spec(
        _legacy._node_spec(
            reasoning_name,
            "Synthesize the upstream evidence into a concise reasoning chain before final answer generation.",
            _node_input_specs(target_node),
            [reasoning_output],
        )
    )
    answer_node = _instantiate_node_from_spec(
        _legacy._node_spec(
            answer_name,
            "Generate the final concise answer based only on the reasoning chain and the original goal.",
            [goal_spec, reasoning_output],
            _node_output_specs(target_node),
        )
    )

    workflow.add_node(reasoning_node, update_graph=False)
    workflow.add_node(answer_node, update_graph=False)
    workflow.remove_node(target_name, update_graph=False)
    for predecessor_name in predecessor_names:
        _safe_add_edge(workflow, predecessor_name, reasoning_node.name)
    _safe_add_edge(workflow, reasoning_node.name, answer_node.name)
    for successor_name in successor_names:
        _safe_add_edge(workflow, answer_node.name, successor_name)
    workflow.update_graph()
    return [reasoning_node.name, answer_node.name] + successor_names, f"split_reasoning_and_answer: split {target_name} into {reasoning_node.name} -> {answer_node.name}"

def _apply_local_rewire_add_shortcut_edge(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    answer_name = _find_node_name_by_roles(workflow, ["answer"])
    if answer_name is None:
        return None, "add_shortcut_edge: no answer node found"
    source_name = _find_node_name_by_exact_names(workflow, ["evidence_extraction", "extract_facts", "extract_supporting_facts"])
    if source_name is None:
        source_name = _find_node_name_by_roles(workflow, ["extract", "evidence"], exclude_names=[answer_name, "query_decomposition"])
    if source_name is None:
        return None, "add_shortcut_edge: no extraction/evidence node found"
    if workflow.edge_exists((source_name, answer_name)):
        return None, f"add_shortcut_edge: edge ({source_name}, {answer_name}) already exists"

    source_output = _first_consumable_output_spec(workflow.get_node(source_name))
    if source_output is None:
        return None, "add_shortcut_edge: source node has no consumable output"
    _replace_node_inputs(workflow.get_node(answer_name), [source_output["name"]], [source_output])
    _safe_add_edge(workflow, source_name, answer_name)
    workflow.update_graph()
    return [answer_name], f"add_shortcut_edge: added shortcut {source_name} -> {answer_name}"



def _apply_local_rewire_linear_chain(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    topo_names = _topological_node_names(workflow)
    if len(topo_names) < 3:
        return None, "linear_chain: workflow needs at least 3 nodes"
    topo_rank = {name: idx for idx, name in enumerate(topo_names)}
    ordered_names = [
        node.name
        for node in sorted(
            workflow.nodes,
            key=lambda node: (
                min(_legacy._STRUCTURE_ROLE_ORDER.get(role, 5) for role in _legacy._infer_workflow_node_roles(node)),
                topo_rank.get(node.name, 10**6),
                node.name,
            ),
        )
    ]
    for edge in list(workflow.edges):
        workflow.remove_edge(edge, update_graph=False)
    for idx in range(len(ordered_names) - 1):
        source_name = ordered_names[idx]
        target_name = ordered_names[idx + 1]
        source_output = _first_consumable_output_spec(workflow.get_node(source_name))
        if source_output is not None:
            _replace_node_inputs(workflow.get_node(target_name), [], [source_output])
        _safe_add_edge(workflow, source_name, target_name)
    workflow.update_graph()
    return ordered_names[1:], f"linear_chain: rewired workflow to {' -> '.join(ordered_names)}"



def _apply_local_rewire_swap_middle_stages(workflow: WorkFlowGraph) -> Tuple[Optional[List[str]], str]:
    ordered_names = _topological_node_names(workflow)
    if len(ordered_names) < 4:
        return None, "swap_middle_stages: workflow needs at least 4 nodes"

    first_name = ordered_names[0]
    last_name = ordered_names[-1]
    middle_names = ordered_names[1:-1]
    if len(middle_names) < 2:
        return None, "swap_middle_stages: not enough middle stages"
    middle_left = middle_names[0]
    middle_right = middle_names[1]

    first_node = workflow.get_node(first_name)
    middle_left_node = workflow.get_node(middle_left)
    middle_right_node = workflow.get_node(middle_right)
    last_node = workflow.get_node(last_name)

    first_output = _first_consumable_output_spec(first_node)
    middle_left_output = _first_consumable_output_spec(middle_left_node)
    middle_right_output = _first_consumable_output_spec(middle_right_node)
    if first_output is None or middle_left_output is None or middle_right_output is None:
        return None, "swap_middle_stages: missing consumable outputs on middle stages"

    _replace_node_inputs(middle_right_node, _node_output_names(middle_left_node), [first_output])
    _replace_node_inputs(middle_left_node, _node_output_names(first_node), [middle_right_output])
    _replace_node_inputs(last_node, _node_output_names(middle_right_node), [middle_left_output])

    for edge_pair in [
        (first_name, middle_left),
        (middle_left, middle_right),
        (middle_right, last_name),
        (first_name, middle_right),
        (middle_right, middle_left),
        (middle_left, last_name),
    ]:
        _safe_remove_edge(workflow, edge_pair[0], edge_pair[1])
    _safe_add_edge(workflow, first_name, middle_right)
    _safe_add_edge(workflow, middle_right, middle_left)
    _safe_add_edge(workflow, middle_left, last_name)
    workflow.update_graph()
    return [middle_right, middle_left, last_name], f"swap_middle_stages: rewired {first_name} -> {middle_right} -> {middle_left} -> {last_name}"

def _apply_local_structure_variant(workflow: WorkFlowGraph, style: str, variant: str) -> Tuple[Optional[List[str]], str]:
    if style == "INSERT_NODE" and variant == "insert_output_validator":
        return _apply_local_insert_output_validator(workflow)
    if style == "INSERT_NODE" and variant == "insert_evidence_organizer":
        return _apply_local_insert_evidence_organizer(workflow)
    if style == "INSERT_NODE" and variant == "insert_reasoning_chain_stage":
        return _apply_local_insert_reasoning_chain_stage(workflow)
    if style == "DELETE_NODE" and variant == "delete_evidence_stage":
        target_name = _find_node_name_by_exact_names(workflow, ["evidence_extraction", "extract_facts", "extract_supporting_facts", "evidence_extraction_and_organization"])
        if target_name is None:
            target_name = _find_node_name_by_roles(workflow, ["extract", "evidence"], exclude_names=["query_decomposition", "answer_generation"])
        return _apply_delete_target(workflow, target_name or "", "delete_evidence_stage")
    if style == "DELETE_NODE" and variant == "delete_redundant_synthesis":
        target_name = _find_node_name_by_exact_names(workflow, ["multi_hop_synthesis", "multi_hop_reasoning", "reasoning_and_answer"])
        if target_name is None:
            target_name = _find_node_name_by_roles(workflow, ["synthesize"], exclude_names=["answer_generation"])
        return _apply_delete_target(workflow, target_name or "", "delete_redundant_synthesis")
    if style == "MERGE_NODE" and variant == "merge_extract_organize":
        return _apply_local_merge_extract_organize(workflow)
    if style == "MERGE_NODE" and variant == "merge_synthesize_answer":
        return _apply_local_merge_synthesize_answer(workflow)
    if style == "SPLIT_NODE" and variant == "split_extract_and_organize":
        return _apply_local_split_extract_and_organize(workflow)
    if style == "SPLIT_NODE" and variant == "split_reasoning_and_answer":
        return _apply_local_split_reasoning_and_answer(workflow)
    if style == "REWIRE_EDGE" and variant == "add_shortcut_edge":
        return _apply_local_rewire_add_shortcut_edge(workflow)
    if style == "REWIRE_EDGE" and variant == "linear_chain":
        return _apply_local_rewire_linear_chain(workflow)
    if style == "REWIRE_EDGE" and variant == "swap_middle_stages":
        return _apply_local_rewire_swap_middle_stages(workflow)
    # LLM_PROPOSE: open-ended structure edit proposed by the LLM planner
    if style == "LLM_PROPOSE" and variant == "llm_open_structure":
        # This variant is handled externally by _materialize_llm_propose_structure_edit
        # Return empty affected list to signal external handling needed
        return [], "llm_open_structure requires external LLM-driven materialization"
    return None, f"unsupported structure variant: style={style}, variant={variant}"


def _materialize_llm_propose_structure_edit(llm, wf_generator, anchor_workflow: WorkFlowGraph, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-driven open-ended structure edit: ask LLM to propose INSERT/DELETE/MERGE/SPLIT/REWIRE operations."""
    workflow = _legacy._clone_workflow_graph(anchor_workflow)
    node_names = [n.name for n in workflow.nodes]
    edge_desc = [(getattr(e, "source", ""), getattr(e, "target", "")) for e in (workflow.edges or [])]
    instructions = candidate["edit"].get("instructions", [])
    rationale = candidate.get("rationale", "")

    propose_prompt = f"""You are a workflow structure optimizer. Given the current workflow topology, propose a concrete structural modification.

Current nodes: {node_names}
Current edges: {edge_desc}
Workflow goal: {getattr(workflow, 'goal', '')}

Planner rationale: {rationale}
Planner instructions: {instructions}

Propose ONE structural operation from: INSERT_NODE, DELETE_NODE, MERGE_NODES, SPLIT_NODE, REWIRE_EDGE.
Return a JSON object with:
- "operation": one of the above
- "details": object with operation-specific fields:
  - INSERT_NODE: {{"new_node_name": str, "description": str, "insert_after": str, "inputs": [...], "outputs": [...]}}
  - DELETE_NODE: {{"node_name": str}}
  - MERGE_NODES: {{"source_nodes": [str, str], "merged_name": str, "description": str}}
  - SPLIT_NODE: {{"node_name": str, "split_into": [{{"name": str, "description": str}}, ...]}}
  - REWIRE_EDGE: {{"remove_edges": [[src, tgt], ...], "add_edges": [[src, tgt], ...]}}

Return ONLY the JSON object."""

    try:
        from evoagentx.core.message import Message, MessageType
        response = llm.execute(
            Message(role="user", content=propose_prompt, return_msg_type=MessageType.REQUEST)
        )
        raw_output = getattr(response, "content", str(response))
        proposed = _legacy._extract_first_json_value(raw_output)
        if not isinstance(proposed, dict) or "operation" not in proposed:
            return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: LLM did not return valid operation JSON: {raw_output[:200]}", "prompt_ops_by_node": {}, "selected_op_family": ""}
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: LLM call failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    operation = proposed.get("operation", "")
    details = proposed.get("details", {})
    affected_nodes = []

    try:
        if operation == "DELETE_NODE":
            del_name = details.get("node_name", "")
            if workflow.node_exists(del_name):
                workflow.remove_node(del_name)
                affected_nodes = [n.name for n in workflow.nodes]  # all remaining nodes may need re-wiring check
        elif operation == "INSERT_NODE":
            new_name = details.get("new_node_name", f"llm_inserted_{len(workflow.nodes)}")
            insert_after = details.get("insert_after", "")
            new_node = WorkFlowNode(
                name=new_name,
                description=details.get("description", "LLM-proposed node"),
                inputs=details.get("inputs", []),
                outputs=details.get("outputs", []),
            )
            workflow.add_node(new_node)
            if insert_after and workflow.node_exists(insert_after):
                # Rewire: insert_after -> new_node, new_node -> old_downstream
                for edge in list(workflow.edges or []):
                    if getattr(edge, "source", None) == insert_after:
                        old_target = getattr(edge, "target", None)
                        workflow.add_edge(WorkFlowEdge(source=new_name, target=old_target))
                        workflow.remove_edge(insert_after, old_target)
                workflow.add_edge(WorkFlowEdge(source=insert_after, target=new_name))
            affected_nodes = [new_name]
        elif operation == "REWIRE_EDGE":
            for src, tgt in (details.get("remove_edges", []) or []):
                try:
                    workflow.remove_edge(src, tgt)
                except Exception:
                    pass
            for src, tgt in (details.get("add_edges", []) or []):
                workflow.add_edge(WorkFlowEdge(source=src, target=tgt))
            affected_nodes = list(set(
                [s for s, _ in details.get("add_edges", [])] +
                [t for _, t in details.get("add_edges", [])]
            ))
        else:
            return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: unsupported operation '{operation}'", "prompt_ops_by_node": {}, "selected_op_family": ""}
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: mutation failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    valid, reasons, _ = _legacy._validate_workflow_structure_for_evolution(workflow)
    if not valid:
        return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: invalid after mutation: {reasons}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    _legacy._enforce_workflow_contracts(workflow)
    target_nodes = [name for name in affected_nodes if workflow.node_exists(name)]
    if target_nodes:
        try:
            workflow = wf_generator.generate_agents(goal=anchor_workflow.goal, workflow=workflow, target_node_names=target_nodes)
        except Exception as exc:
            return {"workflow": None, "status": "materialization_failed", "message": f"LLM_PROPOSE: agent generation failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    return {
        "workflow": workflow,
        "status": "materialized",
        "message": f"LLM_PROPOSE applied: operation={operation}, details={details}, affected={target_nodes}",
        "prompt_ops_by_node": {},
        "selected_op_family": "",
    }


def _autorepair_role_coverage(workflow) -> int:
    """Satisfy the 3-distinct-roles validator by appending a role-indicative
    verb to the description of the most-generic nodes. Conservative: only
    touches nodes whose current inferred roles set is empty or == {"generic"},
    and only appends a short tag (never renames or rewrites prose). This is a
    safety net for cases where the planner LLM produces a semantically valid
    structure but uses verbs the keyword-matching validator does not recognize.

    Returns the number of tags injected. # ROLE_AND_DRIFT_PATCH_V1
    """
    try:
        meta = _legacy._workflow_role_meta(workflow)
    except Exception:
        return 0
    if int(meta.get("count", 0)) >= 3:
        return 0
    covered = set(meta.get("covered_roles", []) or [])
    ordered_roles = ["reason", "verify", "transform", "gather", "understand", "produce"]
    role_verbs = {
        "understand": "analyze",
        "gather": "gather",
        "transform": "transform",
        "reason": "reason",
        "verify": "verify",
        "produce": "produce",
    }
    missing = [r for r in ordered_roles if r not in covered]
    if not missing:
        return 0
    nodes = list(getattr(workflow, "nodes", []) or [])
    if not nodes:
        return 0
    scored = []
    for idx, node in enumerate(nodes):
        try:
            roles = set(_legacy._infer_workflow_node_roles(node))
        except Exception:
            roles = set()
        if not roles or roles == {"generic"}:
            priority = 0
        else:
            priority = 1 + len(roles & covered)
        scored.append((priority, idx, node))
    scored.sort(key=lambda x: (x[0], x[1]))
    needed = 3 - int(meta.get("count", 0))
    assigned = set()
    repairs = 0
    for role in missing:
        if needed <= 0:
            break
        verb = role_verbs.get(role, role)
        for _, _, node in scored:
            name = getattr(node, "name", "") or ""
            if name in assigned:
                continue
            try:
                desc = getattr(node, "description", "") or ""
                if verb in desc.lower() or verb in name.lower():
                    continue
                trailing_dot = desc.rstrip().endswith(".")
                sep = " " if trailing_dot else ". "
                new_desc = (desc.rstrip() + sep + f"This step will {verb} the provided inputs.").strip()
                node.description = new_desc
                assigned.add(name)
                repairs += 1
                needed -= 1
                break
            except Exception:
                continue
    return repairs


def _autorepair_global_interface(workflow) -> int:
    """Ensure at least one initial node exposes a required `goal` input and
    at least one end node exposes a required `answer` output. Uses a
    RENAME-FIRST strategy: if the LLM chose a semantically-equivalent name
    (e.g. `final_answer`, `result`, `problem`, `query`), rename it to the
    canonical `goal`/`answer` rather than adding a duplicate parameter.
    Only ADD a new parameter when no existing parameter on the relevant
    node is a viable canonical candidate.

    Returns the number of repairs applied. # GLOBAL_INTERFACE_RENAME_FIRST_V1

    Rationale: the previous ADD-only strategy broke `generate_agents` when
    the LLM picked names like `final_answer` - autorepair would append a
    second `answer` output, the declared outputs became `[final_answer,
    answer]`, but the LLM-synthesized prompt only had ONE section, so the
    `GeneratedAgent` pydantic validator rejected the result and the node
    fell back to a generic agent. This made structure_edit catastrophically
    degrade F1 (observed 0.5126 -> 0.1092 on MATH).
    """
    GOAL_ALIASES = ("problem", "task", "question", "prompt", "query",
                    "input", "instruction", "goal_text", "user_goal",
                    "user_input", "requirement")
    ANSWER_ALIASES = ("answer", "result", "final", "output", "solution",
                      "response", "conclusion")

    def _canonicalize(params, canonical_name, aliases):
        params = list(params or [])
        # Already has canonical required?
        for p in params:
            if (getattr(p, "name", "") or "").lower() == canonical_name and getattr(p, "required", True):
                return "present"
        # Alias match -> rename the first matching param
        for p in params:
            n = (getattr(p, "name", "") or "").lower()
            if not n:
                continue
            if any(alias in n for alias in aliases):
                try:
                    p.name = canonical_name
                    p.required = True
                    return "renamed"
                except Exception:
                    return "missing"
        # Single param -> rename it (covers exotic names the alias list missed)
        if len(params) == 1:
            p = params[0]
            try:
                p.name = canonical_name
                p.required = True
                return "renamed"
            except Exception:
                return "missing"
        return "missing"

    repairs = 0
    try:
        initial_node_names = workflow.find_initial_nodes() or []
        end_node_names = workflow.find_end_nodes() or []
    except Exception:
        return 0

    # --- entry nodes: ensure at least one initial node exposes a required `goal` input.
    goal_ok = False
    missing_initial = []
    for name in initial_node_names:
        node = workflow.get_node(name)
        status = _canonicalize(getattr(node, "inputs", []) or [], "goal", GOAL_ALIASES)
        if status == "present":
            goal_ok = True
            break
        elif status == "renamed":
            goal_ok = True
            repairs += 1
            break
        else:
            missing_initial.append(name)
    if not goal_ok and initial_node_names:
        target_name = missing_initial[0] if missing_initial else initial_node_names[0]
        node = workflow.get_node(target_name)
        try:
            node.inputs = list(getattr(node, "inputs", []) or []) + [Parameter(
                name="goal",
                type="string",
                description="The original task description / problem statement passed in as the workflow's global input.",
                required=True,
            )]
            repairs += 1
        except Exception:
            pass

    # --- end nodes: ensure at least one end node exposes a required `answer` output.
    answer_ok = False
    missing_end = []
    for name in end_node_names:
        node = workflow.get_node(name)
        status = _canonicalize(getattr(node, "outputs", []) or [], "answer", ANSWER_ALIASES)
        if status == "present":
            answer_ok = True
            break
        elif status == "renamed":
            answer_ok = True
            repairs += 1
            break
        else:
            missing_end.append(name)
    if not answer_ok and end_node_names:
        target_name = missing_end[0] if missing_end else end_node_names[0]
        node = workflow.get_node(target_name)
        try:
            node.outputs = list(getattr(node, "outputs", []) or []) + [Parameter(
                name="answer",
                type="string",
                description="The final answer produced by the workflow, returned as the workflow's global output.",
                required=True,
            )]
            repairs += 1
        except Exception:
            pass

    return repairs


def _materialize_structure_edit(wf_generator, anchor_workflow: WorkFlowGraph, candidate: Dict[str, Any]) -> Dict[str, Any]:
    workflow = _legacy._clone_workflow_graph(anchor_workflow)
    style = candidate["edit"]["style"]
    variant = candidate["edit"].get("structure_variant", "")

    affected_nodes, mutation_message = _apply_local_structure_variant(workflow, style, variant)
    if not affected_nodes:
        return {"workflow": None, "status": "materialization_failed", "message": mutation_message, "prompt_ops_by_node": {}, "selected_op_family": ""}

    _autorepair_global_interface(workflow)
    _autorepair_role_coverage(workflow)  # ROLE_AND_DRIFT_PATCH_V1
    valid, reasons, _ = _legacy._validate_workflow_structure_for_evolution(workflow)
    if not valid:
        return {"workflow": None, "status": "materialization_failed", "message": f"structure_edit invalid after local mutation: {reasons}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow)
    target_nodes = [name for name in sorted(set(affected_nodes)) if workflow.node_exists(name)]
    try:
        workflow = wf_generator.generate_agents(goal=anchor_workflow.goal, workflow=workflow, target_node_names=target_nodes)
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": f"structure_edit partial agent generation failed: {exc}", "prompt_ops_by_node": {}, "selected_op_family": ""}

    post_safety = _workflow_safety_status(workflow)
    if post_safety["prompt_errors"]:
        return {
            "workflow": None,
            "status": "materialization_failed",
            "validation_status": "post_generation_invalid_prompt_template",
            "message": f"structure_edit generated invalid prompt templates after local mutation: {post_safety['prompt_errors'][:5]}",
            "prompt_ops_by_node": {},
            "selected_op_family": "",
        }
    if not post_safety["structure_valid"]:
        return {
            "workflow": None,
            "status": "materialization_failed",
            "validation_status": "post_generation_invalid_structure",
            "message": f"structure_edit invalid after local mutation: {post_safety['structure_reasons']}",
            "prompt_ops_by_node": {},
            "selected_op_family": "",
        }
    # Fix #9: Adapt downstream node prompts to reflect structural changes
    downstream_adapted = []
    for affected_name in target_nodes:
        for edge in (workflow.edges or []):
            downstream_name = getattr(edge, "target", None)
            if getattr(edge, "source", None) == affected_name and downstream_name and downstream_name not in target_nodes:
                ds_node = None
                for n in workflow.nodes:
                    if n.name == downstream_name:
                        ds_node = n
                        break
                if ds_node is None:
                    continue
                ds_prompt = _legacy._get_node_primary_prompt(ds_node) or ""
                if not ds_prompt.strip():
                    continue
                # Check if downstream prompt references the affected node's old outputs
                affected_node = None
                for n in workflow.nodes:
                    if n.name == affected_name:
                        affected_node = n
                        break
                if affected_node is None:
                    continue
                # Inject a coordination hint about upstream structural change
                coordination_hint = (
                    f"\n\n[COORDINATION NOTE: Upstream node '{affected_name}' was structurally modified "
                    f"(style={style}, variant={variant}). Ensure your inputs from this node are still valid.]"
                )
                if coordination_hint.strip() not in ds_prompt:
                    _legacy._set_node_prompt(ds_node, ds_prompt.rstrip() + coordination_hint)
                    downstream_adapted.append(downstream_name)
    adapt_msg = f", downstream_adapted={downstream_adapted}" if downstream_adapted else ""

    return {
        "workflow": workflow,
        "status": "materialized",
        "message": (
            f"structure_edit applied: style={style}, variant={variant}, affected_nodes={target_nodes}, "
            f"prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}{adapt_msg}; {mutation_message}"
        ),
        "prompt_ops_by_node": {},
        "selected_op_family": "",
    }


def _node_spec_changed(old_node, new_spec: dict) -> bool:
    """Check if a node spec differs from an existing node enough to require agent regeneration."""
    old_desc = (getattr(old_node, "description", "") or "").strip()
    new_desc = (new_spec.get("description", "") or "").strip()
    if old_desc != new_desc:
        return True
    old_inputs = sorted(getattr(inp, "name", "") for inp in (getattr(old_node, "inputs", []) or []))
    new_inputs = sorted(inp.get("name", "") for inp in (new_spec.get("inputs", []) or []))
    if old_inputs != new_inputs:
        return True
    old_outputs = sorted(getattr(out, "name", "") for out in (getattr(old_node, "outputs", []) or []))
    new_outputs = sorted(out.get("name", "") for out in (new_spec.get("outputs", []) or []))
    if old_outputs != new_outputs:
        return True
    return False


_FORBIDDEN_NODE_NAME_KEYWORDS = (
    "verify", "verification", "validate", "validation",
    "audit", "correctness_check",
    "cross_check", "crosscheck", "double_check", "doublecheck",
    "compare_to_reference", "compare_with_gold", "compare_to_gold",
    "test_assertion", "derive_test", "guess_test", "extract_test",
    "run_test", "run_tests", "run_code", "execute_code", "execute_python",
    "code_execution", "run_calculator", "use_calculator",
    "ground_truth_lookup", "gold_lookup", "reference_lookup",
)


def _check_goal_forbidden_node_names(changed_names) -> tuple:
    """Common-FORBIDDEN-keyword guard for structure_edit newly-added/modified nodes.

    Every dataset Goal in this repo explicitly forbids verify/validate/run_test/
    execute-code style nodes (they hallucinate at runtime without an oracle or
    sandbox). This check blocks such node names regardless of Goal text parsing,
    so the guarantee survives Goal rewrites. It is purely name-based: node names
    in legitimate workflows never use these anti-pattern tokens.

    Returns (violating_name, matched_keyword) on first hit, else (None, None).
    """
    for name in changed_names or ():
        lname = str(name).lower()
        for kw in _FORBIDDEN_NODE_NAME_KEYWORDS:
            if kw in lname:
                return name, kw
    return None, None


def _autoinject_terminal_evidence_input(workflow: "WorkFlowGraph", anchor_workflow: "WorkFlowGraph") -> List[str]:
    """For multi-node structures, ensure every terminal node carries at least one
    evidence-keyword input (goal/question/context/...). The runtime resolves
    such inputs from the workflow-level execution data, so no extra edge is
    needed - the existing predecessor edges keep the structure connected.

    This unblocks weak-LLM-proposed multi-stage splits (problem_framing ->
    solution_derivation -> answer_finalization) where the terminal forgets to
    declare `goal`, which would otherwise be hard-rejected by
    `_check_evidence_preservation`. Strong models that already declare `goal`
    on the terminal trigger no injection (no-op).

    Returns a list of human-readable log lines describing the injections.
    """
    logs: List[str] = []
    nodes = list(getattr(workflow, "nodes", []) or [])
    if len(nodes) < 2:
        return logs
    try:
        end_node_names = set(workflow.find_end_nodes() or [])
    except Exception:
        return logs
    evidence_keywords = ("goal", "context", "passage", "paragraph", "document", "evidence", "question")

    # Locate the canonical evidence param. Priority: the new workflow's own
    # start node(s), then the anchor workflow. Falls back to a synthesized
    # `goal:string` Parameter so the terminal still gets the runtime binding.
    def _find_param_in(graph) -> Optional[Parameter]:
        try:
            graph_nodes = list(getattr(graph, "nodes", []) or [])
        except Exception:
            return None
        try:
            start_names = set(name for name in (n.name for n in graph_nodes)
                              if not (graph.get_node_predecessors(name) or []))
        except Exception:
            start_names = set()
        for n in graph_nodes:
            if start_names and n.name not in start_names:
                continue
            for inp in (getattr(n, "inputs", []) or []):
                inp_name_lower = str(getattr(inp, "name", "") or "").lower()
                if any(kw in inp_name_lower for kw in evidence_keywords):
                    return inp
        return None

    canonical_param = _find_param_in(workflow) or _find_param_in(anchor_workflow)
    if canonical_param is None:
        try:
            canonical_param = Parameter(name="goal", type="string",
                                        description="Original task input forwarded from the workflow goal.",
                                        required=True)
        except Exception:
            return logs

    canonical_name = str(getattr(canonical_param, "name", "goal") or "goal")

    for node in nodes:
        if getattr(node, "name", "") not in end_node_names:
            continue
        input_names_lower = [str(getattr(inp, "name", "") or "").lower()
                             for inp in (getattr(node, "inputs", []) or [])]
        if any(any(kw in inp_name for kw in evidence_keywords) for inp_name in input_names_lower):
            continue
        # Avoid duplicate-by-name collisions.
        if canonical_name.lower() in input_names_lower:
            continue
        try:
            injected = copy.deepcopy(canonical_param)
        except Exception:
            try:
                injected = Parameter(name=canonical_name, type=getattr(canonical_param, "type", "string"),
                                     description=getattr(canonical_param, "description", "Original task input."),
                                     required=True)
            except Exception:
                continue
        new_inputs = list(getattr(node, "inputs", []) or [])
        new_inputs.append(injected)
        try:
            node.inputs = new_inputs
        except Exception:
            try:
                object.__setattr__(node, "inputs", new_inputs)
            except Exception:
                continue
        logs.append(f"auto-injected '{canonical_name}' into terminal '{node.name}' inputs (runtime-resolved, no edge added)")
    return logs


def _materialize_structure_rewrite(wf_generator, anchor_workflow: WorkFlowGraph, candidate: Dict[str, Any], dataset_name: str = "") -> Dict[str, Any]:
    """Build a new WorkFlowGraph from the planner-provided new_workflow spec.

    Unchanged nodes (same name, description, inputs, outputs) are deep-copied
    from anchor_workflow so their agents and prompts are preserved.  New or
    modified nodes are created from spec and sent to generate_agents.
    """
    new_wf_spec = (candidate.get("edit") or {}).get("new_workflow")
    if not isinstance(new_wf_spec, dict):
        return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: new_workflow must be a dict", "prompt_ops_by_node": {}, "selected_op_family": ""}
    nodes_spec = new_wf_spec.get("nodes") or []
    edges_spec = new_wf_spec.get("edges") or []
    if not nodes_spec:
        return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: new_workflow.nodes is empty", "prompt_ops_by_node": {}, "selected_op_family": ""}

    old_node_map = {n.name: n for n in (anchor_workflow.nodes or [])}
    built_nodes = []
    changed_node_names = set()
    seen_names = set()

    for spec in nodes_spec:
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        if name in old_node_map and not _node_spec_changed(old_node_map[name], spec):
            built_nodes.append(copy.deepcopy(old_node_map[name]))
        else:
            built_nodes.append(_instantiate_node_from_spec(spec))
            changed_node_names.add(name)

    built_edges = []
    for edge_spec in edges_spec:
        src = str(edge_spec.get("source", "")).strip()
        tgt = str(edge_spec.get("target", "")).strip()
        if src and tgt and src in seen_names and tgt in seen_names:
            built_edges.append(_instantiate_edge(src, tgt))

    try:
        workflow = WorkFlowGraph(goal=anchor_workflow.goal, nodes=built_nodes, edges=built_edges)
    except Exception as exc:
        return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: failed to build WorkFlowGraph: %s" % exc, "prompt_ops_by_node": {}, "selected_op_family": ""}

    _autorepair_global_interface(workflow)
    _autorepair_role_coverage(workflow)  # ROLE_AND_DRIFT_PATCH_V1
    # Auto-inject the canonical evidence input (goal/question/...) into
    # any terminal node that lacks one, so weak-LLM-proposed multi-stage
    # splits (e.g. problem_framing->solution_derivation->answer_finalization)
    # pass the evidence-preservation validator and reach evaluation. The
    # workflow runtime resolves such inputs from environment data, so no
    # extra edge is required.
    try:
        _injection_logs = _autoinject_terminal_evidence_input(workflow, anchor_workflow)
        for _line in _injection_logs:
            print(f">>> [structure_edit] {_line}")
    except Exception as _autoinject_exc:
        print(f">>> [structure_edit] terminal-evidence auto-inject skipped: {_autoinject_exc}")
    # Pre-validation contract pass: heading discipline / parse_mode / input-sync
    # repairs in _enforce_workflow_contracts must run BEFORE validation, otherwise
    # planner candidates whose only issue is fixable heading drift (e.g. stray
    #  headings under str/json mode) get rejected before repair runs.
    # The function is idempotent, so the post-generate_agents call below is still
    # correct on the freshly rewritten prompts.
    try:
        _legacy._enforce_workflow_contracts(workflow)
    except Exception as _enforce_exc:
        print(f">>> [structure_edit] pre-validate enforce_workflow_contracts skipped: {_enforce_exc}")
    valid, reasons, _ = _legacy._validate_workflow_structure_for_evolution(workflow)
    if not valid:
        return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: invalid structure: %s" % reasons, "prompt_ops_by_node": {}, "selected_op_family": ""}

    _violating, _kw = _check_goal_forbidden_node_names(changed_node_names)
    if _violating is not None:
        return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: new/changed node '%s' matches common FORBIDDEN keyword '%s' (verify/validate/run_test/execute-code nodes hallucinate at runtime without an oracle and strictly hurt metrics). Propose a different node role." % (_violating, _kw), "prompt_ops_by_node": {}, "selected_op_family": ""}

    # No-op structure_edit guard: reject candidates whose materialized graph is
    # semantically identical to the incumbent (same node names, same edge count,
    # no changed nodes). Such candidates only exercise LLM re-sampling noise; if
    # accepted they inflate the baseline F1 by pure variance and every later real
    # edit is then measured against the inflated number and rejected. This is
    # exactly how the MBPP plateau showed spurious 0.7442 -> 0.7558 evolution.
    # Keeps single-node / already-near-optimal benchmarks honest without blocking
    # any real structural proposal (real structure_edit always modifies at least
    # one node name or edge count).
    if not changed_node_names:
        _old_node_names = {str(getattr(n, 'name', '') or '') for n in (anchor_workflow.nodes or [])}
        _new_node_names = {str(getattr(n, 'name', '') or '') for n in built_nodes}
        _old_edge_count = len(list(anchor_workflow.edges or []))
        if _new_node_names == _old_node_names and len(built_edges) == _old_edge_count:
            return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: produced no material change (same nodes=%s and same edge_count=%d as incumbent). Re-evaluating an identical graph only measures LLM sampling noise. Propose an actual structural change (add/remove/rename a node, or add/remove an edge) or switch to prompt_edit." % (sorted(_new_node_names), _old_edge_count), "prompt_ops_by_node": {}, "selected_op_family": ""}

    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow)

    if changed_node_names:
        try:
            workflow = wf_generator.generate_agents(
                goal=anchor_workflow.goal, workflow=workflow,
                target_node_names=sorted(changed_node_names),
                dataset_name=dataset_name)
        except Exception as exc:
            return {"workflow": None, "status": "materialization_failed", "message": "structure_edit: agent generation failed: %s" % exc, "prompt_ops_by_node": {}, "selected_op_family": ""}

        # Post-generate_agents contract pass: agent generation invokes a fresh
        # LLM call that may emit extra `## <name>` headings, miss the parse_mode,
        # or drift from input bindings - violations the contract layer can fix
        # automatically. Without this second sweep, those auto-fixable defects
        # fall straight into post_safety and produce `post_generation_invalid_*`
        # rejections, wasting the candidate slot for purely cosmetic drift.
        try:
            extra_repaired, extra_mode_changed = _legacy._enforce_workflow_contracts(workflow)
            if extra_repaired or extra_mode_changed:
                print(f">>> [structure_edit] post-generate_agents contract repair: prompts_repaired={extra_repaired}, parse_mode_changed={extra_mode_changed}")
                repaired_cnt += extra_repaired
                mode_cnt += extra_mode_changed
        except Exception as _post_enforce_exc:
            print(f">>> [structure_edit] post-generate_agents enforce_workflow_contracts skipped: {_post_enforce_exc}")

    post_safety = _workflow_safety_status(workflow)
    if post_safety["prompt_errors"]:
        return {"workflow": None, "status": "materialization_failed", "validation_status": "post_generation_invalid_prompt_template", "message": "structure_edit: invalid prompt templates: %s" % post_safety["prompt_errors"][:5], "prompt_ops_by_node": {}, "selected_op_family": ""}
    if not post_safety["structure_valid"]:
        return {"workflow": None, "status": "materialization_failed", "validation_status": "post_generation_invalid_structure", "message": "structure_edit: invalid structure after generation: %s" % post_safety["structure_reasons"], "prompt_ops_by_node": {}, "selected_op_family": ""}

    instructions = candidate.get("edit", {}).get("instructions", [])
    return {
        "workflow": workflow,
        "status": "materialized",
        "message": "structure_edit applied: changed_nodes=%s, total_nodes=%d, edges=%d, prompts_repaired=%d; %s" % (
            sorted(changed_node_names), len(built_nodes), len(built_edges), repaired_cnt,
            "; ".join(str(i) for i in instructions[:3]),
        ),
        "prompt_ops_by_node": {},
        "selected_op_family": "",
    }


def _materialize_candidate(
    llm,
    wf_generator,
    anchor_workflow: WorkFlowGraph,
    candidate: Dict[str, Any],
    baseline_package,
    prompt_history: PromptHistory,
    modification_history: ModificationHistory,
    prompt_retry_per_node: int,
    dataset_name: str = "",
) -> Dict[str, Any]:
    kind = candidate["edit"]["kind"]
    if kind == "prompt_edit":
        return _materialize_prompt_edit(
            llm,
            anchor_workflow,
            candidate,
            baseline_package,
            prompt_history,
            modification_history,
            prompt_retry_per_node,
            dataset_name=dataset_name,
        )
    if kind == "params_edit":
        return _materialize_params_edit(anchor_workflow, candidate)
    if kind == "structure_edit":
        # New path: planner provides complete new_workflow spec
        if (candidate.get("edit") or {}).get("new_workflow"):
            return _materialize_structure_rewrite(wf_generator, anchor_workflow, candidate, dataset_name=dataset_name)
        # Legacy path: hardcoded style+variant dispatch
        if candidate["edit"].get("style") == "LLM_PROPOSE" and candidate["edit"].get("structure_variant") == "llm_open_structure":
            return _materialize_llm_propose_structure_edit(llm, wf_generator, anchor_workflow, candidate)
        return _materialize_structure_edit(wf_generator, anchor_workflow, candidate)
    return {"workflow": None, "status": "materialization_failed", "message": f"unsupported edit kind={kind}", "prompt_ops_by_node": {}, "selected_op_family": ""}

def _planner_record(
    iteration: int,
    candidate: Dict[str, Any],
    baseline_results: Dict[str, float],
    baseline_utility: float,
    materialization_status: str,
    validation_status: str = "",
    duplicate_kind: str = "",
    candidate_results: Optional[Dict[str, float]] = None,
    candidate_utility: Optional[float] = None,
    accepted: bool = False,
    node_metric_deltas: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_estimated_full_f1: Optional[float] = None,
    candidate_estimated_full_f1: Optional[float] = None,
) -> ModificationRecord:
    candidate_results = candidate_results or {}
    cand_utility = float(candidate_utility if candidate_utility is not None else 0.0)
    baseline_score = _safe_rate(baseline_estimated_full_f1)
    candidate_score = _safe_rate(candidate_estimated_full_f1)
    return ModificationRecord(
        iteration=int(iteration),
        candidate_id=candidate["candidate_id"],
        target_component=candidate["target"]["component"],
        target_subtype=candidate["target"]["subtype"],
        target_node_name=candidate["target"]["node_name"],
        rca_rank=int(candidate["target"]["rca_rank"]),
        edit_kind=candidate["edit"]["kind"],
        style=candidate["edit"]["style"],
        op_family=candidate["edit"].get("op_family", ""),
        structure_variant=candidate["edit"].get("structure_variant", ""),
        rationale=candidate.get("rationale", ""),
        history_reference=candidate.get("history_reference", ""),
        expected_effect=candidate.get("expected_effect", ""),
        materialization_status=materialization_status,
        validation_status=validation_status,
        duplicate_kind=duplicate_kind,
        baseline_f1=_get_primary_metric(baseline_results),
        baseline_em=_safe_rate(baseline_results.get("em", 0.0)),
        baseline_utility=float(baseline_utility),
        baseline_estimated_full_f1=baseline_score,
        candidate_f1=_get_primary_metric(candidate_results),
        candidate_em=_safe_rate(candidate_results.get("em", 0.0)),
        candidate_utility=cand_utility,
        candidate_estimated_full_f1=candidate_score,
        utility_delta=cand_utility - float(baseline_utility),
        estimated_full_f1_delta=candidate_score - baseline_score,
        reward_like_delta=cand_utility - float(baseline_utility),
        accepted=bool(accepted),
        node_metric_deltas=node_metric_deltas or {},
    )


def _print_planner_candidates(iteration: int, planner_candidates: Sequence[Dict[str, Any]]):
    view = [{"candidate_id": item["candidate_id"], "target": item["target"], "edit": {"kind": item["edit"]["kind"], "style": item["edit"]["style"], "op_family": item["edit"].get("op_family", ""), "structure_variant": item["edit"].get("structure_variant", "")}} for item in planner_candidates]
    print(f">>> [Iter {iteration}] planner candidates: {_json(view)}")



def _preview_rejections(rejections: Sequence[Dict[str, Any]], limit: int = 3) -> List[str]:
    lines: List[str] = []
    for item in list(rejections or [])[:limit]:
        candidate_id = str(item.get("candidate_id") or "?")
        rejection_type = str(item.get("rejection_type") or "unknown")
        reason = str(item.get("reason") or "").strip()
        lines.append(f"{candidate_id}:{rejection_type}:{reason}")
    return lines


def _resolve_planner_candidates_for_iteration(
    *,
    iteration: int,
    llm,
    wf_generator,
    anchor_workflow: WorkFlowGraph,
    planner_context: Dict[str, Any],
    planner_candidate_count: int,
    planner_repair_rounds: int,
    planner_candidate_regen_rounds: int,
    baseline_package,
    prompt_history: PromptHistory,
    modification_history: ModificationHistory,
    prompt_retry_per_node: int,
    evaluation_cache: EvaluationCache,
    execution_llm,
    current_eval_indices: Sequence[int],
    eval_mode: str,
    baseline_results: Dict[str, Any],
    baseline_utility: float,
    baseline_estimated_full_f1: float = 0.0,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "ok",
        "error": "",
        "planner_raw_output": "",
        "planner_raw_outputs": [],
        "planner_candidates": [],
        "ready_candidates": [],
        "planner_parse_failures": 0,
        "semantic_rejections": [],
        "materialization_rejections": [],
        "refill_rounds_used": 0,
        "projection_events": [],
    }

    # Surface the iteration index on planner_context so the deterministic
    # fallback synthesiser can rotate styles/targets across iterations and
    # avoid proposing a duplicate candidate when the LLM planner repeatedly
    # fails on small/cheap models.
    planner_context = dict(planner_context)
    planner_context.setdefault("iteration", int(iteration))

    raw_candidates, planner_raw, parse_failures = _plan_candidates_with_llm(
        llm,
        planner_context,
        planner_candidate_count,
        planner_repair_rounds,
    )
    result["planner_raw_output"] = planner_raw
    result["planner_raw_outputs"].append({"stage": "initial", "raw_output": planner_raw})
    result["planner_parse_failures"] += int(parse_failures)

    planner_candidates, invalid_slots, sanitize_info = _sanitize_planner_candidates(
        raw_candidates,
        planner_context,
        planner_candidate_count,
    )
    result["semantic_rejections"].extend(invalid_slots)
    result["projection_events"].extend(sanitize_info.get("projection_events", []))

    refill_rounds_remaining = max(0, int(planner_candidate_regen_rounds))
    while len(planner_candidates) < planner_candidate_count and refill_rounds_remaining > 0:
        missing_count = planner_candidate_count - len(planner_candidates)
        refill_rounds_remaining -= 1
        result["refill_rounds_used"] += 1
        refill_raw_candidates, refill_raw_output, refill_parse_failures = _refill_planner_candidates_with_llm(
            llm,
            planner_context,
            planner_candidates,
            result["semantic_rejections"],
            missing_count,
            planner_repair_rounds,
        )
        result["planner_raw_outputs"].append({"stage": f"refill_{result['refill_rounds_used']}", "raw_output": refill_raw_output})
        result["planner_parse_failures"] += int(refill_parse_failures)
        refill_valid, refill_invalid, refill_info = _sanitize_planner_candidates(
            refill_raw_candidates,
            planner_context,
            missing_count,
            existing_candidate_ids=[candidate["candidate_id"] for candidate in planner_candidates],
        )
        result["semantic_rejections"].extend(refill_invalid)
        result["projection_events"].extend(refill_info.get("projection_events", []))
        if not refill_valid:
            break
        planner_candidates.extend(refill_valid[:missing_count])

    if not planner_candidates:
        result["status"] = "planner_failure"
        result["error"] = "planner yielded no semantically valid candidate after sanitize/refill"
        return result

    all_seen_ids = {candidate["candidate_id"] for candidate in planner_candidates}
    ready_candidates: List[Tuple[Dict[str, Any], WorkFlowGraph, Dict[str, Any]]] = []
    pending_candidates = list(planner_candidates)
    while pending_candidates:
        failed_slots: List[Dict[str, Any]] = []
        next_pending: List[Dict[str, Any]] = []
        for planner_candidate in pending_candidates:
            _dataset_name_ctx = str(planner_context.get("dataset_name", "") or "").strip()
            materialized = _materialize_candidate(
                llm,
                wf_generator,
                anchor_workflow,
                planner_candidate,
                baseline_package,
                prompt_history,
                modification_history,
                prompt_retry_per_node,
                dataset_name=_dataset_name_ctx,
            )
            print(
                f">>> [Iter {iteration}] materialized candidate[{planner_candidate['candidate_id']}] "
                f"kind={planner_candidate['edit']['kind']}, style={planner_candidate['edit']['style']}: {materialized['message']}"
            )
            if materialized.get("workflow") is None:
                modification_history.add_record(
                    _planner_record(
                        iteration,
                        planner_candidate,
                        baseline_results,
                        baseline_utility,
                        materialized.get("status", "materialization_failed"),
                        materialized.get("validation_status", ""),
                        baseline_estimated_full_f1=baseline_estimated_full_f1,
                    )
                )
                failed_slot = {
                    "slot_index": len(ready_candidates) + len(failed_slots) + 1,
                    "candidate_id": planner_candidate["candidate_id"],
                    "reason": materialized.get("message", "materialization_failed"),
                    "rejection_type": materialized.get("validation_status", materialized.get("status", "materialization_failed")),
                }
                failed_slots.append(failed_slot)
                result["materialization_rejections"].append(failed_slot)
                all_seen_ids.add(planner_candidate["candidate_id"])
                continue

            candidate_workflow = materialized["workflow"]
            prompt_errors = _legacy._validate_workflow_prompt_templates(candidate_workflow)
            if prompt_errors:
                print(f">>> [Iter {iteration}] candidate[{planner_candidate['candidate_id']}] invalid prompt templates: {prompt_errors[:5]}")
                modification_history.add_record(
                    _planner_record(
                        iteration,
                        planner_candidate,
                        baseline_results,
                        baseline_utility,
                        "materialized",
                        "invalid_prompt_template",
                        baseline_estimated_full_f1=baseline_estimated_full_f1,
                    )
                )
                failed_slot = {
                    "slot_index": len(ready_candidates) + len(failed_slots) + 1,
                    "candidate_id": planner_candidate["candidate_id"],
                    "reason": f"invalid prompt template: {prompt_errors[:5]}",
                    "rejection_type": "invalid_prompt_template",
                }
                failed_slots.append(failed_slot)
                result["materialization_rejections"].append(failed_slot)
                all_seen_ids.add(planner_candidate["candidate_id"])
                continue
            # Only validate structure for structure_edit candidates.
            # Prompt/params edits don't change the graph topology, so the baseline's
            # structure (already accepted at generation time) remains valid.
            _cand_edit_kind = (planner_candidate.get("edit") or {}).get("kind", "")
            if _cand_edit_kind == "structure_edit":
                structure_valid, structure_reasons, _ = _legacy._validate_workflow_structure_for_evolution(candidate_workflow)
                if not structure_valid:
                    print(f">>> [Iter {iteration}] candidate[{planner_candidate['candidate_id']}] structure invalid: {structure_reasons}")
                    modification_history.add_record(
                        _planner_record(
                            iteration,
                            planner_candidate,
                            baseline_results,
                            baseline_utility,
                            "materialized",
                            "invalid_structure",
                            baseline_estimated_full_f1=baseline_estimated_full_f1,
                        )
                    )
                    failed_slot = {
                        "slot_index": len(ready_candidates) + len(failed_slots) + 1,
                        "candidate_id": planner_candidate["candidate_id"],
                        "reason": f"invalid structure: {structure_reasons}",
                        "rejection_type": "invalid_structure",
                    }
                    failed_slots.append(failed_slot)
                    result["materialization_rejections"].append(failed_slot)
                    all_seen_ids.add(planner_candidate["candidate_id"])
                    continue
            candidate_workflow, _, _ = _legacy._canonicalize_workflow_graph(candidate_workflow)
            duplicate_package = evaluation_cache.get(
                workflow_graph=candidate_workflow,
                llm=execution_llm,
                eval_indices=current_eval_indices,
                eval_mode=eval_mode,
            )
            if duplicate_package is not None:
                duplicate_kind = "incumbent_equivalent" if duplicate_package.workflow_fingerprint == baseline_package.workflow_fingerprint else "previously_seen_candidate"
                duplicate_fp = workflow_fingerprint(candidate_workflow, execution_llm, current_eval_indices, eval_mode)[:12]
                print(f">>> [Iter {iteration}] candidate[{planner_candidate['candidate_id']}] duplicate workflow ({duplicate_kind}, fp={duplicate_fp}).")
                modification_history.add_record(
                    _planner_record(
                        iteration,
                        planner_candidate,
                        baseline_results,
                        baseline_utility,
                        "materialized",
                        "duplicate",
                        duplicate_kind=duplicate_kind,
                        baseline_estimated_full_f1=baseline_estimated_full_f1,
                    )
                )
                _dup_node = str((planner_candidate.get("target") or {}).get("node_name", "") or "")
                _dup_kind = str((planner_candidate.get("edit") or {}).get("kind", "") or "")
                _dup_hint = ""
                if _dup_kind == "prompt_edit" and _dup_node:
                    _dup_hint = (
                        f" | prompt_edit on node '{_dup_node}' produced the same workflow after repair; "
                        "target a DIFFERENT node or switch to structure_edit/params_edit."
                    )
                failed_slot = {
                    "slot_index": len(ready_candidates) + len(failed_slots) + 1,
                    "candidate_id": planner_candidate["candidate_id"],
                    "reason": f"duplicate workflow ({duplicate_kind}, fp={duplicate_fp}){_dup_hint}",
                    "rejection_type": "duplicate_workflow",
                    "target_node": _dup_node,
                    "edit_kind": _dup_kind,
                }
                failed_slots.append(failed_slot)
                result["materialization_rejections"].append(failed_slot)
                all_seen_ids.add(planner_candidate["candidate_id"])
                continue
            ready_candidates.append((planner_candidate, candidate_workflow, dict(materialized.get("prompt_ops_by_node") or {})))
            all_seen_ids.add(planner_candidate["candidate_id"])

        missing_count = planner_candidate_count - len(ready_candidates)
        if missing_count <= 0 or not failed_slots or refill_rounds_remaining <= 0:
            break

        refill_rounds_remaining -= 1
        result["refill_rounds_used"] += 1
        # ROLE_AND_DRIFT_PATCH_V1: compute exhausted prompt-edit target
        # nodes from the failed-slot history and surface them on planner
        # context so the refill prompt can forbid re-targeting them.
        _dup_counts: Dict[str, int] = {}
        for _fs in failed_slots:
            if _fs.get("rejection_type") != "duplicate_workflow":
                continue
            if _fs.get("edit_kind") != "prompt_edit":
                continue
            _node = str(_fs.get("target_node") or "").strip()
            if not _node:
                continue
            _dup_counts[_node] = _dup_counts.get(_node, 0) + 1
        _exhausted_nodes = sorted(n for n, c in _dup_counts.items() if c >= 2)
        if _exhausted_nodes:
            planner_context = dict(planner_context)
            planner_context["exhausted_prompt_nodes"] = _exhausted_nodes
        refill_raw_candidates, refill_raw_output, refill_parse_failures = _refill_planner_candidates_with_llm(
            llm,
            planner_context,
            [candidate for candidate, _, _ in ready_candidates],
            failed_slots,
            missing_count,
            planner_repair_rounds,
        )
        result["planner_raw_outputs"].append({"stage": f"materialization_refill_{result['refill_rounds_used']}", "raw_output": refill_raw_output})
        result["planner_parse_failures"] += int(refill_parse_failures)
        refill_valid, refill_invalid, refill_info = _sanitize_planner_candidates(
            refill_raw_candidates,
            planner_context,
            missing_count,
            existing_candidate_ids=sorted(all_seen_ids),
        )
        result["semantic_rejections"].extend(refill_invalid)
        result["projection_events"].extend(refill_info.get("projection_events", []))
        next_pending = list(refill_valid[:missing_count])
        # Track how many CONSECUTIVE refill rounds produced zero materializable
        # candidates. Single empty round may be transient (LLM hiccup), but
        # >=3 in a row means the planner is stuck and burning budget.
        if not next_pending:
            _consec_zero = int(result.get("_consec_zero_refill", 0)) + 1
            result["_consec_zero_refill"] = _consec_zero
            if _consec_zero >= 3:
                break
            # Otherwise loop again with the same failed_slots; the next refill
            # call will retry with the accumulated rejection history.
            pending_candidates = []
            continue
        result["_consec_zero_refill"] = 0
        for candidate in next_pending:
            all_seen_ids.add(candidate["candidate_id"])
        pending_candidates = next_pending
    result["planner_candidates"] = [candidate for candidate, _, _ in ready_candidates]
    result["ready_candidates"] = ready_candidates
    if not ready_candidates:
        result["status"] = "planner_failure"
        result["error"] = "planner candidates were invalid or non-materializable after sanitize/refill"
    return result
def _print_planner_intervention_summary(modification_history: ModificationHistory):
    rows = []
    for key, bucket in modification_history.style_summary().items():
        attempts = float(bucket.get("attempts", 0.0))
        accepts = float(bucket.get("accepts", 0.0))
        mean_delta = float(bucket.get("mean_utility_delta", 0.0))
        rows.append((mean_delta, accepts / max(1.0, attempts), attempts, key))
    if not rows:
        return
    rows.sort(reverse=True)
    print("\n>>> Planner Intervention Summary:")
    for mean_delta, accept_rate, attempts, key in rows[:12]:
        edit_kind, style, op_family, structure_variant = key
        print(f"  - {edit_kind}:{style}:{op_family or '-'}:{structure_variant or '-'} -> attempts={attempts:.0f}, accepted_rate={accept_rate:.4f}, mean_utility_delta={mean_delta:.4f}")

def run_llm_workflow_optimization(
    *,
    llm,
    executor_llm=None,
    benchmark: HotPotQA,
    workflow_goal: str = WORKFLOW_GOAL,
    initial_workflow: Optional[WorkFlowGraph] = None,
    sample_k: int = 80,
    eval_seed: int = 42,
    eval_mode: str = "dev",
    target_f1: float = 0.95,
    max_opt_iterations: int = 10,
    no_improve_patience: int = 5,
    max_targets: int = 3,
    strong_rca_threshold: float = 0.05,
    prompt_retry_per_node: int = 3,
    num_workers: int = 50,
    reward_config: Optional[RewardConfig] = None,
    fixed_eval_indices: Optional[List[int]] = None,
    planner_candidate_count: int = 2,
    planner_repair_rounds: int = 1,
    planner_candidate_regen_rounds: int = 6,  # REFILL_PERSISTENCE_PATCH_V1
    adaptive_eval_enabled: bool = True,
    adaptive_success_sample_ratio: float = 0.2,
    adaptive_success_f1_threshold: float = 1.0,
    history_dir: Optional[str] = None,
) -> Dict[str, Any]:
    reward_config = reward_config or RewardConfig()
    _dataset_name = _resolve_dataset_name(benchmark)
    planner_llm = llm
    execution_llm = executor_llm or llm
    wf_generator = _legacy.WorkFlowGenerator(llm=planner_llm, tools=None)
    if initial_workflow is None:
        print(">>> Generating initial workflow from WORKFLOW_GOAL ...")
        workflow_graph = _legacy._generate_valid_initial_workflow(wf_generator=wf_generator, base_goal=workflow_goal, max_regenerations=2)
    else:
        workflow_graph = _legacy._clone_workflow_graph(initial_workflow)

    repaired_cnt, mode_cnt = _legacy._enforce_workflow_contracts(workflow_graph)
    if repaired_cnt or mode_cnt:
        print(f">>> [Init] workflow contract safety pass: prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}")

    # Cold-start repair: verify each node prompt covers input placeholders and output contracts
    cold_start_repairs = 0
    for node in workflow_graph.nodes:
        prompt_text = _legacy._get_node_primary_prompt(node) or ""
        if not prompt_text.strip():
            continue
        # Check input placeholder coverage
        missing_inputs = []
        for inp in (getattr(node, "inputs", None) or []):
            inp_name = getattr(inp, "name", "")
            if inp_name and f"{{{inp_name}}}" not in prompt_text and inp_name not in prompt_text:
                missing_inputs.append(inp_name)
        # Check output contract mention
        missing_outputs = []
        for out in (getattr(node, "outputs", None) or []):
            out_name = getattr(out, "name", "")
            if out_name and out_name not in prompt_text:
                missing_outputs.append(out_name)
        if missing_inputs or missing_outputs:
            repair_lines = []
            if missing_inputs:
                repair_lines.append("\n\nIMPORTANT: Use these input variables: " + ", ".join(f"{{{v}}}" for v in missing_inputs))
            if missing_outputs:
                repair_lines.append("\nOUTPUT CONTRACT: Your response MUST include: " + ", ".join(missing_outputs))
            new_prompt = prompt_text.rstrip() + "".join(repair_lines)
            _legacy._set_node_prompt(node, new_prompt)
            cold_start_repairs += 1
    if cold_start_repairs:
        print(f">>> [Init] cold-start prompt repair: {cold_start_repairs} node(s) had missing input/output references patched")

    agent_manager = _legacy.AgentManager(tools=None, llm=execution_llm)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=execution_llm.config)

    if fixed_eval_indices is None:
        fixed_eval_indices = _legacy._build_fixed_eval_indices(benchmark=benchmark, eval_mode=eval_mode, sample_k=sample_k, seed=eval_seed)
    if not fixed_eval_indices:
        raise ValueError("No evaluation samples available for LLM workflow optimization.")
    fixed_eval_indices = [int(idx) for idx in fixed_eval_indices]
    current_eval_indices = list(fixed_eval_indices)
    eval_subset_trace: List[Dict[str, Any]] = [
        {
            "iteration": 0,
            "subset_role": "init_full",
            "size": len(current_eval_indices),
            "first10": current_eval_indices[:10],
            "indices": list(current_eval_indices),
            "source_iteration": None,
        }
    ]
    success_rule_desc = (
        "aflow_f1==1.00"
        if float(adaptive_success_f1_threshold) >= 1.0 - 1e-12
        else f"aflow_f1>={adaptive_success_f1_threshold:.2f}"
    )
    current_subset_meta: Dict[str, Any] = {
        "source_iteration": 0,
        "success_count": 0,
        "failure_count": len(current_eval_indices),
        "retained_success_count": 0,
        "missing_record_count": 0,
        "invalid_example_count": 0,
    }
    print(f">>> Fixed eval subset prepared: size={len(fixed_eval_indices)}, seed={eval_seed}, mode={eval_mode}, first10={fixed_eval_indices[:10]}")
    if adaptive_eval_enabled:
        print(
            f">>> Adaptive eval subset enabled: retain all failures + {adaptive_success_sample_ratio:.0%} of successes "
            f"(success := {success_rule_desc})"
        )

    prompt_history = PromptHistory()
    for node in workflow_graph.nodes:
        initial_prompt = _legacy._get_node_primary_prompt(node)
        if initial_prompt:
            prompt_history.add_record(node.name, PromptRecord(iteration=0, prompt_text=initial_prompt, metrics={"f1": 0.0, "em": 0.0}, failure_prob=0.0, operations_applied=["initial"]))

    evaluation_cache = EvaluationCache()
    modification_history = ModificationHistory()
    # Load persisted history from disk (cross-run continuity)
    if history_dir:
        import json as _json_load
        _mh_path = os.path.join(history_dir, "modification_history.json")
        _ph_path = os.path.join(history_dir, "prompt_history.json")
        if os.path.isfile(_mh_path):
            try:
                with open(_mh_path, "r", encoding="utf-8") as _fload:
                    modification_history = ModificationHistory.from_dict(_json_load.load(_fload))
                print(f">>> [History] Loaded {len(modification_history.records)} modification records from {_mh_path}")
            except Exception as _load_err:
                print(f">>> [History] Failed to load modification history: {_load_err}")
        if os.path.isfile(_ph_path):
            try:
                with open(_ph_path, "r", encoding="utf-8") as _fload:
                    prompt_history = PromptHistory.from_dict(_json_load.load(_fload))
                print(f">>> [History] Loaded prompt history from {_ph_path}")
            except Exception as _load_err:
                print(f">>> [History] Failed to load prompt history: {_load_err}")
    calibration_profile: Optional[FactorCalibrationProfile] = None

    incumbent_workflow, _, _ = _legacy._canonicalize_workflow_graph(workflow_graph)
    incumbent_package, init_cached = _legacy._get_or_run_evaluation_package(
        evaluation_cache=evaluation_cache,
        workflow_graph=incumbent_workflow,
        llm=execution_llm,
        agent_manager=agent_manager,
        benchmark=benchmark,
        eval_indices=current_eval_indices,
        eval_mode=eval_mode,
        iteration=0,
        num_workers=num_workers,
        calibration_profile=calibration_profile,
        run_rca=not DISABLE_RCA,
    )
    if incumbent_package.results:
        print(f">>> [Init] raw initial workflow metrics: f1={_safe_rate(incumbent_package.results.get('f1', 0.0)):.4f}, em={_safe_rate(incumbent_package.results.get('em', 0.0)):.4f}, cache_hit={init_cached}")
    calibration_profile = build_factor_calibration_profile(evaluation_packages=evaluation_cache.values(), action_history=None)
    if calibration_profile is not None:
        print(f">>> [Init] factor calibration profile prepared: health_prior={float(calibration_profile.health_prior):.4f}")
        if not DISABLE_RCA:
            incumbent_package = _legacy._refresh_package_rca(package=incumbent_package, workflow_graph=incumbent_workflow, calibration_profile=calibration_profile)
            print(">>> [Init] refreshed initial RCA using calibrated factors (no workflow edit)")
        else:
            print(">>> [Init] DISABLE_RCA=True: skipped RCA refresh")
    if not incumbent_package.results:
        raise RuntimeError("Initial workflow evaluation produced no valid benchmark metrics. Check workflow execution warnings above.")

    incumbent_results = dict(incumbent_package.results)
    incumbent_utility = compute_workflow_utility(workflow_graph=incumbent_workflow, results=incumbent_results, config=reward_config)

    # Per-sample F1 tracker: enables fair F1 comparison across iterations with different eval subsets.
    # Each sample keeps its last-known F1; never-evaluated samples default to 0.0.
    sample_f1_tracker: Dict[int, float] = {}
    _update_sample_f1_tracker(sample_f1_tracker, benchmark, eval_mode, current_eval_indices, incumbent_package)
    estimated_init_f1 = _compute_estimated_full_f1(sample_f1_tracker, fixed_eval_indices)

    best_f1_results = dict(incumbent_results)
    best_f1_workflow = _legacy._clone_workflow_graph(incumbent_workflow)
    best_f1_package = incumbent_package
    best_f1 = estimated_init_f1
    init_f1 = estimated_init_f1

    print(f">>> [Init] cached best package prepared: aflow_f1={best_f1:.4f}, em={_safe_rate(best_f1_results.get('em', 0.0)):.4f}, utility={incumbent_utility:.4f}, cache_hit={init_cached}")
    _legacy._record_prompt_history_snapshot(workflow_graph=incumbent_workflow, root_causes=incumbent_package.root_causes, metrics=incumbent_package.results, prompt_history=prompt_history, iteration=0)

    # --- Pre-optimization simplification trial (dataset-agnostic) ---
    # Over-complex initial workflows often gate the answer through validator/formatter nodes that
    # hallucinate or drop payload. Try removing each intermediate node (pred+succ surrounded) and
    # adopt the simplification when F1 holds or improves. Uses the same materializer as the main loop
    # so agents/prompts of successor nodes are regenerated for the new input set.
    try:
        if isinstance(incumbent_workflow.nodes, list) and len(incumbent_workflow.nodes) >= 3:
            simplify_order = _select_removal_candidates(incumbent_workflow, incumbent_package.root_causes or [])
            simplify_budget = min(3, len(simplify_order))
            if simplify_budget > 0:
                print(f">>> [Init] simplification trial enabled: candidates={simplify_order[:simplify_budget]} (incumbent_nodes={len(incumbent_workflow.nodes)})")
            for removal_name in simplify_order[:simplify_budget]:
                trial_spec = _build_delete_node_new_workflow_spec(incumbent_workflow, removal_name)
                if trial_spec is None:
                    print(f">>> [Init] simplification skip target={removal_name}: cannot build spec")
                    continue
                trial_candidate = {
                    "candidate_id": f"init_simplify_delete_{removal_name}",
                    "target": {"component": "Structure", "subtype": "Coverage", "node_name": removal_name, "rca_rank": 0},
                    "edit": {"kind": "structure_edit", "style": "DELETE", "op_family": "DELETE", "structure_variant": "init_simplify", "new_workflow": trial_spec, "instructions": [f"init simplification: remove {removal_name}"]},
                    "rationale": "init simplification trial",
                    "history_reference": "",
                    "expected_effect": "reduce workflow depth without losing F1",
                }
                materialized = _materialize_structure_rewrite(wf_generator, incumbent_workflow, trial_candidate, dataset_name=_dataset_name)
                if materialized.get("status") != "materialized" or materialized.get("workflow") is None:
                    print(f">>> [Init] simplification reject target={removal_name}: {materialized.get('message', '')[:200]}")
                    continue
                trial_workflow = materialized["workflow"]
                try:
                    trial_package, trial_cached = _legacy._get_or_run_evaluation_package(
                        evaluation_cache=evaluation_cache,
                        workflow_graph=trial_workflow,
                        llm=execution_llm,
                        agent_manager=agent_manager,
                        benchmark=benchmark,
                        eval_indices=current_eval_indices,
                        eval_mode=eval_mode,
                        iteration=0,
                        num_workers=num_workers,
                        calibration_profile=calibration_profile,
                        run_rca=False,
                    )
                except Exception as _eval_exc:
                    print(f">>> [Init] simplification eval error target={removal_name}: {_eval_exc}")
                    continue
                trial_results = trial_package.results or {}
                trial_tracker = dict(sample_f1_tracker)
                _update_sample_f1_tracker(trial_tracker, benchmark, eval_mode, current_eval_indices, trial_package)
                trial_f1 = _compute_estimated_full_f1(trial_tracker, fixed_eval_indices)
                incumbent_complexity = _complexity_key(incumbent_workflow)
                trial_complexity = _complexity_key(trial_workflow)
                adopt = (trial_f1 + 1e-9 >= best_f1) and (trial_complexity < incumbent_complexity)
                print(
                    f">>> [Init] simplification trial target={removal_name}: trial_f1={trial_f1:.4f} (incumbent_f1={best_f1:.4f}), "
                    f"trial_complexity={trial_complexity}, incumbent_complexity={incumbent_complexity}, cache_hit={trial_cached}, adopt={adopt}"
                )
                if adopt:
                    incumbent_workflow = _legacy._clone_workflow_graph(trial_workflow)
                    incumbent_results = dict(trial_results)
                    if DISABLE_RCA:
                        incumbent_package = trial_package
                    else:
                        try:
                            incumbent_package = _legacy._refresh_package_rca(package=trial_package, workflow_graph=incumbent_workflow, calibration_profile=calibration_profile)
                        except Exception:
                            incumbent_package = trial_package
                    incumbent_utility = compute_workflow_utility(workflow_graph=incumbent_workflow, results=incumbent_results, config=reward_config)
                    sample_f1_tracker = trial_tracker
                    best_f1 = trial_f1
                    best_f1_results = dict(trial_results)
                    best_f1_results["estimated_full_f1"] = trial_f1
                    best_f1_workflow = _legacy._clone_workflow_graph(incumbent_workflow)
                    best_f1_package = trial_package
                    init_f1 = trial_f1
                    print(f">>> [Init] simplification accepted: removed {removal_name}, new nodes={len(incumbent_workflow.nodes)}, best_f1={best_f1:.4f}")
    except Exception as _simplify_exc:
        print(f">>> [Init] simplification trial skipped due to error: {_simplify_exc}")
    # --- end simplification trial ---


    if DISABLE_RCA:
        init_target_pool = _build_uniform_pseudo_pool(incumbent_workflow)
        init_pool_mode = "no_rca"
        init_strength = dict(_RCA_DISABLED_STRENGTH)
        print(f">>> [Init] DISABLE_RCA=True: built uniform pseudo target pool of size={len(init_target_pool)}")
    else:
        init_target_pool, init_pool_mode, _, init_strength = _legacy._build_rca_target_pool(root_causes=incumbent_package.root_causes, node_stats=incumbent_package.node_stats, node_fail_streak={}, action_history=None, max_node_fail_streak=0, max_targets=max_targets, strong_rca_threshold=strong_rca_threshold)
        _legacy._print_rca_diagnostics(iteration=0, root_causes=incumbent_package.root_causes, evidences=incumbent_package.evidences, target_pool=init_target_pool, pool_mode=init_pool_mode, strength=init_strength)
    prev_rca_snapshot = _top_rca_snapshot(init_target_pool)

    accumulated_success_indices: Set[int] = set()

    if adaptive_eval_enabled:
        current_eval_indices, init_subset_stats = _build_next_eval_subset(
            benchmark=benchmark,
            eval_mode=eval_mode,
            base_eval_indices=fixed_eval_indices,
            current_eval_indices=current_eval_indices,
            evaluation_package=incumbent_package,
            eval_seed=eval_seed,
            iteration=0,
            success_keep_ratio=adaptive_success_sample_ratio,
            success_f1_threshold=adaptive_success_f1_threshold,
            accumulated_success_indices=accumulated_success_indices,
        )
        current_subset_meta = {
            "source_iteration": 0,
            "success_count": int(init_subset_stats.get("success_count", 0) or 0),
            "failure_count": int(init_subset_stats.get("failure_count", 0) or 0),
            "retained_success_count": int(init_subset_stats.get("retained_success_count", 0) or 0),
            "missing_record_count": int(init_subset_stats.get("missing_record_count", 0) or 0),
            "invalid_example_count": int(init_subset_stats.get("invalid_example_count", 0) or 0),
        }
        eval_subset_trace.append(
            {
                "iteration": 0,
                "subset_role": "iter1_active",
                "size": len(current_eval_indices),
                "first10": current_eval_indices[:10],
                "indices": list(current_eval_indices),
                "source_iteration": 0,
                "stats": init_subset_stats,
            }
        )
        print(
            f">>> [Init] Iter 1 eval subset prepared: size={init_subset_stats['next_size']} "
            f"(successes={init_subset_stats['success_count']}, failures={init_subset_stats['failure_count']}, "
            f"retained_success={init_subset_stats['retained_success_count']}, "
            f"accumulated_pool={init_subset_stats.get('accumulated_success_pool_size', 0)}, "
            f"never_eval={init_subset_stats.get('never_evaluated_count', 0)}, "
            f"first10={init_subset_stats['first10_next']})"
        )

    no_improve_count = 0
    _incumbent_changed = True
    iteration_trace: List[Dict[str, Any]] = []

    for iter_idx in range(1, max_opt_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f">>> [Iter {iter_idx}/{max_opt_iterations}] Starting iteration")
        print(f"{'=' * 60}")
        iter_trace: Dict[str, Any] = {"iteration": iter_idx, "accepted": False}
        iter_trace["eval_subset_size"] = len(current_eval_indices)
        iter_trace["eval_subset_first10"] = current_eval_indices[:10]
        iter_trace["adaptive_subset_source_iteration"] = current_subset_meta.get("source_iteration")
        iter_trace["active_subset_size"] = len(current_eval_indices)
        iter_trace["success_count"] = int(current_subset_meta.get("success_count", 0) or 0)
        iter_trace["failure_count"] = int(current_subset_meta.get("failure_count", 0) or 0)
        iter_trace["retained_success_count"] = int(current_subset_meta.get("retained_success_count", 0) or 0)
        iter_trace["planner_parse_failures"] = 0
        iter_trace["planner_semantic_rejections"] = 0
        iter_trace["planner_materialization_rejections"] = 0
        iter_trace["planner_refill_rounds_used"] = 0
        iter_trace["planner_final_valid_candidate_count"] = 0
        iter_trace["candidate_error_summaries"] = []
        print(
            f">>> [Iter {iter_idx}] eval subset size={len(current_eval_indices)} "
            f"source_iter={current_subset_meta.get('source_iteration')} "
            f"failures={current_subset_meta.get('failure_count', 0)} "
            f"retained_success={current_subset_meta.get('retained_success_count', 0)} "
            f"first10={current_eval_indices[:10]}"
        )

        refreshed_profile = build_factor_calibration_profile(evaluation_packages=evaluation_cache.values(), action_history=None)
        if refreshed_profile is not None:
            calibration_profile = refreshed_profile

        anchor_workflow, repaired_cnt, mode_cnt = _legacy._canonicalize_workflow_graph(incumbent_workflow)
        if repaired_cnt or mode_cnt:
            print(f">>> [Iter {iter_idx}] anchor safety pass: prompts_repaired={repaired_cnt}, parse_mode_changed={mode_cnt}")
            incumbent_workflow = _legacy._clone_workflow_graph(anchor_workflow)

        if _incumbent_changed:
            if DISABLE_RCA:
                if getattr(incumbent_package, 'evidences', None):
                    baseline_package = incumbent_package
                    baseline_cached = True
                    print(f">>> [Iter {iter_idx}] DISABLE_RCA=True: baseline reused without RCA refresh")
                else:
                    baseline_package, baseline_cached = _legacy._get_or_run_evaluation_package(
                        evaluation_cache=evaluation_cache,
                        workflow_graph=anchor_workflow,
                        llm=execution_llm,
                        agent_manager=agent_manager,
                        benchmark=benchmark,
                        eval_indices=current_eval_indices,
                        eval_mode=eval_mode,
                        iteration=iter_idx,
                        num_workers=num_workers,
                        calibration_profile=calibration_profile,
                        run_rca=False,
                    )
                    print(f">>> [Iter {iter_idx}] DISABLE_RCA=True: baseline evaluated without RCA (cache_hit={baseline_cached})")
            else:
                if getattr(incumbent_package, 'evidences', None):
                    baseline_package = _legacy._refresh_package_rca(
                        package=incumbent_package,
                        workflow_graph=anchor_workflow,
                        calibration_profile=calibration_profile,
                    )
                    baseline_cached = True
                    print(f">>> [Iter {iter_idx}] baseline RCA refreshed (incumbent changed)")
                else:
                    baseline_package, baseline_cached = _legacy._get_or_run_evaluation_package(
                        evaluation_cache=evaluation_cache,
                        workflow_graph=anchor_workflow,
                        llm=execution_llm,
                        agent_manager=agent_manager,
                        benchmark=benchmark,
                        eval_indices=current_eval_indices,
                        eval_mode=eval_mode,
                        iteration=iter_idx,
                        num_workers=num_workers,
                        calibration_profile=calibration_profile,
                    )
                    print(f">>> [Iter {iter_idx}] baseline evaluated (no evidence, cache_hit={baseline_cached})")
            _incumbent_changed = False
        else:
            baseline_package = incumbent_package
            baseline_cached = True
            print(f">>> [Iter {iter_idx}] baseline reused (incumbent unchanged, same RCA)")
        baseline_results = baseline_package.results
        baseline_root_causes = baseline_package.root_causes
        baseline_evidences = baseline_package.evidences
        base_f1 = _get_primary_metric(baseline_results)
        base_em = _safe_rate(baseline_results.get("em", 0.0))
        baseline_utility = compute_workflow_utility(workflow_graph=anchor_workflow, results=baseline_results, config=reward_config)
        iter_trace["baseline_f1"] = base_f1
        iter_trace["baseline_em"] = base_em
        iter_trace["baseline_utility"] = baseline_utility
        print(f">>> [Iter {iter_idx}] baseline(best) metrics: aflow_f1={base_f1:.4f}, em={base_em:.4f}, utility={baseline_utility:.4f}, cache_hit={baseline_cached}")
        if not baseline_results:
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "empty_baseline_results"
            iteration_trace.append(iter_trace)
            print(f">>> [Iter {iter_idx}] baseline evaluation returned no valid metrics. stop.")
            break

        _legacy._record_prompt_history_snapshot(workflow_graph=anchor_workflow, root_causes=baseline_root_causes, metrics=baseline_results, prompt_history=prompt_history, iteration=iter_idx, allow_new_versions=False)
        incumbent_workflow = _legacy._clone_workflow_graph(anchor_workflow)
        incumbent_package = baseline_package
        incumbent_results = dict(baseline_results)
        incumbent_utility = baseline_utility

        # Update per-sample tracker and compute estimated full-set F1 for fair cross-iteration comparison
        _update_sample_f1_tracker(sample_f1_tracker, benchmark, eval_mode, current_eval_indices, baseline_package)
        estimated_baseline_f1 = _compute_estimated_full_f1(sample_f1_tracker, fixed_eval_indices)
        iter_trace["estimated_full_f1"] = estimated_baseline_f1
        print(f">>> [Iter {iter_idx}] estimated full-set F1={estimated_baseline_f1:.4f} (subset_f1={base_f1:.4f}, tracker_samples={len(sample_f1_tracker)}/{len(fixed_eval_indices)})")

        if estimated_baseline_f1 > best_f1:
            best_f1 = estimated_baseline_f1
            best_f1_results = dict(baseline_results)
            best_f1_results["estimated_full_f1"] = estimated_baseline_f1
            best_f1_workflow = _legacy._clone_workflow_graph(anchor_workflow)
            best_f1_package = baseline_package
        if best_f1 >= target_f1:
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "target_reached"
            iteration_trace.append(iter_trace)
            print(f">>> [Iter {iter_idx}] historical best already reached target F1={target_f1}. Done!")
            break
        if DISABLE_RCA:
            _stop_due_to_missing_signal = not baseline_evidences
        else:
            _stop_due_to_missing_signal = (not baseline_evidences) or (not baseline_root_causes)
        if _stop_due_to_missing_signal:
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "no_evidence" if DISABLE_RCA else "no_evidence_or_root_causes"
            iteration_trace.append(iter_trace)
            print(f">>> [Iter {iter_idx}] no evidence" + ("" if DISABLE_RCA else "/root causes") + ". stop.")
            break
        obs_coverage = baseline_package.obs_coverage
        print(f">>> [Iter {iter_idx}] observation coverage: {obs_coverage:.3f}")
        if obs_coverage < 0.15:
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "low_observation_coverage"
            iteration_trace.append(iter_trace)
            print(f">>> [Iter {iter_idx}] low evidence coverage (<0.15). Planner would be driven by noise, stop optimization.")
            break

        if DISABLE_RCA:
            target_pool = _build_uniform_pseudo_pool(anchor_workflow)
            pool_mode = "no_rca"
            rca_strength = dict(_RCA_DISABLED_STRENGTH)
            print(f">>> [Iter {iter_idx}] DISABLE_RCA=True: built uniform pseudo target pool of size={len(target_pool)}")
        else:
            target_pool, pool_mode, _, rca_strength = _legacy._build_rca_target_pool(root_causes=baseline_root_causes, node_stats=baseline_package.node_stats, node_fail_streak={}, action_history=None, max_node_fail_streak=0, max_targets=max_targets, strong_rca_threshold=strong_rca_threshold)
            _legacy._print_rca_diagnostics(iteration=iter_idx, root_causes=baseline_root_causes, evidences=baseline_evidences, target_pool=target_pool, pool_mode=pool_mode, strength=rca_strength)
        if not target_pool:
            no_improve_count += 1
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "empty_planner_target_pool"
            iteration_trace.append(iter_trace)
            _pool_label = "pseudo" if DISABLE_RCA else "RCA"
            print(f">>> [Iter {iter_idx}] no actionable {_pool_label} target for planner. no_improve={no_improve_count}/{no_improve_patience}")
            if no_improve_count > no_improve_patience:
                print(f">>> [Iter {iter_idx}] no improvement exceeded patience ({no_improve_patience}). stop.")
                break
            continue
        # Planner 上下文构建 & 候选方案生成
        planner_context = _planner_context(
            anchor_workflow,
            baseline_package,
            baseline_utility,
            target_pool,
            pool_mode,
            rca_strength,
            best_f1_workflow,
            best_f1_results,
            prompt_history,
            modification_history,
            previous_rca_snapshot=prev_rca_snapshot,
            baseline_estimated_full_f1=estimated_baseline_f1,
            workflow_goal=workflow_goal,
            dataset_name=_dataset_name,
            no_improve_count=no_improve_count,
        )
        prev_rca_snapshot = _top_rca_snapshot(target_pool)
        try:
            planner_resolution = _resolve_planner_candidates_for_iteration(
                iteration=iter_idx,
                llm=llm,
                wf_generator=wf_generator,
                anchor_workflow=anchor_workflow,
                planner_context=planner_context,
                planner_candidate_count=planner_candidate_count,
                planner_repair_rounds=planner_repair_rounds,
                planner_candidate_regen_rounds=planner_candidate_regen_rounds,
                baseline_package=baseline_package,
                prompt_history=prompt_history,
                modification_history=modification_history,
                prompt_retry_per_node=prompt_retry_per_node,
                evaluation_cache=evaluation_cache,
                execution_llm=execution_llm,
                current_eval_indices=current_eval_indices,
                eval_mode=eval_mode,
                baseline_results=baseline_results,
                baseline_utility=baseline_utility,
                baseline_estimated_full_f1=estimated_baseline_f1,
            )
        except Exception as exc:
            no_improve_count += 1
            iter_trace["planner_error"] = str(exc)
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "planner_failure"
            iteration_trace.append(iter_trace)
            modification_history.add_record(ModificationRecord(iteration=iter_idx, candidate_id="planner_failure", target_component="Planner", target_subtype="Planning", target_node_name="__PLANNER__", rca_rank=0, edit_kind="planner_failure", style="PLANNER_FAILURE", rationale=str(exc), materialization_status="planner_failure", validation_status="invalid_planner_output", baseline_f1=base_f1, baseline_em=base_em, baseline_utility=baseline_utility, baseline_estimated_full_f1=estimated_baseline_f1))
            print(f">>> [Iter {iter_idx}] planner failure: {exc}")
            if no_improve_count > no_improve_patience:
                print(f">>> [Iter {iter_idx}] no improvement exceeded patience ({no_improve_patience}). stop.")
                break
            continue

        iter_trace["planner_parse_failures"] = int(planner_resolution.get("planner_parse_failures", 0) or 0)
        iter_trace["planner_semantic_rejections"] = len(planner_resolution.get("semantic_rejections", []) or [])
        iter_trace["planner_materialization_rejections"] = len(planner_resolution.get("materialization_rejections", []) or [])
        iter_trace["planner_refill_rounds_used"] = int(planner_resolution.get("refill_rounds_used", 0) or 0)
        iter_trace["planner_final_valid_candidate_count"] = len(planner_resolution.get("ready_candidates", []) or [])

        for raw_info in planner_resolution.get("planner_raw_outputs", []) or []:
            raw_text = str(raw_info.get("raw_output") or "")
            stage = str(raw_info.get("stage") or "planner")
            print(f">>> [Iter {iter_idx}] planner {stage} raw output captured ({len(raw_text)} chars)")
        if planner_resolution.get("projection_events"):
            preview = planner_resolution.get("projection_events", [])[:3]
            print(f">>> [Iter {iter_idx}] planner target projections(top): {_json(preview)}")
        if planner_resolution.get("semantic_rejections"):
            print(
                f">>> [Iter {iter_idx}] planner semantic rejections={len(planner_resolution.get('semantic_rejections', []))} "
                f"preview={_preview_rejections(planner_resolution.get('semantic_rejections', []))}"
            )
        if planner_resolution.get("materialization_rejections"):
            print(
                f">>> [Iter {iter_idx}] planner materialization rejections={len(planner_resolution.get('materialization_rejections', []))} "
                f"preview={_preview_rejections(planner_resolution.get('materialization_rejections', []))}"
            )
        if planner_resolution.get("planner_candidates"):
            _print_planner_candidates(iter_idx, planner_resolution.get("planner_candidates") or [])

        if not planner_resolution.get("ready_candidates"):
            no_improve_count += 1
            iter_trace["planner_error"] = planner_resolution.get("error") or "planner produced no valid materialized candidate"
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "planner_failure"
            iteration_trace.append(iter_trace)
            modification_history.add_record(ModificationRecord(iteration=iter_idx, candidate_id="planner_failure", target_component="Planner", target_subtype="Planning", target_node_name="__PLANNER__", rca_rank=0, edit_kind="planner_failure", style="PLANNER_FAILURE", rationale=iter_trace["planner_error"], materialization_status="planner_failure", validation_status="invalid_or_unmaterializable_planner_output", baseline_f1=base_f1, baseline_em=base_em, baseline_utility=baseline_utility, baseline_estimated_full_f1=estimated_baseline_f1))
            print(f">>> [Iter {iter_idx}] planner failure: {iter_trace['planner_error']}")
            if no_improve_count > no_improve_patience:
                print(f">>> [Iter {iter_idx}] no improvement exceeded patience ({no_improve_patience}). stop.")
                break
            continue

        # --- Bonus DELETE candidate on stagnation (dataset-agnostic) ---
        # When the planner has failed to improve for 2+ iterations and the anchor workflow still
        # has more than 2 nodes, we inject a synthetic DELETE candidate. This is processed exactly
        # like a planner candidate (same materialization path, same acceptance rule). If the deletion
        # produces equal-or-better estimated_full_f1, the tie-break on complexity in
        # _aflow_f1_first_acceptance will prefer the simpler workflow.
        try:
            bonus_enabled = (
                no_improve_count >= 2
                and isinstance(anchor_workflow.nodes, list)
                and len(anchor_workflow.nodes) > 2
            )
            if bonus_enabled:
                ready_candidates_bonus = list(planner_resolution.get("ready_candidates", []) or [])
                existing_ids = {c[0].get("candidate_id", "") for c in ready_candidates_bonus}
                recent_deletes: set = set()
                try:
                    for rec in (modification_history.recent_records(12) or []):
                        if getattr(rec, "edit_kind", "") == "structure_edit" and getattr(rec, "style", "").upper().startswith("DELETE"):
                            nm = getattr(rec, "target_node_name", "") or ""
                            if nm and nm != "__STRUCTURE__":
                                recent_deletes.add(nm)
                except Exception:
                    recent_deletes = set()
                removal_order = _select_removal_candidates(anchor_workflow, getattr(baseline_package, "root_causes", None) or [])
                bonus_target = None
                for nm in removal_order:
                    if nm not in recent_deletes:
                        bonus_target = nm
                        break
                if bonus_target is None and removal_order:
                    bonus_target = removal_order[0]
                bonus_spec = _build_delete_node_new_workflow_spec(anchor_workflow, bonus_target) if bonus_target else None
                if bonus_spec is not None and bonus_target:
                    bonus_cid = f"iter{iter_idx}_bonus_delete_{bonus_target}"
                    if bonus_cid not in existing_ids:
                        bonus_candidate = {
                            "candidate_id": bonus_cid,
                            "target": {"component": "Structure", "subtype": "Coverage", "node_name": bonus_target, "rca_rank": 0},
                            "edit": {"kind": "structure_edit", "style": "DELETE", "op_family": "DELETE", "structure_variant": "bonus_stagnation", "new_workflow": bonus_spec, "instructions": [f"stagnation fallback: remove {bonus_target}"]},
                            "rationale": f"no_improve={no_improve_count}: try removing intermediate node {bonus_target}",
                            "history_reference": "",
                            "expected_effect": "simplify workflow to break stagnation without F1 loss",
                        }
                        bonus_materialized = _materialize_structure_rewrite(wf_generator, anchor_workflow, bonus_candidate, dataset_name=_dataset_name)
                        if bonus_materialized.get("status") == "materialized" and bonus_materialized.get("workflow") is not None:
                            ready_candidates_bonus.append((bonus_candidate, bonus_materialized["workflow"], {}))
                            planner_resolution["ready_candidates"] = ready_candidates_bonus
                            print(f">>> [Iter {iter_idx}] bonus DELETE candidate injected: target={bonus_target} (no_improve={no_improve_count}, nodes={len(anchor_workflow.nodes)})")
                        else:
                            print(f">>> [Iter {iter_idx}] bonus DELETE materialization failed target={bonus_target}: {bonus_materialized.get('message', '')[:200]}")
        except Exception as _bonus_exc:
            print(f">>> [Iter {iter_idx}] bonus DELETE injection skipped due to error: {_bonus_exc}")
        # --- end bonus DELETE injection ---

        # --- Bonus INSERT candidate on stagnation (small-workflow path) ---
        # Complementary to the bonus DELETE above: when no_improve_count >= 2 AND
        # the workflow has <=2 nodes (so DELETE has no safe target), inject a
        # synthetic INSERT_NODE candidate that appends a domain-agnostic
        # validator node after the terminal. This gives the optimizer a real
        # structure_edit option on workflows like HumanEval where the LLM
        # planner often only proposes Prompt-target candidates because the
        # supported_targets list is Prompt-only.
        try:
            insert_bonus_enabled = (
                no_improve_count >= 2
                and isinstance(anchor_workflow.nodes, list)
                and len(anchor_workflow.nodes) <= 2
            )
            if insert_bonus_enabled:
                ready_candidates_bonus = list(planner_resolution.get("ready_candidates", []) or [])
                existing_ids = {c[0].get("candidate_id", "") for c in ready_candidates_bonus}
                # Skip if a validator was added recently — avoid re-inserting
                # the same node every iteration once we already escaped the
                # initial plateau.
                recent_inserts: set = set()
                try:
                    for rec in (modification_history.recent_records(12) or []):
                        if getattr(rec, "edit_kind", "") == "structure_edit" and getattr(rec, "style", "").upper().startswith("INSERT"):
                            recent_inserts.add(getattr(rec, "target_node_name", "") or "")
                except Exception:
                    recent_inserts = set()
                already_has_validator = bool(_find_node_name_by_exact_names(anchor_workflow, ["refine_output", "polish_output", "output_refiner", "validate_output", "review_and_finalize"]))
                if not already_has_validator and "refine_output" not in recent_inserts and "validate_output" not in recent_inserts:
                    bonus_insert_spec = _build_insert_output_validator_new_workflow_spec(anchor_workflow)
                    if bonus_insert_spec is not None:
                        bonus_insert_cid = f"iter{iter_idx}_bonus_insert_refiner"
                        if bonus_insert_cid not in existing_ids:
                            bonus_insert_candidate = {
                                "candidate_id": bonus_insert_cid,
                                "target": {"component": "Structure", "subtype": "Coverage", "node_name": "__STRUCTURE__", "rca_rank": 0},
                                "edit": {"kind": "structure_edit", "style": "INSERT_NODE", "op_family": "INSERT", "structure_variant": "insert_output_validator", "new_workflow": bonus_insert_spec, "instructions": ["stagnation fallback: append a refine_output stage that rewrites/polishes the upstream draft (no oracle, no test cases)."]},
                                "rationale": f"no_improve={no_improve_count}: small workflow ({len(anchor_workflow.nodes)} nodes) with no DELETE target — try appending a refine_output stage to escape plateau.",
                                "history_reference": "",
                                "expected_effect": "a refine_output stage may polish the draft (clarity/format/missing fields) without altering existing prompts or introducing oracle-based checks.",
                            }
                            bonus_insert_materialized = _materialize_structure_rewrite(wf_generator, anchor_workflow, bonus_insert_candidate, dataset_name=_dataset_name)
                            if bonus_insert_materialized.get("status") == "materialized" and bonus_insert_materialized.get("workflow") is not None:
                                ready_candidates_bonus.append((bonus_insert_candidate, bonus_insert_materialized["workflow"], {}))
                                planner_resolution["ready_candidates"] = ready_candidates_bonus
                                print(f">>> [Iter {iter_idx}] bonus INSERT candidate injected: refine_output (no_improve={no_improve_count}, nodes={len(anchor_workflow.nodes)})")
                            else:
                                print(f">>> [Iter {iter_idx}] bonus INSERT materialization failed: {bonus_insert_materialized.get('message', '')[:200]}")
        except Exception as _bonus_insert_exc:
            print(f">>> [Iter {iter_idx}] bonus INSERT injection skipped due to error: {_bonus_insert_exc}")
        # --- end bonus INSERT injection ---


        evaluated_candidates: List[Tuple[Dict[str, Any], WorkFlowGraph, Any, float, Dict[str, Any], float, Dict[int, float]]] = []
        for planner_candidate, candidate_workflow, prompt_ops_by_node in planner_resolution.get("ready_candidates", []) or []:
            cand_package, cand_cached = _legacy._get_or_run_evaluation_package(
                evaluation_cache=evaluation_cache,
                workflow_graph=candidate_workflow,
                llm=execution_llm,
                agent_manager=agent_manager,
                benchmark=benchmark,
                eval_indices=current_eval_indices,
                eval_mode=eval_mode,
                iteration=iter_idx,
                num_workers=num_workers,
                calibration_profile=calibration_profile,
                run_rca=False,
            )
            candidate_results = cand_package.results
            cand_utility = compute_workflow_utility(workflow_graph=candidate_workflow, results=candidate_results, config=reward_config)
            cand_tracker = dict(sample_f1_tracker)
            _update_sample_f1_tracker(cand_tracker, benchmark, eval_mode, current_eval_indices, cand_package)
            estimated_cand_f1 = _compute_estimated_full_f1(cand_tracker, fixed_eval_indices)
            error_stats = _evaluation_execution_error_stats(
                benchmark=benchmark,
                eval_mode=eval_mode,
                eval_indices=current_eval_indices,
                evaluation_package=cand_package,
            )
            error_threshold = _candidate_execution_error_threshold(len(current_eval_indices))
            error_stats["guardrail_threshold"] = error_threshold
            error_stats["invalidated_by_error_rate"] = bool(error_stats.get("error_count", 0) > error_threshold)
            iter_trace["candidate_error_summaries"].append(
                {
                    "candidate_id": planner_candidate["candidate_id"],
                    "error_count": int(error_stats.get("error_count", 0) or 0),
                    "error_rate": float(error_stats.get("error_rate", 0.0) or 0.0),
                    "invalidated_by_error_rate": bool(error_stats.get("invalidated_by_error_rate", False)),
                    "first5_error_example_ids": list(error_stats.get("first5_error_example_ids", []) or []),
                }
            )
            print(f">>> [Iter {iter_idx}] candidate[{planner_candidate['candidate_id']}] metrics: f1={_safe_rate(candidate_results.get('f1', 0.0)):.4f}, em={_safe_rate(candidate_results.get('em', 0.0)):.4f}, estimated_full_f1={estimated_cand_f1:.4f}, utility={cand_utility:.4f}, cache_hit={cand_cached}")
            print(
                f">>> [Iter {iter_idx}] candidate[{planner_candidate['candidate_id']}] execution errors: "
                f"count={int(error_stats.get('error_count', 0))}/{len(current_eval_indices)} "
                f"rate={float(error_stats.get('error_rate', 0.0)):.2%} "
                f"first_ids={list(error_stats.get('first5_error_example_ids', []) or [])}"
            )
            if error_stats.get("invalidated_by_error_rate"):
                print(
                    f">>> [Iter {iter_idx}] candidate[{planner_candidate['candidate_id']}] invalidated by execution-error guardrail "
                    f"(threshold={error_threshold})."
                )
                modification_history.add_record(
                    _planner_record(
                        iter_idx,
                        planner_candidate,
                        baseline_results,
                        baseline_utility,
                        "evaluated",
                        "execution_error_guardrail",
                        candidate_results=candidate_results,
                        candidate_utility=cand_utility,
                        accepted=False,
                        baseline_estimated_full_f1=estimated_baseline_f1,
                        candidate_estimated_full_f1=estimated_cand_f1,
                    )
                )
                continue
            _legacy._record_prompt_history_snapshot(workflow_graph=candidate_workflow, root_causes=cand_package.root_causes, metrics=candidate_results, prompt_history=prompt_history, iteration=iter_idx, ops_by_node=prompt_ops_by_node)
            evaluated_candidates.append((planner_candidate, candidate_workflow, cand_package, cand_utility, error_stats, estimated_cand_f1, cand_tracker))

        if not evaluated_candidates:
            no_improve_count += 1
            iter_trace["best_after_iter"] = best_f1
            iter_trace["stop_reason"] = "no_valid_planner_candidate"
            iteration_trace.append(iter_trace)
            print(f">>> [Iter {iter_idx}] planner produced no valid unique candidate. no_improve={no_improve_count}/{no_improve_patience}")
            if no_improve_count > no_improve_patience:
                print(f">>> [Iter {iter_idx}] no improvement exceeded patience ({no_improve_patience}). stop.")
                break
            continue
        evaluated_candidates.sort(
            key=lambda item: _candidate_priority_key(item[1], item[2].results, score_override=item[5]),
            reverse=True,
        )
        best_candidate, best_candidate_workflow, best_candidate_package, best_candidate_utility, best_candidate_error_stats, best_candidate_estimated_f1, best_candidate_tracker = evaluated_candidates[0]
        accepted, acceptance_reason = _aflow_f1_first_acceptance(
            candidate_workflow=best_candidate_workflow,
            candidate_results=best_candidate_package.results,
            incumbent_workflow=incumbent_workflow,
            incumbent_results=incumbent_results,
            candidate_score=best_candidate_estimated_f1,
            incumbent_score=estimated_baseline_f1,
            score_label="estimated_full_f1",
        )
        incumbent_complexity_ref = _complexity_key(incumbent_workflow)
        candidate_complexity_ref = _complexity_key(best_candidate_workflow)
        iter_trace["selected_candidate_id"] = best_candidate["candidate_id"]
        iter_trace["selected_edit_kind"] = best_candidate["edit"]["kind"]
        iter_trace["selected_style"] = best_candidate["edit"]["style"]
        iter_trace["selected_target_rank"] = int(best_candidate["target"]["rca_rank"])

        for planner_candidate, candidate_workflow, cand_package, cand_utility, candidate_error_stats, estimated_cand_f1, cand_tracker in evaluated_candidates:
            candidate_accepted = accepted and planner_candidate["candidate_id"] == best_candidate["candidate_id"]
            # Compute per-node metric deltas for richer modification history
            _node_deltas = {}
            if incumbent_package.node_stats and cand_package.node_stats:
                for _nd_name in set(list(incumbent_package.node_stats.keys()) + list(cand_package.node_stats.keys())):
                    _base_ns = (incumbent_package.node_stats or {}).get(_nd_name, {})
                    _cand_ns = (cand_package.node_stats or {}).get(_nd_name, {})
                    _base_ret = _base_ns.get("return", {}) if isinstance(_base_ns, dict) else {}
                    _cand_ret = _cand_ns.get("return", {}) if isinstance(_cand_ns, dict) else {}
                    _node_deltas[_nd_name] = {
                        "task_ok_delta": round(_safe_rate(_cand_ret.get("task_ok", 0)) - _safe_rate(_base_ret.get("task_ok", 0)), 4),
                        "content_ok_delta": round(_safe_rate(_cand_ret.get("content_ok", 0)) - _safe_rate(_base_ret.get("content_ok", 0)), 4),
                    }
            modification_history.add_record(
                _planner_record(
                    iter_idx,
                    planner_candidate,
                    baseline_results,
                    baseline_utility,
                    "evaluated",
                    "valid",
                    candidate_results=cand_package.results,
                    candidate_utility=cand_utility,
                    accepted=candidate_accepted,
                    node_metric_deltas=_node_deltas,
                    baseline_estimated_full_f1=estimated_baseline_f1,
                    candidate_estimated_full_f1=estimated_cand_f1,
                )
            )
            if candidate_accepted:
                sample_f1_tracker.update(cand_tracker)
            if estimated_cand_f1 > best_f1:
                best_f1 = estimated_cand_f1
                best_f1_results = dict(cand_package.results)
                best_f1_results["estimated_full_f1"] = estimated_cand_f1
                best_f1_workflow = _legacy._clone_workflow_graph(candidate_workflow)
                best_f1_package = cand_package
                if not candidate_accepted:
                    sample_f1_tracker.update(cand_tracker)

        iter_trace["candidate_f1"] = _get_primary_metric(best_candidate_package.results)
        iter_trace["candidate_em"] = _safe_rate(best_candidate_package.results.get("em", 0.0))
        iter_trace["candidate_estimated_full_f1"] = best_candidate_estimated_f1
        iter_trace["candidate_execution_error_count"] = int(best_candidate_error_stats.get("error_count", 0) or 0)
        iter_trace["candidate_execution_error_rate"] = float(best_candidate_error_stats.get("error_rate", 0.0) or 0.0)
        iter_trace["candidate_invalidated_by_error_rate"] = bool(best_candidate_error_stats.get("invalidated_by_error_rate", False))
        iter_trace["candidate_utility"] = best_candidate_utility
        iter_trace["candidate_complexity"] = candidate_complexity_ref
        iter_trace["acceptance_reason"] = acceptance_reason
        iter_trace["accepted_by_rule"] = accepted
        if accepted:
            incumbent_workflow = _legacy._clone_workflow_graph(best_candidate_workflow)
            incumbent_results = dict(best_candidate_package.results)
            incumbent_package = best_candidate_package
            incumbent_utility = best_candidate_utility
            no_improve_count = 0
            _incumbent_changed = True
            iter_trace["accepted"] = True
            iter_trace["best_after_iter"] = best_f1
            print(
                f">>> [Iter {iter_idx}] selected best candidate[{best_candidate['candidate_id']}] accepted as new incumbent "
                f"(reason={acceptance_reason}, estimated_full_f1={best_candidate_estimated_f1:.4f}, subset_f1={_safe_rate(best_candidate_package.results.get('f1', 0.0)):.4f}, "
                f"candidate_complexity={candidate_complexity_ref}, incumbent_complexity={incumbent_complexity_ref})"
            )
        else:
            # Adaptive patience consumption: a candidate that clearly regresses
            # utility (>=0.03 below incumbent) is a stronger no-improvement signal
            # than a tie / near-tie, because the planner is actively damaging the
            # workflow rather than stalling. Consume patience faster in that case
            # so the optimizer cannot burn iterations on a plateau where every
            # candidate strictly hurts. Threshold 0.03 is safe because the reward
            # scale for F1-dominated utility is O(0.5); normal search noise is
            # well under that, and any true improvement would be accepted anyway.
            _util_delta = best_candidate_utility - incumbent_utility
            _patience_bump = 2 if _util_delta < -0.03 else 1
            no_improve_count += _patience_bump
            iter_trace["best_after_iter"] = best_f1
            _decay_note = f" [accelerated +{_patience_bump}: utility regressed by {-_util_delta:.3f}]" if _patience_bump > 1 else ""
            print(
                f">>> [Iter {iter_idx}] selected best candidate[{best_candidate['candidate_id']}] did not beat incumbent "
                f"(reason={acceptance_reason}, cand_estimated_full_f1={best_candidate_estimated_f1:.4f}, "
                f"incumbent_estimated_full_f1={estimated_baseline_f1:.4f}, cand_subset_f1={_safe_rate(best_candidate_package.results.get('f1', 0.0)):.4f}, "
                f"incumbent_subset_f1={_safe_rate(incumbent_results.get('f1', 0.0)):.4f}, cand_complexity={candidate_complexity_ref}, "
                f"incumbent_complexity={incumbent_complexity_ref}). no_improve={no_improve_count}/{no_improve_patience}{_decay_note}"
            )
            if no_improve_count > no_improve_patience:
                iteration_trace.append(iter_trace)
                print(f">>> [Iter {iter_idx}] no improvement exceeded patience ({no_improve_patience}). stop.")
                break
        # Fix #10: Active convergence detection - monitor search space coverage
        if iter_idx >= 3 and modification_history:
            style_stats = modification_history.style_summary()
            total_styles_tried = len([k for k, v in style_stats.items() if v.get("attempts", 0) >= 1])
            total_possible_styles = len(_PROMPT_STYLES) + len(_PARAM_STYLES) + 1  # +1 for structure_edit (direct rewrite)
            coverage_ratio = total_styles_tried / max(1, total_possible_styles)
            recent_5 = modification_history.recent_records(5)
            recent_5_all_failed = len(recent_5) >= 5 and all(not r.accepted for r in recent_5)
            recent_5_avg_delta = sum(r.utility_delta for r in recent_5) / max(1, len(recent_5)) if recent_5 else 0.0
            if coverage_ratio >= 0.7 and recent_5_all_failed and recent_5_avg_delta <= 0.0:
                print(
                    f">>> [Iter {iter_idx}] convergence detected: coverage={coverage_ratio:.2f}, "
                    f"last5_all_failed={recent_5_all_failed}, avg_delta={recent_5_avg_delta:.4f}. "
                    f"Search space substantially explored with diminishing returns. Stopping."
                )
                iter_trace["stop_reason"] = "convergence_detected"
                iteration_trace.append(iter_trace)
                break
        if adaptive_eval_enabled:
            current_eval_indices, subset_stats = _build_next_eval_subset(
                benchmark=benchmark,
                eval_mode=eval_mode,
                base_eval_indices=fixed_eval_indices,
                current_eval_indices=current_eval_indices,
                evaluation_package=incumbent_package,
                eval_seed=eval_seed,
                iteration=iter_idx,
                success_keep_ratio=adaptive_success_sample_ratio,
                success_f1_threshold=adaptive_success_f1_threshold,
                accumulated_success_indices=accumulated_success_indices,
            )
            current_subset_meta = {
                "source_iteration": iter_idx,
                "success_count": int(subset_stats.get("success_count", 0) or 0),
                "failure_count": int(subset_stats.get("failure_count", 0) or 0),
                "retained_success_count": int(subset_stats.get("retained_success_count", 0) or 0),
                "missing_record_count": int(subset_stats.get("missing_record_count", 0) or 0),
                "invalid_example_count": int(subset_stats.get("invalid_example_count", 0) or 0),
            }
            iter_trace["next_eval_subset"] = subset_stats
            eval_subset_trace.append(
                {
                    "iteration": iter_idx,
                    "subset_role": "next_iteration",
                    "size": len(current_eval_indices),
                    "first10": current_eval_indices[:10],
                    "indices": list(current_eval_indices),
                    "source_iteration": iter_idx,
                    "stats": subset_stats,
                }
            )
            print(
                f">>> [Iter {iter_idx}] next eval subset prepared: size={subset_stats['next_size']} "
                f"(successes={subset_stats['success_count']}, failures={subset_stats['failure_count']}, "
                f"retained_success={subset_stats['retained_success_count']}, "
                f"accumulated_pool={subset_stats.get('accumulated_success_pool_size', 0)}, "
                f"never_eval={subset_stats.get('never_evaluated_count', 0)}, "
                f"missing={subset_stats['missing_record_count']}, "
                f"first10={subset_stats['first10_next']})"
            )
        else:
            eval_subset_trace.append(
                {
                    "iteration": iter_idx,
                    "subset_role": "fixed",
                    "size": len(current_eval_indices),
                    "first10": current_eval_indices[:10],
                    "indices": list(current_eval_indices),
                }
            )

        iteration_trace.append(iter_trace)

    workflow_graph = incumbent_workflow
    print(f"\n{'=' * 60}")
    print(">>> LLM planner optimization finished.")
    final_estimated_f1 = _compute_estimated_full_f1(sample_f1_tracker, fixed_eval_indices)
    print(f">>> Best estimated full-set F1={best_f1:.4f} (final tracker F1={final_estimated_f1:.4f}, tracked_samples={len(sample_f1_tracker)}/{len(fixed_eval_indices)})")
    print(f">>> Best metrics across <= {max_opt_iterations} iterations: {best_f1_results}")
    incumbent_complexity = {k: round(v, 4) for k, v in workflow_complexity_metrics(incumbent_workflow).items()}
    print(f">>> Final incumbent summary: aflow_f1={_safe_rate(incumbent_results.get('f1', 0.0)):.4f}, em={_safe_rate(incumbent_results.get('em', 0.0)):.4f}, utility={incumbent_utility:.4f}, complexity={incumbent_complexity})")
    _legacy._print_iteration_f1_trace(init_f1=init_f1, iteration_trace=iteration_trace)
    _print_planner_intervention_summary(modification_history)

    print("\n>>> Prompt Evolution Summary:")
    for node_name in prompt_history.get_all_node_names():
        records = prompt_history.get_history(node_name)
        if len(records) > 1:
            first_f1 = _get_primary_metric(records[0].metrics)
            best_rec = prompt_history.get_best_record(node_name)
            best_f1_node = _get_primary_metric(best_rec.metrics) if best_rec else 0.0
            print(f"  - {node_name}: {len(records)} versions, F1 {first_f1:.4f} -> {best_f1_node:.4f} (best @ iter {best_rec.iteration if best_rec else 0})")

    # Persist history to disk (cross-run continuity)
    if history_dir:
        import json as _json_h
        try:
            os.makedirs(history_dir, exist_ok=True)
            with open(os.path.join(history_dir, "modification_history.json"), "w", encoding="utf-8") as _fh:
                _json_h.dump(modification_history.to_dict(), _fh, indent=2)
            with open(os.path.join(history_dir, "prompt_history.json"), "w", encoding="utf-8") as _fh:
                _json_h.dump(prompt_history.to_dict(), _fh, indent=2)
            print(f">>> [History] Saved history to {history_dir}")
        except Exception as _he:
            print(f">>> [History] Failed to save history: {_he}")
    return {
        "workflow_graph": workflow_graph,
        "best_workflow": best_f1_workflow,
        "best_results": best_f1_results,
        "best_package": best_f1_package,
        "incumbent_workflow": incumbent_workflow,
        "incumbent_results": incumbent_results,
        "incumbent_package": incumbent_package,
        "incumbent_utility": incumbent_utility,
        "iteration_trace": iteration_trace,
        "prompt_history": prompt_history,
        "fixed_eval_indices": fixed_eval_indices,
        "active_eval_indices": current_eval_indices,
        "eval_subset_trace": eval_subset_trace,
        "agent_manager": agent_manager,
        "evaluation_cache": evaluation_cache,
        "sample_f1_tracker": dict(sample_f1_tracker),
        "estimated_full_f1": best_f1,
        "calibration_profile": calibration_profile,
        "modification_history": modification_history,
    }


class LLMWorkflowOptimizer(Optimizer):
    graph: Optional[WorkFlowGraph] = Field(default=None, description="Initial workflow graph. If missing, generate from workflow_goal.")
    evaluator: Optional[Evaluator] = Field(default=None, description="Optional evaluator. Default path uses inline evaluation runner.")
    executor_llm: Optional[Any] = Field(default=None, description="LLM used to execute workflow agents during evaluation.")
    workflow_goal: str = Field(default=WORKFLOW_GOAL, description="Goal used to generate the initial workflow when graph is missing.")
    sample_k: int = Field(default=80, description="Number of fixed evaluation samples.")
    eval_seed: int = Field(default=42, description="Seed for fixed evaluation subset.")
    eval_mode: str = Field(default="dev", description="Benchmark split used during optimization.")
    target_f1: float = Field(default=0.95, description="Early-stop target F1.")
    no_improve_patience: int = Field(default=5, description="Stop after this many consecutive non-improving iterations.")
    max_targets: int = Field(default=3, description="Maximum actionable targets from RCA.")
    strong_rca_threshold: float = Field(default=0.05, description="Threshold separating strong RCA from weak RCA.")
    prompt_retry_per_node: int = Field(default=3, description="Maximum prompt rewrite attempts per node.")
    num_workers: int = Field(default=50, description="Parallel evaluation workers.")
    planner_candidate_count: int = Field(default=2, description="Number of planner candidates generated each iteration.")
    planner_repair_rounds: int = Field(default=1, description="Maximum JSON repair rounds for planner output.")
    planner_candidate_regen_rounds: int = Field(default=6, description="Maximum targeted candidate refill rounds per iteration before counting planner failure.")
    adaptive_eval_enabled: bool = Field(default=True, description="Whether to shrink the next iteration's eval subset to failures plus a fraction of successes.")
    adaptive_success_sample_ratio: float = Field(default=0.2, description="Fraction of successful samples retained for the next iteration.")
    adaptive_success_f1_threshold: float = Field(default=1.0, description="Per-sample AFlow F1 full-match threshold used to mark a sample as successful for adaptive resampling.")
    reward_config: RewardConfig = Field(default_factory=RewardConfig, description="Utility configuration.")
    iteration_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Optimization trace.")
    best_results: Dict[str, float] = Field(default_factory=dict, description="Best metrics observed so far.")
    fixed_eval_indices: Optional[List[int]] = Field(default=None, description="Optional precomputed fixed evaluation subset.")
    active_eval_indices: Optional[List[int]] = Field(default=None, description="Current adaptive evaluation subset after optimization.")
    eval_subset_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Per-iteration adaptive evaluation subset trace.")
    calibration_profile: Optional[FactorCalibrationProfile] = Field(default=None, description="Factor calibration profile used during optimization.")
    modification_history: Optional[ModificationHistory] = Field(default=None, description="History of planner modifications.")
    history_dir: Optional[str] = Field(default=None, description="Directory to persist optimization history across runs.")

    def optimize(self, dataset: HotPotQA, **kwargs):
        payload = run_llm_workflow_optimization(llm=self.llm, executor_llm=self.executor_llm, benchmark=dataset, workflow_goal=self.workflow_goal, initial_workflow=self.graph, sample_k=self.sample_k, eval_seed=self.eval_seed, eval_mode=self.eval_mode, target_f1=self.target_f1, max_opt_iterations=self.max_steps, no_improve_patience=self.no_improve_patience, max_targets=self.max_targets, strong_rca_threshold=self.strong_rca_threshold, prompt_retry_per_node=self.prompt_retry_per_node, num_workers=self.num_workers, reward_config=self.reward_config, fixed_eval_indices=self.fixed_eval_indices, planner_candidate_count=self.planner_candidate_count, planner_repair_rounds=self.planner_repair_rounds, planner_candidate_regen_rounds=self.planner_candidate_regen_rounds, adaptive_eval_enabled=self.adaptive_eval_enabled, adaptive_success_sample_ratio=self.adaptive_success_sample_ratio, adaptive_success_f1_threshold=self.adaptive_success_f1_threshold, history_dir=self.history_dir)
        self.graph = payload["workflow_graph"]
        self.best_results = dict(payload["best_results"])
        self.iteration_trace = list(payload["iteration_trace"])
        self.fixed_eval_indices = list(payload["fixed_eval_indices"])
        self.active_eval_indices = list(payload.get("active_eval_indices") or [])
        self.eval_subset_trace = list(payload.get("eval_subset_trace") or [])
        self.calibration_profile = payload.get("calibration_profile")
        self.modification_history = payload.get("modification_history")
        self._agent_manager = payload.get("agent_manager")
        return payload

    def step(self, **kwargs):
        raise NotImplementedError("LLMWorkflowOptimizer.step is encoded inside optimize() as a planner-driven loop.")

    def evaluate(self, dataset, eval_mode: str = "test", graph: Optional[WorkFlowGraph] = None, indices: Optional[List[int]] = None, **kwargs) -> dict:
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError("No workflow graph is available for evaluation.")
        execution_llm = self.executor_llm or self.llm
        agent_manager = getattr(self, "_agent_manager", None)
        if agent_manager is None:
            from ..agents.agent_manager import AgentManager as _AM
            agent_manager = _AM(tools=None, llm=execution_llm)
            agent_manager.add_agents_from_workflow(graph, llm_config=execution_llm.config)

        use_aflow_hotpotqa = isinstance(dataset, AFlowHotPotQA)
        use_math = isinstance(dataset, (MATH, AFlowMATH))
        use_humaneval = isinstance(dataset, (HumanEval, AFlowHumanEval))
        use_mbpp = isinstance(dataset, (MBPP, AFlowMBPP))
        use_gsm8k = isinstance(dataset, (GSM8K, AFlowGSM8K))
        use_drop = isinstance(dataset, AFlowDROP)

        def _collate(example):
            if use_math:
                prompt = example["problem"]
                return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt}
            if use_humaneval:
                prompt = example["prompt"]
                entry_point = example.get("entry_point", "")
                return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt, "entry_point": entry_point}
            if use_mbpp:
                prompt = example["prompt"]
                entry_point = example.get("entry_point", "")
                if entry_point:
                    prompt = prompt.rstrip() + "\n\nIMPORTANT: The generated function MUST be named `" + entry_point + "`. Return ONLY the function definition (and any needed helpers/imports). Do NOT include asserts, test cases, check() functions, or explanations."
                return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt, "entry_point": entry_point}
            if use_drop:
                prompt = example["context"]
                return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt}
            if use_gsm8k:
                prompt = example["question"]
                return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt}
            if use_aflow_hotpotqa:
                paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
                ctx = "\n".join(" ".join(p) for p in paragraphs)
                prompt = "Context: " + ctx + "\n\nQuestion: " + example["question"] + "\n\nAnswer:"
                return {"goal": prompt, "problem": prompt}
            prompt = "### User Question:\n" + example["question"] + "\n\n"
            ctx_parts = []
            for item in example["context"]:
                ctx_parts.append("Document [" + str(item[0]) + "]: " + "".join(item[1]))
            prompt += "### Reference Documents:\n" + "\n".join(ctx_parts)
            prompt += ("\n\n### Instruction:\nBased ONLY on the Reference Documents above, "
                       "extract the precise entity name that answers the User Question.")
            return {"goal": prompt, "problem": prompt}

        def _postprocess_math(output):
            """Post-process for MATH: preserve \boxed{} and LaTeX, do minimal cleaning."""
            import re as _re
            import regex as _regex
            if isinstance(output, dict):
                output = output.get("answer", output.get("result", output))
            if output is None:
                return ""
            content = str(output).strip()
            _xml_m = _re.search(r"<answer>\s*(.*?)\s*</answer>", content, _re.DOTALL | _re.IGNORECASE)
            if _xml_m:
                content = _xml_m.group(1).strip()
            content = content.replace("```json", "").replace("```", "")
            boxed_pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
            boxed_matches = _regex.findall(boxed_pattern, content, _regex.DOTALL)
            if boxed_matches:
                return "\\boxed{" + boxed_matches[-1].strip() + "}"
            _ans_m = _re.search(r"(?:(?:final )?answer\s*(?:is|:|=)\s*)(.+?)\s*[.]?\s*$", content, _re.IGNORECASE | _re.DOTALL)
            if _ans_m:
                return _ans_m.group(1).strip().rstrip(".")
            _lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            if _lines:
                return _lines[-1]
            return content.strip()

        def _postprocess_drop(output):
            """DROP: dict/JSON unwrap + markdown section extraction + prefix stripping. Matches AFlow pipeline and training postprocess (rl_workflow_optimizer)."""
            import re as _re
            import json as _json
            if isinstance(output, dict):
                output = output.get("answer", output.get("result", output))
            if output is None:
                return ""
            content = str(output).strip()
            if content.startswith("{"):
                try:
                    _parsed = _json.loads(content)
                    if isinstance(_parsed, dict):
                        for _k in ("answer", "result", "output", "response", "final_answer"):
                            if _k in _parsed:
                                content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            # AgentGenerator emits  template. Extract the last  /  section body so parse_mode=str does not leak reasoning prose.
            _section_pattern = _re.compile(r"(?im)^#{1,4}\s*(?:answer|final[_\s-]*answer|final[_\s-]*output)\s*$")
            _matches = list(_section_pattern.finditer(content))
            if _matches:
                _last = _matches[-1]
                _body = content[_last.end():]
                _next_header = _re.search(r"(?m)^#{1,4}\s+\S", _body)
                if _next_header:
                    _body = _body[:_next_header.start()]
                content = _body.strip()
            # Secondary fallback: if preamble tokens like  / Reasoning: remain, keep only the last non-empty line.
            _lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            if len(_lines) > 1:
                _lowered_pre = " ".join(_lines[:-1]).lower()
                if any(_kw in _lowered_pre for _kw in ("thought", "reasoning", "step ", "explain", "because", "therefore")):
                    content = _lines[-1]
            content = _re.sub(r"^\s*(?:Answer|Final answer|The answer is|The final answer is)[: \s]+", "", content, flags=_re.IGNORECASE).strip()
            content = content.strip().strip('"').strip("'").strip(".").strip(",").strip()
            return content

        def _postprocess_hotpotqa(output):
            import re as _re
            import json as _json
            import string as _string

            if isinstance(output, dict):
                output = output.get("answer", output.get("result", output))
            if output is None:
                return ""
            content = str(output).strip()

            # --- JSON unwrapping (str-mode nodes that still output JSON) ---
            if content.startswith("{"):
                try:
                    _parsed = _json.loads(content)
                    if isinstance(_parsed, dict):
                        for _k in ("answer", "result", "output", "response", "final_answer", "inferred_answer"):
                            if _k in _parsed:
                                content = str(_parsed[_k]).strip()
                                break
                        else:
                            for _v in _parsed.values():
                                if isinstance(_v, str) and _v.strip():
                                    content = _v.strip()
                                    break
                except (ValueError, TypeError):
                    pass

            # XML answer tags
            _xml_m = _re.search(r"<answer>\s*(.*?)\s*</answer>", content, _re.DOTALL | _re.IGNORECASE)
            if _xml_m:
                content = _xml_m.group(1).strip()
            content = content.replace("```json", "").replace("```", "")

            # --- Prefix stripping (safe patterns only; no "The"/"It is" removal) ---
            def _strip_prefixes(text):
                text = _re.sub(r"^\s*(?:Answer|Final answer|The answer is|The final answer is|The extracted output is|The direct answer is|The output is|Result|Output)[:\s]+", "", text, flags=_re.IGNORECASE)
                text = _re.sub(r"^\s*(?:The\s+)?(?:answer|result|output)\s+(?:is|was|=)[:\s]*", "", text, flags=_re.IGNORECASE)
                text = _re.sub(r"^(?:Based on\b[^,]{0,120},\s*)", "", text, flags=_re.IGNORECASE)
                text = _re.sub(r"^(?:According to\b[^,]{0,120},\s*)", "", text, flags=_re.IGNORECASE)
                text = _re.sub(r"^(?:From (?:the )?(?:workflow|execution|context|results?|information|evidence)[^,]{0,80},\s*)", "", text, flags=_re.IGNORECASE)
                text = _re.sub(r"^(?:After\b[^,]{0,120},\s*)", "", text, flags=_re.IGNORECASE)
                return text.strip()

            content = _strip_prefixes(content)

            # Take first non-empty line
            _lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            if _lines:
                content = _lines[0]
            content = _strip_prefixes(content)

            # --- First-sentence truncation ---
            # Cut at ". " / "! " / "? " followed by uppercase (sentence boundary)
            # Only if at least 2 words precede the split to protect abbreviations
            _ABBREVS = {"dr", "mr", "mrs", "ms", "jr", "sr", "st", "prof", "gen", "col", "sgt", "lt", "capt", "rev", "hon", "vs", "etc", "inc", "ltd", "co", "no"}
            for _sm in _re.finditer(r"[.!?]\s+[A-Z]", content):
                _before = content[:_sm.start() + 1].strip()
                if len(_before.split()) >= 2:
                    _last_word = _before.rstrip(".").split()[-1].lower()
                    if _last_word in _ABBREVS:
                        continue
                    content = _before
                    break

            # --- Trailing explanation truncation ---
            # "Yes, because..." / "X, who is..." / "X, which..." etc.
            _comma_expl = _re.search(
                r"[,;]\s+(?:because|since|as |which|who|that |but |and |however|this |it |the |where|when|a |an |is |was |are |were )",
                content, _re.IGNORECASE
            )
            if _comma_expl:
                content = content[:_comma_expl.start()].strip()

            # Strip quotes and trailing punctuation
            content = content.strip().strip('"').strip("'").strip(".").strip(",").strip()

            # --- AFlow-style normalize_answer (lowercase, remove articles/punctuation, fix whitespace) ---
            def _normalize_answer(s):
                s = s.lower()
                exclude = set(_string.punctuation)
                s = "".join(ch for ch in s if ch not in exclude)
                s = _re.sub(r"\b(a|an|the)\b", " ", s)
                return " ".join(s.split())

            return _normalize_answer(content)

        def _postprocess_humaneval(output):
            """Post-process for HumanEval: extract code from LLM output.
            Uses code_extract() as fallback — analogous to normalize_answer for HotPotQA
            and extract_answer for MATH.
            """
            import re as _re
            import json as _json
            from evoagentx.utils.sanitize import code_extract as _code_extract
            if isinstance(output, dict):
                output = output.get("code", output.get("solution", output.get("answer", output.get("result", output))))
            if output is None:
                return ""
            content = str(output).strip()
            # JSON unwrapping
            if content.startswith("{"):
                try:
                    _parsed = _json.loads(content)
                    if isinstance(_parsed, dict):
                        for _k in ("code", "solution", "answer", "result", "output"):
                            if _k in _parsed:
                                content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            # Extract code from markdown fences
            _fence_m = _re.search(r"```(?:python)?\s*\n(.*?)```", content, _re.DOTALL)
            if _fence_m:
                content = _fence_m.group(1).strip()
            # AFlow-style code_extract: find the longest syntactically valid Python block.
            # This handles cases where the LLM mixes natural language with code.
            extracted = _code_extract(content)
            if extracted and extracted.strip():
                content = extracted
            return content
        def _postprocess_mbpp(output):
            """MBPP: extract code from LLM output. Matches AFlow CodeFormatter pipeline."""
            import re as _re
            import json as _json
            from evoagentx.utils.sanitize import code_extract as _code_extract
            if isinstance(output, dict):
                output = output.get("code", output.get("solution", output.get("answer", output.get("result", output))))
            if output is None:
                return ""
            content = str(output).strip()
            # JSON unwrapping
            if content.startswith("{"):
                try:
                    _parsed = _json.loads(content)
                    if isinstance(_parsed, dict):
                        for _k in ("code", "solution", "answer", "result", "output"):
                            if _k in _parsed:
                                content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            # AFlow-style: extract ALL markdown code blocks (not just first)
            _py_blocks = _re.findall(r"```python\s*([\s\S]*?)\s*```", content)
            if _py_blocks:
                content = "\n\n".join(_py_blocks)
            else:
                _gen_blocks = _re.findall(r"```\s*([\s\S]*?)\s*```", content)
                if _gen_blocks:
                    content = "\n\n".join(_gen_blocks)
            # Find longest syntactically valid Python block
            extracted = _code_extract(content)
            if extracted and extracted.strip():
                content = extracted
            return content

        def _postprocess_gsm8k(output):
            """GSM8K: dict/JSON unwrap + answer normalization (strip $, commas, units)."""
            import re as _re
            import json as _json
            if isinstance(output, dict):
                output = output.get('answer', output.get('result', output))
            if output is None:
                return ''
            content = str(output).strip()
            # JSON unwrap
            if content.startswith('{'):
                try:
                    _parsed = _json.loads(content)
                    if isinstance(_parsed, dict):
                        for _k in ('answer', 'result', 'output', 'response', 'final_answer'):
                            if _k in _parsed:
                                content = str(_parsed[_k]).strip()
                                break
                except (_json.JSONDecodeError, ValueError):
                    pass
            # AgentGenerator template often emits `## Thought ... ## answer <value>`. Extract the last `## answer` / `## final_answer` section body so the GSM8K scoring regex (extract_last_number) does not pick numbers out of the reasoning prose that would override the correct final number. Matches rl_workflow_optimizer training postprocess.
            _section_pattern = _re.compile(r"(?im)^#{1,4}\s*(?:answer|final[_\s-]*answer|final[_\s-]*output)\s*$")
            _matches = list(_section_pattern.finditer(content))
            if _matches:
                _last = _matches[-1]
                _body = content[_last.end():]
                _next_header = _re.search(r"(?m)^#{1,4}\s+\S", _body)
                if _next_header:
                    _body = _body[:_next_header.start()]
                content = _body.strip()
            # Secondary fallback: if reasoning preamble tokens (`## Thought`, "Reasoning:", "Step 1:", etc.) remain with multiple lines, keep only the last non-empty line so the scoring regex sees just the bare number.
            _lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
            if len(_lines) > 1:
                _lowered_pre = ' '.join(_lines[:-1]).lower()
                if any(_kw in _lowered_pre for _kw in ('thought', 'reasoning', 'step ', 'explain', 'because', 'therefore')):
                    content = _lines[-1]
            # Normalize: strip $, commas, units, trailing punctuation from every line
            _normalized_lines = []
            for _line in content.split('\n'):
                _l = _line.strip()
                _l = _l.replace('$', '').replace(',', '')
                _l = _re.sub(r'\s*(dollars|cents|percent|%|units?|items?|people|hours?|minutes?|seconds?|days?|weeks?|months?|years?|miles?|km|meters?|feet|inches?|pounds?|kg|grams?|liters?|gallons?|pieces?|pairs?|boxes?|bags?|bottles?|cups?|slices?|tickets?|books?|pages?|students?|children|adults?|apples?|oranges?|balls?|cars?|dogs?|cats?|birds?|fish|eggs?|cookies?|candies|marbles?|coins?|stamps?|stickers?|flowers?|trees?|shirts?|toys?|games?|points?|goals?|runs?|laps?|trips?|times?|ways?|groups?|rows?|columns?|layers?|levels?|steps?|blocks?|miles?|km|mph|kph)\s*$', '', _l, flags=_re.IGNORECASE)
                _l = _l.rstrip('.')
                _normalized_lines.append(_l)
            return '\n'.join(_normalized_lines)

        _postprocess = _postprocess_drop if use_drop else (_postprocess_gsm8k if use_gsm8k else (_postprocess_math if use_math else (_postprocess_mbpp if use_mbpp else (_postprocess_humaneval if use_humaneval else _postprocess_hotpotqa))))

        num_workers = getattr(self, "num_workers", 50) or 50
        evaluator = Evaluator(
            llm=execution_llm,
            agent_manager=agent_manager,
            collate_func=_collate,
            output_postprocess_func=_postprocess,
            verbose=True,
            num_workers=num_workers,
        )

        print(f">>> [Evaluate] Running evaluation: mode={eval_mode}, indices={'all' if indices is None else len(indices)}, num_workers={num_workers}")
        from evoagentx.core.callbacks import suppress_logger_info
        with suppress_logger_info():
            results = evaluator.evaluate(
                graph=graph,
                benchmark=dataset,
                eval_mode=eval_mode,
                indices=indices,
                update_agents=True,
            )

        # Recompute metrics from evaluation records for reliability
        _extract_primary_score = _get_primary_metric  # module-level helper

        evaluation_records = evaluator.get_all_evaluation_records()
        if evaluation_records:
            if indices:
                total_f1, total_em, counted = 0.0, 0.0, 0
                for idx in indices:
                    example = dataset.get_example_by_index(index=int(idx), mode=eval_mode)
                    if example is None:
                        continue
                    _raw_eid = dataset.get_id(example=example)
                    record = evaluation_records.get(_raw_eid) or evaluation_records.get(str(_raw_eid))
                    if record:
                        _m = record.get("metrics") or {}
                        total_f1 += _extract_primary_score(_m)
                        total_em += float(_m.get("em", _m.get("pass@1", 0.0)))
                        counted += 1
                n = max(1, counted)
                results = {"f1": total_f1 / n, "em": total_em / n, "count": counted}
            else:
                total_f1 = sum(_extract_primary_score(r.get("metrics") or {}) for r in evaluation_records.values())
                total_em = sum(float((r.get("metrics") or {}).get("em", (r.get("metrics") or {}).get("pass@1", 0.0))) for r in evaluation_records.values())
                n = max(1, len(evaluation_records))
                results = {"f1": total_f1 / n, "em": total_em / n, "count": n}

        print(f">>> [Evaluate] Results: {results}")
        return results or {}

    def convergence_check(self, *args, **kwargs) -> bool:
        return False