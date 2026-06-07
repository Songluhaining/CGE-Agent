import copy
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

# ==============================================================================
# 0) Universal patch: sanitize workflow-agent configs before BaseModule.from_dict
# ==============================================================================
from evoagentx.core.module import BaseModule
from evoagentx.core.module_utils import parse_json_from_text, repair_and_load_json


_original_from_dict_func = BaseModule.from_dict.__func__


def _sanitize_data_recursively(data: Any) -> Any:
    """
    Remove unused input variables from agent dicts to avoid downstream validation
    failures when workflow generator creates extra placeholders.
    """
    if isinstance(data, dict):
        if (
            "inputs" in data
            and "prompt" in data
            and isinstance(data.get("inputs"), list)
            and isinstance(data.get("prompt"), str)
        ):
            prompt_text = data["prompt"]
            valid_inputs = []
            for inp in data["inputs"]:
                var_name = inp.get("name") if isinstance(inp, dict) else getattr(inp, "name", None)
                if var_name and f"{{{var_name}}}" not in prompt_text:
                    continue
                valid_inputs.append(inp)
            data["inputs"] = valid_inputs

        for _, value in data.items():
            if isinstance(value, (dict, list)):
                _sanitize_data_recursively(value)
    elif isinstance(data, list):
        for item in data:
            _sanitize_data_recursively(item)
    return data


def _safe_load_json(s: str):
    """json.loads with conservative repair fallback.

    Switching planner LLMs (gemini / qwen / deepseek / openrouter ...) is the
    classic source of "valid-looking JSON that json.loads rejects" -- LaTeX
    backslashes, smart-quote escapes, outer-quote wrapping, etc. The repair
    helper in module_utils handles the common cases without touching
    well-formed payloads, so we always try the cheap path first and only
    escalate on failure.
    """
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return repair_and_load_json(s)
    except Exception:
        return None


def _extract_first_json_value(text: str, required_keys: Optional[List[str]] = None) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        json_candidates = parse_json_from_text(text)
    except Exception:
        json_candidates = []

    parsed_candidates: List[Tuple[str, Any]] = []
    for s in json_candidates:
        parsed = _safe_load_json(s)
        if parsed is None:
            continue
        parsed_candidates.append((s, parsed))
    if not parsed_candidates:
        return None

    parsed_candidates.sort(key=lambda item: len(item[0]), reverse=True)
    if required_keys:
        req = set(required_keys)
        for _, parsed in parsed_candidates:
            if isinstance(parsed, dict) and req.issubset(set(parsed.keys())):
                return parsed
    return parsed_candidates[0][1]


def _extract_json_field_value(text: str, field_name: str) -> Any:
    if not isinstance(text, str) or not text.strip() or not field_name:
        return None
    try:
        json_candidates = parse_json_from_text(text)
    except Exception:
        json_candidates = []

    parsed_candidates: List[Tuple[str, Any]] = []
    for s in json_candidates:
        parsed = _safe_load_json(s)
        if parsed is None:
            continue
        parsed_candidates.append((s, parsed))

    parsed_candidates.sort(key=lambda item: len(item[0]), reverse=True)
    for _, parsed in parsed_candidates:
        if isinstance(parsed, dict) and field_name in parsed:
            return parsed[field_name]
    return None


def _coerce_value_for_field(value: Any, annotation: Any) -> Any:
    ann = str(annotation).lower() if annotation is not None else ""

    if "dict" in ann:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = _extract_first_json_value(value)
            if isinstance(parsed, dict):
                return parsed
        return {"value": value if isinstance(value, str) else str(value)}

    if "list" in ann:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            parsed = _extract_first_json_value(value)
            if isinstance(parsed, list):
                return parsed
        return [value]

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _looks_like_subtask_spec(data: Any) -> bool:
    return isinstance(data, dict) and all(key in data for key in ("name", "description", "inputs", "outputs"))


def _looks_like_generated_agent_spec(data: Any) -> bool:
    return _looks_like_subtask_spec(data) and isinstance(data.get("prompt"), str) and bool(data.get("prompt").strip())


_TASK_PLANNING_SUBTASK_ALLOWED_KEYS = ("name", "description", "reason", "inputs", "outputs")


def _sanitize_subtask_dict(subtask: Any) -> Any:
    """Drop planner-hallucinated keys that do not belong on a TaskPlanning sub_task.

    The TaskPlanning prompt schema only exposes name/description/reason/inputs/
    outputs. Dataset Goal texts sometimes instruct the planner to also set agent-
    level fields ("agent dict MUST set parse_mode"); the planner then invents an
    ``agents`` key with incomplete dicts (missing name/description) that fail
    WorkFlowNode validation. Agent creation is the downstream AgentGenerator's
    job, so we strip those extras here defensively. Well-formed sub_tasks are
    unchanged; only extra keys are dropped.
    """
    if not isinstance(subtask, dict):
        return subtask
    sanitized = {k: v for k, v in subtask.items() if k in _TASK_PLANNING_SUBTASK_ALLOWED_KEYS}
    return sanitized


def _repair_task_planning_output(cls, data: Any) -> Any:
    if not isinstance(data, dict) or getattr(cls, "__name__", "") != "TaskPlanningOutput":
        return data
    if isinstance(data.get("sub_tasks"), list):
        cleaned = [_sanitize_subtask_dict(st) for st in data["sub_tasks"]]
        if cleaned != data["sub_tasks"]:
            data = dict(data)
            data["sub_tasks"] = cleaned
        return data

    content = data.get("content")
    parsed_content = _extract_first_json_value(content, required_keys=["sub_tasks"]) if isinstance(content, str) else None
    if parsed_content is None and isinstance(content, str):
        parsed_content = _extract_first_json_value(content)

    if isinstance(parsed_content, dict) and isinstance(parsed_content.get("sub_tasks"), list):
        repaired = dict(data)
        repaired["sub_tasks"] = parsed_content["sub_tasks"]
        return repaired

    if _looks_like_subtask_spec(data):
        subtask = {k: copy.deepcopy(v) for k, v in data.items() if k in ("name", "description", "reason", "inputs", "outputs")}
        repaired = {"sub_tasks": [subtask]}
        if "content" in data:
            repaired["content"] = data["content"]
        return repaired

    if isinstance(parsed_content, list):
        repaired = dict(data)
        repaired["sub_tasks"] = parsed_content
        return repaired

    if _looks_like_subtask_spec(parsed_content):
        repaired = dict(data)
        repaired["sub_tasks"] = [{k: copy.deepcopy(v) for k, v in parsed_content.items() if k in ("name", "description", "reason", "inputs", "outputs")}]
        return repaired

    return data


def _repair_agent_generation_output(cls, data: Any) -> Any:
    if not isinstance(data, dict) or getattr(cls, "__name__", "") != "AgentGenerationOutput":
        return data

    content = data.get("content")
    parsed_content = (
        _extract_first_json_value(content, required_keys=["selected_agents", "generated_agents"])
        if isinstance(content, str)
        else None
    )
    if parsed_content is None and isinstance(content, str):
        parsed_content = _extract_first_json_value(content)

    repaired = dict(data)
    if isinstance(parsed_content, dict):
        if "selected_agents" in parsed_content and "selected_agents" not in repaired:
            repaired["selected_agents"] = parsed_content.get("selected_agents")
        if "generated_agents" in parsed_content and "generated_agents" not in repaired:
            repaired["generated_agents"] = parsed_content.get("generated_agents")

    if "selected_agents" not in repaired:
        repaired["selected_agents"] = []
    if "generated_agents" not in repaired:
        if _looks_like_generated_agent_spec(data):
            repaired["generated_agents"] = [
                {k: copy.deepcopy(v) for k, v in data.items() if k in ("name", "description", "inputs", "outputs", "prompt", "tool_names")}
            ]
        elif _looks_like_generated_agent_spec(parsed_content):
            repaired["generated_agents"] = [
                {k: copy.deepcopy(v) for k, v in parsed_content.items() if k in ("name", "description", "inputs", "outputs", "prompt", "tool_names")}
            ]
        else:
            # Bare task/subtask dict is not a valid GeneratedAgent. Keep this empty so workflow_generator falls back cleanly.
            repaired["generated_agents"] = []

    if not isinstance(repaired.get("selected_agents"), list):
        repaired["selected_agents"] = _coerce_value_for_field(repaired.get("selected_agents"), list)
    if not isinstance(repaired.get("generated_agents"), list):
        repaired["generated_agents"] = _coerce_value_for_field(repaired.get("generated_agents"), list)

    return repaired


def _repair_action_output_missing_fields(cls, data: Any) -> Any:
    """
    Best-effort repair for ActionOutput parsing failures:
    if required output field is missing (e.g., facts), map extracted payload/content
    into that field before pydantic validation.
    """
    if not isinstance(data, dict):
        return data
    cls_name = getattr(cls, "__name__", "")
    if "ActionOutput" not in cls_name:
        return data

    model_fields = getattr(cls, "model_fields", None) or {}
    if not model_fields:
        return data

    payload_fields = [k for k in model_fields.keys() if k not in ("class_name", "content")]
    if not payload_fields:
        return data

    missing_required: List[str] = []
    for key in payload_fields:
        field_info = model_fields.get(key)
        try:
            is_required = bool(field_info.is_required())
        except Exception:
            is_required = False
        if is_required and key not in data:
            missing_required.append(key)

    if not missing_required:
        return data

    content = data.get("content")
    parsed_from_content = _extract_first_json_value(content) if isinstance(content, str) else None
    extra_payload = {k: v for k, v in data.items() if k not in model_fields and k != "class_name"}

    for miss_key in missing_required:
        if miss_key in data:
            continue
        source_val = None

        content_field_val = _extract_json_field_value(content, miss_key) if isinstance(content, str) else None
        if content_field_val is not None:
            source_val = content_field_val
        elif isinstance(parsed_from_content, dict) and miss_key in parsed_from_content:
            source_val = parsed_from_content[miss_key]
        elif extra_payload:
            source_val = extra_payload if len(extra_payload) > 1 else next(iter(extra_payload.values()))
        elif parsed_from_content is not None:
            source_val = parsed_from_content
        elif content:
            source_val = content

        if source_val is None:
            continue

        ann = getattr(model_fields.get(miss_key), "annotation", None)
        data[miss_key] = _coerce_value_for_field(source_val, ann)

    return data



def _fill_none_with_defaults(cls, data: dict) -> dict:
    """Convert None values to type-appropriate defaults for required Pydantic fields.

    This prevents ValidationError when upstream LLM output has null/None for
    required string (or other) fields.  Works for ActionInput, ActionOutput,
    and any BaseModule subclass.
    """
    if not isinstance(data, dict):
        return data
    model_fields = getattr(cls, 'model_fields', None) or {}
    if not model_fields:
        return data
    for key, field_info in model_fields.items():
        # Skip non-None values and internal fields
        if key in ('class_name', 'content'):
            continue
        if key in data and data[key] is not None:
            continue
        ann = str(getattr(field_info, 'annotation', '') or '').lower()
        # Check if the field has a default (not required) -- skip those
        try:
            is_required = bool(field_info.is_required())
        except Exception:
            is_required = True
        if not is_required:
            continue
        # Map None -> type-appropriate empty value
        if 'list' in ann:
            data[key] = []
        elif 'dict' in ann:
            data[key] = {}
        elif 'int' in ann or 'float' in ann:
            data[key] = 0
        elif 'bool' in ann:
            data[key] = False
        else:
            # Default: treat as string
            data[key] = ''
    return data


def patched_from_dict(cls, data: dict, **kwargs):
    try:
        data_copy = copy.deepcopy(data)
    except Exception:
        data_copy = data
    data_copy = _repair_task_planning_output(cls, data_copy)
    data_copy = _repair_agent_generation_output(cls, data_copy)
    data_copy = _repair_action_output_missing_fields(cls, data_copy)
    _sanitize_data_recursively(data_copy)
    _fill_none_with_defaults(cls, data_copy)
    # Ensure all dict keys are strings (Pydantic requires string keywords)
    if isinstance(data_copy, dict):
        data_copy = {str(k): v for k, v in data_copy.items()}
    return _original_from_dict_func(cls, data_copy, **kwargs)


BaseModule.from_dict = classmethod(patched_from_dict)
print(">>> [System] EvoAgentX BaseModule patch applied (sanitize + output-field auto-repair).")

# ==============================================================================
# 1) Business imports
# ==============================================================================
from evoagentx.agents import AgentManager
from evoagentx.benchmark import AFlowHotPotQA, HotPotQA, AFlowMATH, MATH, HumanEval, AFlowHumanEval, MBPP, AFlowMBPP, GSM8K, AFlowGSM8K, AFlowDROP
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.evaluators import Evaluator
from evoagentx.models import LiteLLMConfig
from evoagentx.models.litellm_model import LiteLLM
from evoagentx.models.model_utils import cost_manager
from evoagentx.optimizers.rl import (
    ActionOutcomeHistory,
    ActionSpec,
    EvaluationCache,
    EvaluationPackage,
    FactorCalibrationProfile,
    OptimizationCandidate,
    PromptHistory,
    PromptRecord,
    RcaTarget,
    RewardConfig,
    build_factor_calibration_profile,
    compute_workflow_utility,
    top_actionable_failure_prob,
    workflow_complexity_metrics,
    workflow_fingerprint,
)
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph

from flagent.Graph.evidence import (
    EvidenceBuffer,
    SampleEvidence,
    build_multi_sample_factor_graph,
    compute_backward_consistency_scores,
    enrich_evidence_with_llm_judge,
    extract_judge_payloads_from_trajectory,
    extract_local_obs_from_trajectory as extract_obs_from_trajectory,
)


load_dotenv()
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")

_STRUCTURE_ROLE_ORDER = {
    "understand": 0,
    "gather": 1,
    "transform": 2,
    "reason": 3,
    "verify": 4,
    "produce": 5,
    "decompose": 0,
    "extract": 1,
    "evidence": 1,
    "synthesize": 3,
    "answer": 5,
    "generic": 6,
}


WORKFLOW_GOAL = """
# Task Description
Design a high-quality executable workflow that directly solves the runtime task instance provided in `goal`.

# Core Requirement
The resulting workflow will be executed later on many concrete task instances. It must transform the runtime input `goal`
into the final task answer.

Important distinction:
- Right now, you are producing the workflow definition itself.
- Later, the returned workflow nodes will be executed on runtime tasks.
- Therefore, the nodes you define must be future runtime task-solving steps, NOT steps for designing, synthesizing,
  serializing, validating, or emitting the workflow definition itself.
- Do NOT create nodes such as `workflow_design`, `workflow_synthesis`, `answer_serialization`, `constraint_extraction`,
  `goal_analysis`, or similar meta-planning steps unless the runtime user task explicitly asks for those artifacts.

# Workflow Logic
1. Input: receive a variable named `goal` containing the full task description and any bundled context.
2. Break execution into 3 to 6 connected workflow nodes with clear, non-overlapping responsibilities.
3. The workflow should cover at least three distinct functional roles chosen from:
   - goal understanding / scoping / decomposition
   - signal gathering / extraction / inspection
   - transformation / normalization / organization
   - reasoning / comparison / synthesis / decision making
   - verification / critique / refinement
   - final answer generation
4. Each non-final node MUST emit explicit intermediate variables that are consumed by downstream nodes.
5. Reuse `goal` only when a node truly needs the original task context; otherwise prefer structured inputs from upstream nodes.
6. The final node MUST return the final answer as a variable named `answer`.

# Constraints
- Input variable MUST be named `goal`.
- Output variable MUST be named `answer`.
- Workflow MUST contain at least 3 nodes and at most 6 nodes.
- Nodes MUST be dependency-connected by meaningful shared variable names so edges can be inferred.
- At least one intermediate output other than `answer` MUST be consumed downstream.
- Prefer informative, task-solving variable names over generic names like `data`, `result`, or `info`.
- Intermediate variables should be task-domain artifacts that help solve `goal`, not workflow-schema artifacts such as
  `required_inputs`, `expected_output`, `functional_roles_needed`, `structural_constraints`, `execution_plan`,
  `workflow_skeleton`, or `synthesized_workflow`.
- Node responsibilities should reflect actual task-solving work, not workflow-design or meta-planning.
- The final node outputs the runtime task answer `answer`; it does not serialize the workflow definition itself.
- Keep the workflow mostly acyclic and concise; only introduce optional feedback inputs when refinement is clearly useful.
- Do not assume unavailable external tools; rely on the provided inputs unless a tool is explicitly available.
- Keep sub-task names, reasons, and descriptions concise to avoid verbose planning output.
"""

_META_WORKFLOW_CUES = {
    "required_inputs",
    "expected_output",
    "expected_outputs",
    "functional_roles_needed",
    "structural_constraints",
    "execution_plan",
    "workflow_skeleton",
    "synthesized_workflow",
    "workflow_plan",
    "workflow_spec",
    "workflow_specification",
    "subtask_definitions",
}


def _safe_rate(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _stats_are_healthy(stats: Optional[Dict[str, float]], threshold: float = 0.95) -> bool:
    stats = stats or {}
    if not stats:
        return True
    return all(_safe_rate(v, 1.0) >= threshold for v in stats.values())


def _clone_workflow_graph(graph: WorkFlowGraph) -> WorkFlowGraph:
    try:
        return copy.deepcopy(graph)
    except Exception:
        return WorkFlowGraph(goal=graph.goal, graph=graph)


def _canonicalize_workflow_graph(graph: WorkFlowGraph) -> Tuple[WorkFlowGraph, int, int]:
    canonical = _clone_workflow_graph(graph)
    repaired_cnt, mode_cnt = _enforce_workflow_contracts(canonical)
    return canonical, repaired_cnt, mode_cnt


def _normalize_op_family(op_family: Any) -> str:
    family = str(op_family or '').upper().strip()
    return family if family in PROMPT_OP_FAMILY_CATALOG else ''


def _filter_operations_by_family(
    operations: List[Dict[str, Any]],
    preferred_op_family: str,
) -> List[Dict[str, Any]]:
    family = _normalize_op_family(preferred_op_family)
    if not family:
        return list(operations or [])
    filtered: List[Dict[str, Any]] = []
    for op in operations or []:
        if str((op or {}).get('op') or '').upper().strip() == family:
            filtered.append(op)
    return filtered



def _parse_health_node_name(name: str) -> Optional[Tuple[str, str, str]]:
    # Edge 子部件已折叠为单一 HealthEdge；Structure 层已移除。
    prefixes = [
        ("HealthPromptBinding_", "Prompt", "Binding"),
        ("HealthPromptContract_", "Prompt", "Contract"),
        ("HealthPromptGrounding_", "Prompt", "Grounding"),
        ("HealthParamsLength_", "Params", "Length"),
        ("HealthParamsParse_", "Params", "Parse"),
        ("HealthReturnType_", "Return", "Type"),
        ("HealthReturnEvidence_", "Return", "Evidence"),
        ("HealthReturnTask_", "Return", "Task"),
        ("HealthEdge_", "Edge", "Edge"),
        ("HealthPrompt_", "Prompt", "Prompt"),
        ("HealthParams_", "Params", "Params"),
        ("HealthReturn_", "Return", "Return"),
    ]
    for prefix, component, subtype in prefixes:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if component == "Edge" and "__TO__" in suffix:
                return component, subtype, suffix.split("__TO__", 1)[1]
            return component, subtype, suffix
    return None


def _parse_edge_suffix(name: str) -> Tuple[Optional[str], Optional[str]]:
    suffix = ""
    for prefix in ("HealthEdge_",):
        if isinstance(name, str) and name.startswith(prefix):
            suffix = name[len(prefix) :]
            break
    if "__TO__" not in suffix:
        return None, None
    source_name, target_name = suffix.split("__TO__", 1)
    return source_name or None, target_name or None


def _infer_roles_from_text_parts(text_parts: List[str]) -> List[str]:
    text = " ".join(str(x) for x in text_parts if x).lower()
    roles = set()

    if any(cue in text for cue in ("understand", "scope", "decompose", "break down", "analyze", "analyse", "parse", "clarify", "plan", "frame", "query", "intent", "requirement", "requirements")):
        roles.add("understand")
    if any(cue in text for cue in ("gather", "collect", "extract", "retrieve", "search", "inspect", "read", "observe", "evidence", "context", "source", "signal", "signals", "fact", "facts", "record", "records")):
        roles.add("gather")
    if any(cue in text for cue in ("transform", "organize", "organise", "normalize", "normalise", "structure", "filter", "cluster", "map", "summarize", "summarise", "rank", "prepare")):
        roles.add("transform")
    if any(cue in text for cue in ("reason", "synthesize", "synthesise", "infer", "compare", "evaluate", "decide", "diagnose", "prioritize", "prioritise", "deduce", "derive")):
        roles.add("reason")
    if any(cue in text for cue in ("verify", "validate", "review", "check", "critique", "test", "audit", "refine")):
        roles.add("verify")
    if any(cue in text for cue in ("answer", "respond", "write", "generate", "compose", "deliver", "final", "finalize", "finalise", "report", "output")):
        roles.add("produce")

    return sorted(roles) if roles else ["generic"]


def _infer_workflow_node_roles(node) -> List[str]:
    text_parts = [
        getattr(node, "name", ""),
        getattr(node, "description", ""),
    ]
    text_parts.extend(getattr(inp, "name", "") for inp in getattr(node, "inputs", []) or [])
    text_parts.extend(getattr(out, "name", "") for out in getattr(node, "outputs", []) or [])
    return _infer_roles_from_text_parts(text_parts)


def _workflow_role_meta(workflow_graph: WorkFlowGraph) -> Dict[str, Any]:
    covered = set()
    per_node: Dict[str, List[str]] = {}
    for node in workflow_graph.nodes:
        roles = _infer_workflow_node_roles(node)
        per_node[node.name] = roles
        covered.update(role for role in roles if role != "generic")
    return {"covered_roles": sorted(covered), "count": len(covered), "per_node": per_node}


def _count_goal_input_nodes(workflow_graph: WorkFlowGraph) -> int:
    count = 0
    for node in getattr(workflow_graph, "nodes", []) or []:
        input_names = {getattr(inp, "name", "").lower() for inp in getattr(node, "inputs", []) or []}
        if "goal" in input_names:
            count += 1
    return count


def _count_invalid_edge_contracts(workflow_graph: WorkFlowGraph) -> int:
    invalid_edges = 0
    for edge in getattr(workflow_graph, "edges", []) or []:
        try:
            source_node = workflow_graph.get_node(edge.source)
            target_node = workflow_graph.get_node(edge.target)
        except Exception:
            invalid_edges += 1
            continue
        source_outputs = {getattr(param, "name", "").lower() for param in getattr(source_node, "outputs", []) or []}
        target_inputs = {getattr(param, "name", "").lower() for param in getattr(target_node, "inputs", []) or []}
        if len(source_outputs & target_inputs) == 0:
            invalid_edges += 1
    return invalid_edges


def _meta_workflow_signal(workflow_graph: WorkFlowGraph) -> Tuple[int, List[str]]:
    signal_hits: List[str] = []
    cue_hits = 0
    for node in getattr(workflow_graph, "nodes", []) or []:
        name = str(getattr(node, "name", "") or "")
        description = str(getattr(node, "description", "") or "")
        input_names = [str(getattr(inp, "name", "") or "") for inp in getattr(node, "inputs", []) or []]
        output_names = [str(getattr(out, "name", "") or "") for out in getattr(node, "outputs", []) or []]
        joined = " ".join([name, description] + input_names + output_names).lower()
        matched_cues = sorted(cue for cue in _META_WORKFLOW_CUES if cue in joined)
        if matched_cues:
            cue_hits += len(matched_cues)
            signal_hits.append(f"{name or '<unnamed>'}: {', '.join(matched_cues)}")
        if "workflow" in joined and any(term in joined for term in ("design", "construct", "specification", "skeleton", "structure")):
            cue_hits += 1
            signal_hits.append(f"{name or '<unnamed>'}: workflow-design language")
    return cue_hits, signal_hits[:6]


def _count_consumed_intermediate_outputs(workflow_graph: WorkFlowGraph) -> int:
    count = 0
    for edge in getattr(workflow_graph, "edges", []) or []:
        try:
            source_node = workflow_graph.get_node(edge.source)
            target_node = workflow_graph.get_node(edge.target)
        except Exception:
            continue
        source_outputs = {getattr(param, "name", "").lower() for param in getattr(source_node, "outputs", []) or []}
        target_inputs = {getattr(param, "name", "").lower() for param in getattr(target_node, "inputs", []) or []}
        consumed = {name for name in (source_outputs & target_inputs) if name and name != "answer"}
        count += len(consumed)
    return count


def _expected_terminal_output_names(workflow_graph: WorkFlowGraph) -> set:
    """Names that count as a valid terminal output for this workflow.

    Always includes the legacy default `answer`. Also includes whatever
    name the per-benchmark goal contract declared (e.g. `final_answer`
    for MATH). Without this, a goal that mandates `final_answer` would
    fail the interface check forever even though the LLM faithfully
    obeyed the contract.
    """
    names = {"answer"}
    try:
        contract = _parse_goal_contract(getattr(workflow_graph, "goal", "") or "")
        n = (contract.get("terminal_output_name") or "").strip().lower()
        if n:
            names.add(n)
    except Exception:
        pass
    return names


def _workflow_interface_ok(workflow_graph: WorkFlowGraph) -> bool:
    try:
        initial_nodes = workflow_graph.find_initial_nodes() or []
        end_nodes = workflow_graph.find_end_nodes() or []
    except Exception:
        return False
    if not initial_nodes or not end_nodes:
        return False

    has_goal_input = False
    for node_name in initial_nodes:
        node = workflow_graph.get_node(node_name)
        required_inputs = {getattr(inp, "name", "").lower() for inp in getattr(node, "inputs", []) or [] if getattr(inp, "required", True)}
        if "goal" in required_inputs:
            has_goal_input = True
            break

    expected_names = _expected_terminal_output_names(workflow_graph)
    has_answer_output = False
    for node_name in end_nodes:
        node = workflow_graph.get_node(node_name)
        required_outputs = {getattr(out, "name", "").lower() for out in getattr(node, "outputs", []) or [] if getattr(out, "required", True)}
        if required_outputs & expected_names:
            has_answer_output = True
            break

    return has_goal_input and has_answer_output


def _parse_goal_contract(workflow_goal: str) -> Dict[str, Any]:
    """Parse the workflow.goal text for explicit contract directives.

    The dataset goal text is the per-benchmark contract authored by the
    user. It may declare:
      - parse_mode: e.g. `parse_mode": "str"` or `parse_mode": "title"`
      - verbatim sentences the terminal prompt MUST contain
      - few-shot anchor line(s) the terminal prompt MUST include
      - terminal output param name (defaults to `answer`)

    Returning a dict with whatever was found, so the auto-repair can
    enforce the actual contract instead of a hard-coded HotpotQA bias.
    """
    import re as _re
    contract: Dict[str, Any] = {}
    if not workflow_goal:
        return contract

    # parse_mode (look for `"parse_mode": "<mode>"` or `parse_mode: <mode>`)
    pm_match = _re.search(
        r"""parse_mode["'`]?\s*:\s*["'`]?(title|str|json)\b""",
        workflow_goal,
        _re.IGNORECASE,
    )
    if pm_match:
        contract["parse_mode"] = pm_match.group(1).lower()

    # Verbatim sentences (rule pattern: "MUST contain this sentence literally: `...`")
    verbatim_rules: List[str] = []
    for m in _re.finditer(
        r"MUST contain this sentence literally:?\s*`([^`]+)`",
        workflow_goal,
        _re.IGNORECASE,
    ):
        s = m.group(1).strip()
        if s:
            verbatim_rules.append(s)

    # One-line anchor (rule pattern: "MUST include this exact one-line anchor: `...`")
    for m in _re.finditer(
        r"MUST include this exact one-line anchor:?\s*`([^`]+)`",
        workflow_goal,
        _re.IGNORECASE,
    ):
        s = m.group(1).strip()
        if s and s not in verbatim_rules:
            verbatim_rules.append(s)

    if verbatim_rules:
        contract["verbatim_rules"] = verbatim_rules

    # Terminal output parameter name (defaults to `answer`)
    out_match = _re.search(
        r"name`?\s+is\s+exactly\s+`?(\w+)`?\s*\(?lowercase",
        workflow_goal,
        _re.IGNORECASE,
    )
    if out_match:
        contract["terminal_output_name"] = out_match.group(1).lower()
    else:
        contract["terminal_output_name"] = "answer"

    return contract


def _check_evidence_preservation(workflow_graph: WorkFlowGraph) -> List[str]:
    """Multi-node QA workflows often forward only summaries between nodes,
    leaving the terminal answer node without the raw context. For HotpotQA
    style multi-hop tasks this collapses to F1<<0.5. Flag terminal nodes
    that do not list `goal` (or any input named like context/passage/
    paragraph) when the workflow has >=2 nodes.
    """
    problems: List[str] = []
    nodes = getattr(workflow_graph, "nodes", []) or []
    if len(nodes) < 2:
        return problems
    try:
        end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        return problems
    evidence_input_keywords = ("goal", "context", "passage", "paragraph", "document", "evidence", "question")
    for node in nodes:
        if getattr(node, "name", "") not in end_nodes:
            continue
        input_names = [str(getattr(inp, "name", "") or "").lower() for inp in getattr(node, "inputs", []) or []]
        if not any(any(kw in n for kw in evidence_input_keywords) for n in input_names):
            problems.append(
                f"terminal node `{node.name}` lacks raw evidence input "
                f"(no goal/context/passage/document field); only sees upstream summaries"
            )
    return problems


def _check_terminal_contract_compliance(workflow_graph: WorkFlowGraph) -> List[str]:
    """Verify terminal nodes match the contract declared in workflow.goal.

    Reads parse_mode and verbatim rules from the goal text, then validates
    each single-output terminal against THAT contract. No hard-coded
    title/str/json bias - whatever the goal text declared is what we
    require.
    """
    problems: List[str] = []
    try:
        end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        return problems
    import re as _re
    contract = _parse_goal_contract(getattr(workflow_graph, "goal", "") or "")
    declared_mode = contract.get("parse_mode") if isinstance(contract, dict) else None
    for node in workflow_graph.nodes:
        if getattr(node, "name", "") not in end_nodes:
            continue
        outs = [str(getattr(out, "name", "") or "") for out in getattr(node, "outputs", []) or []]
        if not outs:
            continue
        if len(outs) > 1:
            # Multi-output terminal: structured (json) path handles it.
            continue
        out_name = outs[0]
        containers = _get_node_prompt_containers(node)
        if not containers:
            continue
        c = containers[0]
        prompt = c.get("prompt", "") or ""
        parse_mode = c.get("parse_mode")
        # mode mismatch only fires when the goal explicitly declares a mode
        if declared_mode in ("title", "str", "json") and parse_mode != declared_mode:
            problems.append(
                f"terminal node `{node.name}` parse_mode is `{parse_mode}`, "
                f"goal contract declares `{declared_mode}`"
            )
        # heading rules depend on chosen mode
        effective_mode = parse_mode if parse_mode in ("title", "str", "json") else (declared_mode or "title")
        if effective_mode == "title":
            if not _re.search(rf"(?m)^##\s+{_re.escape(out_name)}\s*$", prompt, _re.IGNORECASE):
                problems.append(f"terminal node `{node.name}` prompt missing `## {out_name}` heading required by title mode")
        elif effective_mode in ("str", "json"):
            stray = _re.findall(r"(?m)^##\s+[^\n]+\s*$", prompt)
            if stray:
                problems.append(
                    f"terminal node `{node.name}` parse_mode=`{effective_mode}` but prompt still has "
                    f"{len(stray)} `## <name>` heading(s); they will leak into the response"
                )
    return problems


def _auto_repair_initial_workflow_for_contract(workflow_graph: WorkFlowGraph) -> List[str]:
    """In-place best-effort repair of an LLM-generated initial workflow:
      - add `goal` to inputs of any terminal node that lacks evidence
      - set parse_mode=title for single-output terminal nodes
      - append `## <output>` heading to the terminal prompt if missing
    Returns the list of repair actions performed (for logging)."""
    actions: List[str] = []
    nodes = getattr(workflow_graph, "nodes", []) or []
    try:
        end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        end_nodes = set()
    import re as _re

    # Parse the per-benchmark contract from workflow.goal once for this graph.
    # All terminal-node repair below consults `contract` instead of using
    # hard-coded mode/heading/verbatim defaults.
    contract = _parse_goal_contract(getattr(workflow_graph, "goal", "") or "")

    # Build a Param shim by reusing existing inputs' class
    sample_input_class = None
    for n in nodes:
        for inp in getattr(n, "inputs", []) or []:
            sample_input_class = type(inp)
            break
        if sample_input_class:
            break

    evidence_keywords = ("goal", "context", "passage", "paragraph", "document", "evidence", "question")
    for node in nodes:
        nname = getattr(node, "name", "")
        if nname not in end_nodes:
            continue
        # 1) ensure raw evidence available
        if len(nodes) >= 2:
            input_names = [str(getattr(inp, "name", "") or "").lower() for inp in getattr(node, "inputs", []) or []]
            if not any(any(kw in n for kw in evidence_keywords) for n in input_names) and sample_input_class is not None:
                try:
                    new_inp = sample_input_class(
                        name="goal",
                        type="string",
                        description="The original user task / question text. Use this as the source of evidence and constraints.",
                        required=True,
                    )
                    node.inputs = list(node.inputs) + [new_inp]
                    actions.append(f"added `goal` input to terminal `{nname}`")
                    # Local sync: keep agent['inputs'] consistent so downstream
                    # prompt repair / runtime formatting both see `goal`.
                    for _agent_spec in (getattr(node, "agents", []) or []):
                        if not isinstance(_agent_spec, dict):
                            continue
                        _ai = _agent_spec.get("inputs") or []
                        if not isinstance(_ai, list):
                            _ai = []
                        _existing = {str(i.get("name", "")).lower() for i in _ai if isinstance(i, dict)}
                        if "goal" not in _existing:
                            _ai.append({
                                "name": "goal",
                                "type": "string",
                                "description": "The original user task / question text. Use this as the source of evidence and constraints.",
                                "required": True,
                            })
                            _agent_spec["inputs"] = _ai
                            actions.append(f"synced `goal` into agent `{_agent_spec.get('name','?')}` of `{nname}`")
                except Exception as _e:
                    actions.append(f"could not auto-add `goal` to `{nname}`: {_e}")

        outs = [str(getattr(out, "name", "") or "") for out in getattr(node, "outputs", []) or []]
        if not outs or len(outs) > 1:
            continue
        out_name = outs[0]

        containers = _get_node_prompt_containers(node)
        if not containers:
            continue
        c = containers[0]
        prompt = c.get("prompt", "") or ""

        # 2) parse_mode: respect the contract declared in workflow.goal text.
        # Per-benchmark goal text can mandate `str` (e.g. MBPP/HumanEval -
        # raw code, no markdown headings) or `title` (e.g. HotpotQA - extract
        # `## answer` block) or `json`. If no mode is declared in the goal,
        # fall back to `title` ONLY when the LLM did not already pick a mode;
        # otherwise leave the LLM's choice alone.
        declared_mode = contract.get("parse_mode") if isinstance(contract, dict) else None
        existing_mode = c.get("parse_mode")
        if declared_mode in ("title", "str", "json"):
            target_mode = declared_mode
        elif existing_mode in ("title", "str", "json"):
            target_mode = existing_mode  # respect LLM choice when contract is silent
        else:
            target_mode = "title"
        if existing_mode != target_mode:
            c["parse_mode"] = target_mode
            actions.append(f"set parse_mode=`{target_mode}` on terminal `{nname}` (contract-derived)")

        # 3a) heading discipline based on parse_mode.
        # title mode: keep `## <out_name>`, strip any other `## <name>` heading
        #             (rule 3 in HotpotQA-style contracts: extra headings break the title parser).
        # str / json mode: strip ALL `## <name>` headings - the response is
        #                  consumed verbatim (str) or as JSON, and a stray
        #                  heading text leaks into exec()/json.loads().
        section_heading_re = _re.compile(r"(?m)^##\s+([^\n]+?)\s*$")
        headings = section_heading_re.findall(prompt)
        if target_mode == "title":
            # Ensure `## <out_name>` heading present
            has_out_heading = any(h.strip().lower() == out_name.lower() for h in headings)
            if not has_out_heading:
                sep = "\n\n" if prompt and not prompt.endswith("\n") else ""
                prompt = prompt.rstrip() + sep + f"\n\n## {out_name}\n"
                actions.append(f"appended `## {out_name}` heading to `{nname}` prompt")
            # Strip foreign `## <name>` headings (and a single line of
            # placeholder text immediately under each) - they fight the
            # title parser and the model often emits them in the response.
            stripped_count = 0
            def _strip_foreign(match):
                nonlocal stripped_count
                head_name = match.group(1).strip().lower()
                if head_name == out_name.lower():
                    return match.group(0)
                stripped_count += 1
                return ""
            new_prompt = _re.sub(
                r"(?m)^##\s+([^\n]+?)\s*$\n(?:[^\n#][^\n]*\n?)?",
                _strip_foreign,
                prompt,
            )
            if stripped_count:
                prompt = new_prompt
                actions.append(f"stripped {stripped_count} foreign `## <name>` heading(s) from `{nname}` prompt (title mode)")
        elif target_mode in ("str", "json"):
            # Strip every `## <name>` heading - they would otherwise become
            # part of the model response and corrupt str/json consumption.
            new_prompt = _re.sub(
                r"(?m)^##\s+[^\n]+\s*\n(?:[^\n#][^\n]*\n?)?",
                "",
                prompt,
            )
            if new_prompt != prompt:
                prompt = new_prompt
                actions.append(f"stripped `## <name>` heading(s) from `{nname}` prompt ({target_mode} mode)")
        c["prompt"] = prompt

        # 3b) inject verbatim contract sentences (rules / anchor) if missing.
        # Skip rules whose substring is already present (paraphrases count too
        # only when the literal sentence is in the prompt). Each missing rule
        # is appended in a dedicated `# Contract Rules` block at the end.
        verbatim_rules = contract.get("verbatim_rules") if isinstance(contract, dict) else None
        if verbatim_rules:
            missing = [r for r in verbatim_rules if r and r not in prompt]
            if missing:
                # Re-read prompt (it may have been updated above)
                prompt = c.get("prompt", "") or ""
                addendum = "\n\n# Contract Rules (verbatim from goal)\n"
                for r in missing:
                    addendum += f"- {r}\n"
                prompt = prompt.rstrip() + addendum
                c["prompt"] = prompt
                actions.append(f"injected {len(missing)} verbatim contract rule(s) into `{nname}` prompt")

    return actions


def _build_structure_regen_suggestion(reasons: List[str], meta: Dict[str, Any]) -> str:
    """Compose a suggestion string for the WorkFlowGenerator regeneration call."""
    lines = ["Previous attempt failed structural validation. Issues:"]
    for r in reasons:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Please regenerate the workflow with these constraints:")
    lines.append("  1. The TERMINAL node MUST list `goal` in its inputs so it can see the raw question and evidence directly.")
    lines.append("  2. The TERMINAL node MUST use parse_mode=`title` and its prompt MUST contain a `## <terminal_output_name>` heading on its own line (the exact name is whatever the goal contract declares; defaults to `answer`).")
    lines.append("  3. Do NOT add intermediate summarization-only nodes that strip the original context before the terminal node.")
    lines.append("  4. Keep the workflow as compact as possible; prefer 1-2 nodes over a long chain unless reasoning steps are clearly needed.")
    return "\n".join(lines)


def _validate_workflow_structure_for_evolution(
    workflow_graph: WorkFlowGraph,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    node_count = len(getattr(workflow_graph, "nodes", []) or [])
    edge_count = len(getattr(workflow_graph, "edges", []) or [])
    role_meta = _workflow_role_meta(workflow_graph)
    consumed_intermediate_outputs = _count_consumed_intermediate_outputs(workflow_graph)
    goal_input_node_count = _count_goal_input_nodes(workflow_graph)
    invalid_edge_count = _count_invalid_edge_contracts(workflow_graph)
    meta_signal_count, meta_signal_examples = _meta_workflow_signal(workflow_graph)
    interface_ok = _workflow_interface_ok(workflow_graph)

    # GENERALITY_PATCH_V1: only execution-layer hard constraints fail
    # validation. Topology shape (node count, role distribution, degree
    # of goal sharing, data-flow richness) is left for the LLM to decide
    # and for the downstream evaluator to judge empirically.
    reasons: List[str] = []
    if node_count < 1:
        reasons.append("workflow has no nodes")
    if node_count > 12:
        reasons.append(f"workflow has {node_count} nodes; cap at 12 to keep evaluation tractable")
    if node_count >= 2 and edge_count < max(1, node_count - 1):
        reasons.append("workflow is disconnected: need at least node_count-1 edges to form a spanning backbone")
    if invalid_edge_count > 0:
        reasons.append(f"workflow has {invalid_edge_count} edge(s) whose source outputs do not match any target inputs")
    if meta_signal_count >= 2:
        reasons.append(
            "workflow appears to model workflow design/meta-planning instead of directly solving runtime tasks"
        )
    if not interface_ok:
        _expected = sorted(_expected_terminal_output_names(workflow_graph))
        _names = " or ".join(f"`{n}`" for n in _expected) if _expected else "`answer`"
        reasons.append(f"workflow does not preserve the required global interface: initial input `goal`, final output {_names}")

    evidence_problems = _check_evidence_preservation(workflow_graph)
    for ep in evidence_problems:
        reasons.append(ep)
    terminal_problems = _check_terminal_contract_compliance(workflow_graph)
    for tp in terminal_problems:
        reasons.append(tp)

    meta = {
        "node_count": node_count,
        "edge_count": edge_count,
        "role_meta": role_meta,
        "consumed_intermediate_outputs": consumed_intermediate_outputs,
        "goal_input_node_count": goal_input_node_count,
        "invalid_edge_count": invalid_edge_count,
        "meta_signal_count": meta_signal_count,
        "meta_signal_examples": meta_signal_examples,
        "interface_ok": interface_ok,
    }
    return len(reasons) == 0, reasons, meta



def _generate_valid_initial_workflow(
    wf_generator: WorkFlowGenerator,
    base_goal: str,
    max_regenerations: int = 3,
    suggestion: str = "",
) -> WorkFlowGraph:
    """Generate an initial workflow and enforce structural contracts.

    For each attempt:
      1. Ask WorkFlowGenerator to produce a graph from base_goal + suggestion.
      2. Run _auto_repair_initial_workflow_for_contract (in-place, conservative).
      3. Validate via _validate_workflow_structure_for_evolution.
      4. Accept if valid; otherwise add structured suggestion and retry.
    Falls back to the last generated graph (post-repair) so the run still
    proceeds rather than crashing.
    """
    last_reasons: List[str] = []
    last_meta: Dict[str, Any] = {}
    last_graph: Optional[WorkFlowGraph] = None
    cur_suggestion = suggestion or ""

    for attempt in range(max_regenerations + 1):
        workflow_graph = wf_generator.generate_workflow(
            goal=base_goal, retry=1, suggestion=cur_suggestion
        )
        repair_actions = _auto_repair_initial_workflow_for_contract(workflow_graph)
        if repair_actions:
            print(f">>> [Init] auto-repair on attempt {attempt + 1}: {repair_actions}")
        valid, reasons, meta = _validate_workflow_structure_for_evolution(workflow_graph)
        last_graph = workflow_graph
        last_reasons = reasons
        last_meta = meta
        if valid:
            if attempt > 0:
                print(f">>> [Init] accepted regenerated workflow on attempt {attempt + 1}")
            return workflow_graph
        print(f">>> [Init] rejected workflow attempt {attempt + 1}: {reasons}")
        if attempt < max_regenerations:
            cur_suggestion = (suggestion or "") + ("\n\n" if suggestion else "") + _build_structure_regen_suggestion(reasons=reasons, meta=meta)

    print(
        ">>> [Init] WARNING: exhausted regeneration budget; using last (repaired) graph. "
        f"unresolved issues={last_reasons}"
    )
    return last_graph


def _summarize_node_observations(evidences: Dict[str, SampleEvidence]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate node-level observation rates from all samples in one iteration."""
    agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    def _acc(node_name: str, comp: str, dim: str, val: Any):
        node_entry = agg.setdefault(node_name, {"prompt": {}, "params": {}, "return": {}})
        dim_entry = node_entry[comp].setdefault(dim, [0.0, 0.0])
        dim_entry[0] += _safe_rate(val)
        dim_entry[1] += 1.0

    for ev in evidences.values():
        for node_name, dims in (ev.prompt_obs or {}).items():
            for dim, val in (dims or {}).items():
                _acc(node_name, "prompt", dim, val)
        for node_name, dims in (ev.params_obs or {}).items():
            for dim, val in (dims or {}).items():
                _acc(node_name, "params", dim, val)
        for node_name, dims in (ev.return_obs or {}).items():
            for dim, val in (dims or {}).items():
                _acc(node_name, "return", dim, val)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for node_name, comps in agg.items():
        summary[node_name] = {"prompt": {}, "params": {}, "return": {}}
        for comp in ("prompt", "params", "return"):
            for dim, (s, c) in comps[comp].items():
                summary[node_name][comp][dim] = (s / c) if c > 0 else 0.0
    return summary


def _collect_prompt_containers(obj: Any, containers: List[dict]):
    if isinstance(obj, dict):
        if isinstance(obj.get("prompt"), str):
            containers.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _collect_prompt_containers(value, containers)
    elif isinstance(obj, list):
        for item in obj:
            _collect_prompt_containers(item, containers)


def _get_node_prompt_containers(node) -> List[dict]:
    containers: List[dict] = []
    for agent_spec in (node.agents or []):
        _collect_prompt_containers(agent_spec, containers)
    return containers


def _get_node_primary_prompt(node) -> str:
    containers = _get_node_prompt_containers(node)
    if not containers:
        return ""
    return containers[0].get("prompt", "") or ""


def _set_node_prompt(node, new_prompt: str) -> bool:
    changed = False
    for c in _get_node_prompt_containers(node):
        old_prompt = c.get("prompt", "") or ""
        if old_prompt != new_prompt:
            c["prompt"] = new_prompt
            changed = True
    return changed


def _set_node_parse_mode(node, parse_mode: str) -> bool:
    changed = False
    for c in _get_node_prompt_containers(node):
        if c.get("parse_mode") != parse_mode:
            c["parse_mode"] = parse_mode
            changed = True
    return changed


def _get_node_generation_param(node, name: str, default: Any = None) -> Any:
    containers = _get_node_prompt_containers(node)
    if not containers:
        return default
    for c in containers:
        if name in c and c.get(name) is not None:
            return c.get(name)
    return default


def _set_node_generation_param(node, name: str, value: Any) -> bool:
    changed = False
    containers = _get_node_prompt_containers(node)
    for c in containers:
        if c.get(name) != value:
            c[name] = value
            changed = True
    return changed


def _node_needs_structured_parse(node, is_end_node: bool = False) -> bool:
    """Determine whether a node requires JSON structured output parsing.

    Returns True only when JSON is structurally necessary:
      - multiple output fields (need keys to disambiguate), OR
      - dict / object / mapping output types, OR
      - list-of-objects (list[dict], list[object]).

    Plain `list[string]` / `list[str]` is treated as str/title-mode text,
    because mid-tier models routinely emit LaTeX (\boxed{}, \frac{},
    \sqrt{}, |G|), pipe chars, or multi-line prose inside such lists,
    which crashes the JSON repair pipeline. The downstream node receives
    the raw text as a single string, which is semantically equivalent
    to a one-field JSON wrapper for these cases.
    """
    outputs = getattr(node, "outputs", []) or []
    if len(outputs) > 1:
        return True
    for out in outputs:
        t = (out.type or "").lower().strip()
        # dict / object / mapping always needs JSON
        if t.startswith("dict") or t.startswith("object") or t.startswith("mapping"):
            return True
        # list-of-objects (list[dict], list[object]) needs JSON; list[string] does NOT.
        if t.startswith("list"):
            inner = t[4:].lstrip("[ ").rstrip("] ").strip()
            # treat list[str], list[string], list[<empty>] as plain text list
            if inner in ("", "str", "string", "text"):
                continue
            return True
    # Single string (or list-of-strings) output -> str/title mode.
    return False


def _make_prompt_format_safe(prompt_text: str, allowed_input_names: List[str]) -> str:
    """
    Escape all braces except legal placeholders like {goal}.
    This prevents `str.format` from raising unmatched/invalid brace errors.
    """
    text = (prompt_text or "")
    if not text:
        return text

    sentinel_map: Dict[str, str] = {}
    idx = 0

    # Protect existing escaped braces first to avoid repeated escaping.
    text = text.replace("{{", "__EAX_LBRACE_ESC__").replace("}}", "__EAX_RBRACE_ESC__")

    def _protect(match: re.Match) -> str:
        nonlocal idx
        full = match.group(0)
        name = match.group(1)
        if name not in allowed_input_names:
            return full
        token = f"__EAX_PLACEHOLDER_{idx}__"
        idx += 1
        sentinel_map[token] = full
        return token

    protected = _PLACEHOLDER_RE.sub(_protect, text)
    escaped = protected.replace("{", "{{").replace("}", "}}")
    for token, placeholder in sentinel_map.items():
        escaped = escaped.replace(token, placeholder)
    escaped = escaped.replace("__EAX_LBRACE_ESC__", "{{").replace("__EAX_RBRACE_ESC__", "}}")
    return escaped


def _validate_prompt_format(node, prompt_text: str) -> Tuple[bool, str]:
    """Validate prompt can be formatted by CustomizeAction.prepare_action_prompt."""
    try:
        input_names = [inp.name for inp in node.inputs]
        dummy = {name: "x" for name in input_names}
        _ = prompt_text.format(**dummy)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _has_structured_output_contract(prompt_text: str, required_outputs: List[str]) -> bool:
    lowered = (prompt_text or "").lower()
    if "valid json object" not in lowered:
        return False
    for out_name in required_outputs:
        if f'"{out_name.lower()}"' in lowered:
            continue
        if f"`{out_name.lower()}`" in lowered:
            continue
        if re.search(rf"\b{re.escape(out_name.lower())}\b", lowered):
            continue
        return False
    return True


def _has_markdown_output_sections(prompt_text: str) -> bool:
    """Returns True if prompt has AgentGenerator-style Markdown sections (## Thought / ## output_name)."""
    return bool(re.search(r"(?m)^## \S", prompt_text))


def _replace_markdown_output_with_json(prompt_text: str, req_outputs: List[str]) -> str:
    """Replace Markdown ### Output Format block with clean JSON-only format block."""
    json_keys = ", ".join(f'"{o}": "..."' for o in req_outputs)
    json_block = (
        "### Output Format\n"
        "Return ONLY a valid JSON object with no surrounding text:\n"
        "{" + json_keys + "}\n"
        "Do not output markdown, explanation, or prose outside this JSON object."
    )
    if "### Output Format" in prompt_text:
        return re.sub(r"### Output Format.*", json_block, prompt_text, flags=re.S).strip()
    return prompt_text.rstrip() + "\n\n" + json_block


def _compute_observation_coverage(evidences: Dict[str, SampleEvidence], workflow_graph: WorkFlowGraph) -> float:
    """Coverage over node, edge, and structure observation families."""
    if not evidences:
        return 0.0
    num_nodes = max(1, len(workflow_graph.nodes))
    num_edges = len(getattr(workflow_graph, "edges", []) or [])
    total_slots = len(evidences) * (num_nodes * 3 + max(1, num_edges) + 1)
    observed_slots = 0

    for ev in evidences.values():
        observed_slots += len(ev.prompt_obs or {})
        observed_slots += len(ev.params_obs or {})
        observed_slots += len(ev.return_obs or {})
        observed_slots += len(ev.edge_obs or {})
        observed_slots += 1 if (ev.structure_obs or {}) else 0

    return observed_slots / total_slots if total_slots > 0 else 0.0


def _print_iteration_f1_trace(init_f1: float, iteration_trace: List[Dict[str, Any]]):
    best_curve: List[float] = [round(float(init_f1), 4)]
    best_curve.extend(
        round(float(rec["best_after_iter"]), 4)
        for rec in iteration_trace
        if rec.get("best_after_iter") is not None
    )

    print("\n>>> Best F1 Trace:")
    print(f"  - Init: {best_curve[0]:.4f}")
    for rec in iteration_trace:
        if rec.get("best_after_iter") is None:
            continue
        stop_reason = rec.get("stop_reason")
        extra = f", stop={stop_reason}" if stop_reason else ""
        print(
            f"  - Iter {rec.get('iteration', rec.get('iter'))}: "
            f"{float(rec['best_after_iter']):.4f}{extra}"
        )
    print(f">>> Best F1 by iteration: {best_curve}")



def _summarize_evidence_support(evidences: Dict[str, SampleEvidence]) -> Dict[str, float]:
    if not evidences:
        return {
            "samples": 0.0,
            "avg_prompt_nodes": 0.0,
            "avg_params_nodes": 0.0,
            "avg_return_nodes": 0.0,
            "avg_edge_links": 0.0,
            "avg_structure_dims": 0.0,
        }

    sample_count = float(len(evidences))
    prompt_nodes = 0.0
    params_nodes = 0.0
    return_nodes = 0.0
    edge_links = 0.0
    structure_dims = 0.0
    for ev in evidences.values():
        prompt_nodes += float(len(ev.prompt_obs or {}))
        params_nodes += float(len(ev.params_obs or {}))
        return_nodes += float(len(ev.return_obs or {}))
        edge_links += float(len(ev.edge_obs or {}))
        structure_dims += float(len(ev.structure_obs or {}))
    return {
        "samples": sample_count,
        "avg_prompt_nodes": prompt_nodes / sample_count,
        "avg_params_nodes": params_nodes / sample_count,
        "avg_return_nodes": return_nodes / sample_count,
        "avg_edge_links": edge_links / sample_count,
        "avg_structure_dims": structure_dims / sample_count,
    }


def _summarize_root_cause_distribution(root_causes: List[Tuple[str, float]]) -> Dict[str, Any]:
    component_mass: Dict[str, float] = {}
    subtype_mass: Dict[str, float] = {}
    top_by_component: Dict[str, Tuple[str, float]] = {}
    parsed_rows: List[Tuple[str, str, str, float]] = []

    for health_name, failure_prob in root_causes or []:
        parsed = _parse_health_node_name(health_name)
        if not parsed:
            continue
        component, subtype, node_name = parsed
        prob = _safe_rate(failure_prob, 0.0)
        parsed_rows.append((component, subtype, node_name, prob))
        component_mass[component] = component_mass.get(component, 0.0) + prob
        subtype_key = f"{component}.{subtype}"
        subtype_mass[subtype_key] = subtype_mass.get(subtype_key, 0.0) + prob
        best = top_by_component.get(component)
        if best is None or prob > best[1]:
            top_by_component[component] = (health_name, prob)

    component_mass = dict(sorted(component_mass.items(), key=lambda item: item[1], reverse=True))
    subtype_mass = dict(sorted(subtype_mass.items(), key=lambda item: item[1], reverse=True))
    parsed_rows.sort(key=lambda item: item[3], reverse=True)
    return {
        "component_mass": component_mass,
        "subtype_mass": subtype_mass,
        "top_by_component": top_by_component,
        "top_rows": parsed_rows,
    }


def _print_rca_diagnostics(
    *,
    iteration: int,
    root_causes: List[Tuple[str, float]],
    evidences: Dict[str, SampleEvidence],
    target_pool: Optional[List[RcaTarget]] = None,
    pool_mode: Optional[str] = None,
    strength: Optional[Dict[str, float]] = None,
):
    if not root_causes:
        print(f">>> [Iter {iteration}] RCA diagnostics unavailable: no root causes")
        return

    summary = _summarize_root_cause_distribution(root_causes)
    evidence_support = _summarize_evidence_support(evidences)
    top_rows = [
        (comp, subtype, node_name, round(prob, 4))
        for comp, subtype, node_name, prob in summary["top_rows"][:10]
    ]
    top_by_component = {
        comp: (name, round(prob, 4))
        for comp, (name, prob) in summary["top_by_component"].items()
    }
    top_subtypes = list(summary["subtype_mass"].items())[:10]

    print(f">>> [Iter {iteration}] RCA component mass: { {k: round(v, 4) for k, v in summary['component_mass'].items()} }")
    print(f">>> [Iter {iteration}] RCA subtype mass(top): {[(k, round(v, 4)) for k, v in top_subtypes]}")
    print(f">>> [Iter {iteration}] RCA top-by-component: {top_by_component}")
    print(f">>> [Iter {iteration}] RCA top fine-grained targets: {top_rows}")
    if strength:
        print(
            f">>> [Iter {iteration}] RCA strength: "
            f"top1={strength.get('top1', 0.0):.4f}, "
            f"top2={strength.get('top2', 0.0):.4f}, "
            f"margin={strength.get('margin', 0.0):.4f}, "
            f"norm_entropy={strength.get('normalized_entropy', 1.0):.4f}, "
            f"tie_count={int(strength.get('tie_count', 0.0))}, "
            f"saturated={bool(strength.get('saturated', 0.0))}, "
            f"strong={bool(strength.get('strong', 0.0))}"
        )
    print(
        f">>> [Iter {iteration}] Evidence support summary: "
        f"{ {k: round(v, 4) for k, v in evidence_support.items()} }"
    )
    if target_pool is not None:
        print(
            f">>> [Iter {iteration}] RCA actionable targets ({pool_mode or '-'}) : "
            f"{[(x.target_rank, x.component, x.subtype, x.node_name, x.edge_source, x.edge_target, round(x.failure_prob, 4), x.source) for x in target_pool]}"
        )


def _validate_workflow_prompt_templates(workflow_graph: WorkFlowGraph) -> List[str]:
    errors: List[str] = []
    for node in workflow_graph.nodes:
        prompt = _get_node_primary_prompt(node)
        if not prompt:
            continue
        ok, err = _validate_prompt_format(node=node, prompt_text=prompt)
        if not ok:
            errors.append(f"{node.name}: {err}")
    return errors


def _sync_node_inputs_to_agents(workflow_graph: WorkFlowGraph) -> int:
    """Ensure every node.inputs entry is also present in each agent['inputs'] list.

    The runtime builds the agent's inputs_format from agent['inputs'] (a list
    of dicts), NOT from node.inputs. Any input that exists on the node but
    is missing from the agent will be unavailable to the prompt template -
    if the prompt references it via {name}, prompt.format() throws KeyError.

    This sync is benchmark-agnostic: it only adds inputs that already exist on
    the node (so the workflow author / prior repair already vouched for them).
    Returns the count of (agent, input) pairs added."""
    synced = 0
    for node in getattr(workflow_graph, "nodes", []) or []:
        node_inputs = getattr(node, "inputs", []) or []
        if not node_inputs:
            continue
        for agent_spec in (getattr(node, "agents", []) or []):
            if not isinstance(agent_spec, dict):
                continue
            agent_inputs = agent_spec.get("inputs") or []
            if not isinstance(agent_inputs, list):
                agent_inputs = []
            existing = {str(i.get("name", "")).lower() for i in agent_inputs if isinstance(i, dict)}
            for inp in node_inputs:
                name = str(getattr(inp, "name", "") or "")
                if not name or name.lower() in existing:
                    continue
                agent_inputs.append({
                    "name": name,
                    "type": str(getattr(inp, "type", "") or "string"),
                    "description": str(getattr(inp, "description", "") or ""),
                    "required": bool(getattr(inp, "required", True)),
                })
                existing.add(name.lower())
                synced += 1
            agent_spec["inputs"] = agent_inputs
    return synced


def _enforce_workflow_contracts(workflow_graph: WorkFlowGraph) -> Tuple[int, int]:
    """
    Global safety pass:
    - sync node.inputs into each agent['inputs'] (general invariant: prompt placeholders must be backed by agent inputs)
    - repair prompt template/output contract for every node
    - enforce json parse_mode on structured-output nodes (intermediate)
    - enforce str parse_mode on end nodes with single string output
    Returns (prompt_repaired_count, parse_mode_changed_count).
    """
    inputs_synced = _sync_node_inputs_to_agents(workflow_graph)
    if inputs_synced:
        print(f">>> [Contract] synced {inputs_synced} node->agent input(s) before prompt repair")
    prompt_repaired = 0
    parse_mode_changed = 0
    try:
        end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        end_nodes = set()
    for node in workflow_graph.nodes:
        is_end = getattr(node, "name", "") in end_nodes
        prompt = _get_node_primary_prompt(node)
        if prompt:
            repaired = _repair_prompt_contract(
                node=node,
                prompt_text=prompt,
                is_end_node=is_end,
                workflow_goal=getattr(workflow_graph, "goal", "") or "",
            )
            if repaired != prompt and _set_node_prompt(node=node, new_prompt=repaired):
                prompt_repaired += 1
        if _node_needs_structured_parse(node, is_end_node=is_end):
            if _set_node_parse_mode(node=node, parse_mode="json"):
                parse_mode_changed += 1
        else:
            # End nodes with single string output: respect ANY explicit mode
            # already chosen by the initial-workflow contract repair (which
            # in turn read the per-benchmark goal contract). Only fall back
            # to `str` when no mode is set yet. This prevents the contract
            # safety pass from silently flipping a deliberately chosen
            # `title` (HotpotQA) or `str` (MBPP/HumanEval) on every loop.
            existing_modes = {c.get("parse_mode") for c in _get_node_prompt_containers(node)}
            if existing_modes & {"title", "str", "json"}:
                pass  # respect explicit mode
            elif _set_node_parse_mode(node=node, parse_mode="str"):
                parse_mode_changed += 1

        # Mode-aware heading discipline for end nodes.
        # _repair_prompt_contract leaves `## <name>` headings alone (it only
        # handles input/output/grounding wrappers); without this pass, every
        # str/json end node whose prompt still carries `## Thought` / `## answer`
        # markers from AgentGenerator gets bounced by
        # _terminal_node_contract_problems (`prompt still has N ## <name>
        # heading(s); they will leak`) - exactly the rejection loop blocking
        # every structure_edit candidate. str/json: strip every `## <name>`;
        # title: strip foreign headings, ensure `## <out_name>` is present.
        if is_end:
            containers_after = _get_node_prompt_containers(node)
            if containers_after:
                c_after = containers_after[0]
                cur_prompt = c_after.get("prompt", "") or ""
                cur_mode = c_after.get("parse_mode")
                outs_after = [str(getattr(out, "name", "") or "") for out in getattr(node, "outputs", []) or []]
                out_name_after = outs_after[0] if outs_after else ""
                new_prompt = cur_prompt
                if cur_mode in ("str", "json"):
                    new_prompt = re.sub(
                        r"(?m)^##\s+[^\n]+\s*\n(?:[^\n#][^\n]*\n?)?",
                        "",
                        cur_prompt,
                    )
                elif cur_mode == "title" and out_name_after:
                    _stripped_n = [0]
                    def _strip_foreign_after(match):
                        head_name = match.group(1).strip().lower()
                        if head_name == out_name_after.lower():
                            return match.group(0)
                        _stripped_n[0] += 1
                        return ""
                    new_prompt = re.sub(
                        r"(?m)^##\s+([^\n]+?)\s*$\n(?:[^\n#][^\n]*\n?)?",
                        _strip_foreign_after,
                        cur_prompt,
                    )
                    if not re.search(rf"(?m)^##\s+{re.escape(out_name_after)}\s*$", new_prompt, re.IGNORECASE):
                        sep = "" if new_prompt.endswith("\n") else "\n"
                        new_prompt = new_prompt.rstrip() + sep + f"\n\n## {out_name_after}\n"
                if new_prompt != cur_prompt:
                    c_after["prompt"] = new_prompt
                    prompt_repaired += 1
    return prompt_repaired, parse_mode_changed


def _strip_managed_prompt_sections(prompt_text: str) -> str:
    cleaned = prompt_text or ""
    # Strip optimizer-managed sections (# Input Binding / # Output Contract /
    # # Grounding) so the repair layer can re-sync them from the live node
    # I/O. Note: "# Output Format" is intentionally NOT in this list -
    # stripping it would clobber dataset-specific last-line / multi-span /
    # boxed-answer rules baked into the workflow goal or the rewriter output.
    # The re-append logic in _repair_prompt_contract now guards against
    # duplicating an existing # Output Format section (see the existing-section
    # check in both the structured and plain-text branches below).
    for title in ("Input Binding", "Output Contract", "Grounding"):
        pattern = re.compile(
            rf"(?:^|\n)# {re.escape(title)}\n.*?(?=\n# [^\n]+\n|\Z)",
            re.S,
        )
        cleaned = re.sub(pattern, "\n", cleaned)
    # NOTE: ### Output Format stripping is also removed. AgentGenerator
    # emits dataset-aware output format sections under `### Output Format`
    # for benchmarks whose goal text pins specific final-answer formats
    # (DROP multi-span, GSM8K bare number, MATH \\boxed). Stripping them
    # forced the repair layer to re-generate a generic format clause and
    # destroyed the dataset-specific rules. Structured-parse nodes that
    # need a JSON contract still get conversion via
    # _replace_markdown_output_with_json below when a conflicting ## /
    # ### section is detected.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _extract_dataset_format_rules(workflow_goal: str = "", is_end_node: bool = True, dataset_name: str = "") -> str:
    """Recover a dataset-specific final-answer Output Format block.

    The rewrite loop in llm_workflow_optimizer occasionally drops dataset-
    critical format rules during prompt iteration. This helper rebuilds
    them from the workflow goal via keyword detection. Returns a ready-
    to-insert Markdown "# Output Format" section, or "" if no benchmark is
    detected or the node is not an end node (intermediate nodes do not
    carry final-answer format responsibility).

    Detection priority: code > math > drop > gsm8k > hotpotqa.
    """
    if not is_end_node:
        return ""
    _bs = chr(92)
    _boxed = _bs + "boxed"
    _sqrt = _bs + "sqrt"
    _frac = _bs + "frac"
    # Explicit dataset_name takes precedence over keyword matching
    _ds = (dataset_name or "").strip().lower()
    if _ds:
        _is_code = _ds in ("humaneval", "mbpp")
        _is_math = _ds == "math"
        _is_drop = _ds == "drop"
        _is_gsm8k = _ds == "gsm8k"
        _is_hotpotqa = _ds == "hotpotqa"
    else:
        if not workflow_goal:
            return ""
        goal_upper = str(workflow_goal).upper()
        goal_lower = str(workflow_goal).lower()
        _is_code = (
            "HUMANEVAL" in goal_upper or "HUMAN EVAL" in goal_upper
            or "MBPP" in goal_upper or "MOSTLY BASIC PYTHON" in goal_upper
        )
        _is_math = (
            "MATH500" in goal_upper or (_boxed.upper() in goal_upper)
            or "BOXED{" in goal_upper
        )
        _is_drop = "DROP" in goal_upper and (
            "DISCRETE REASONING" in goal_upper or "PASSAGE" in goal_upper
            or "ref_text" in (goal_lower if goal_lower else "")
        )
        _is_gsm8k = "GSM8K" in goal_upper or "GRADE SCHOOL MATH" in goal_upper
        _is_hotpotqa = (
            "HOTPOTQA" in goal_upper or "HOT POT" in goal_upper
            or "MULTI-HOP" in goal_upper.replace("_", "-")
        )
    if _is_code:
        return (
            "# Output Format" + chr(10)
            + "- The final output MUST be raw, executable Python source code only." + chr(10)
            + "- Do NOT wrap the code in markdown fences, JSON, or natural-language explanation." + chr(10)
            + "- The function name MUST exactly match the declared entry_point." + chr(10)
            + "- Include any necessary `import` statements at the top." + chr(10)
            + "- Handle edge cases: empty inputs, single-element lists, type boundaries." + chr(10)
        )
    if _is_math:
        return (
            "# Output Format" + chr(10)
            + "- The VERY LAST LINE of the output MUST be exactly " + _boxed + "{<answer>}." + chr(10)
            + "- Inside the braces, use the simplified canonical form: integers, reduced fractions (a/b), or exact radicals (e.g., " + _sqrt + "{2})." + chr(10)
            + "- Do NOT repeat the problem statement, and do NOT add prose after the " + _boxed + " line." + chr(10)
            + "- Examples: " + _boxed + "{42}, " + _boxed + "{" + _frac + "{3}{7}}, " + _boxed + "{" + _sqrt + "{2}}." + chr(10)
        )
    if _is_drop:
        return (
            "# Output Format" + chr(10)
            + "- The VERY LAST LINE of the output MUST be the bare answer: a number, a short date, or a minimal entity phrase." + chr(10)
            + "- For multi-span answers, join spans with ` and ` (space-and-space), NOT commas." + chr(10)
            + "- Strip articles (a/an/the), punctuation, and surrounding quotes." + chr(10)
            + "- Do NOT repeat the question, do NOT add an 'Answer:' prefix, do NOT wrap in JSON." + chr(10)
        )
    if _is_gsm8k:
        return (
            "# Output Format" + chr(10)
            + "- The VERY LAST LINE of the output MUST be a single bare number (integer or decimal)." + chr(10)
            + "- Remove all units ($, dollars, %, etc.), thousand separators, and currency symbols." + chr(10)
            + "- Do NOT include equations, units, or trailing punctuation on the last line." + chr(10)
            + "- Do NOT wrap the number in JSON, markdown, or explanation." + chr(10)
        )
    if _is_hotpotqa:
        return (
            "# Output Format" + chr(10)
            + "- The final answer MUST be a minimal noun phrase (1-5 words): named entity, date, number, or key term." + chr(10)
            + "- Strip articles (a, an, the), adjectives, verbs, and punctuation." + chr(10)
            + "- Examples: 'Anne Perry', 'novelist', '1985', 'University of Toronto'." + chr(10)
            + "- Never return a full sentence; never wrap in JSON or markdown." + chr(10)
        )
    return ""


def _repair_prompt_contract(node, prompt_text: str, is_end_node: bool = False, workflow_goal: str = "", dataset_name: str = "") -> str:
    """Ensure optimized prompt keeps required placeholders and output contract."""
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        prompt_text = _get_node_primary_prompt(node).strip()

    # Strip AgentGenerator-style <input>{name}</input> wrappers -> clean {name} format.
    prompt_text = re.sub(r"<input>\{(\w+)\}</input>", r"{\1}", prompt_text)
    prompt_text = _strip_managed_prompt_sections(prompt_text)
    req_inputs = [inp.name for inp in node.inputs if getattr(inp, "required", True)]
    req_outputs = [out.name for out in node.outputs if getattr(out, "required", True)]

    missing_inputs = [name for name in req_inputs if f"{{{name}}}" not in prompt_text]
    if missing_inputs:
        prompt_text += "\n\n# Input Binding\n"
        for name in missing_inputs:
            prompt_text += f"- MUST use input variable `{{{name}}}`.\n"

    lowered = prompt_text.lower()
    missing_outputs = [name for name in req_outputs if name.lower() not in lowered]
    if missing_outputs:
        prompt_text += "\n# Output Contract\n"
        prompt_text += "- Return outputs with exact variable names:\n"
        for name in missing_outputs:
            prompt_text += f"  - `{name}`\n"

    grounding_markers = (
        "use only the provided",
        "use only provided inputs",
        "use only provided input",
        "based only on",
        "do not use external knowledge",
    )
    if not any(marker in lowered for marker in grounding_markers):
        prompt_text += "\n# Grounding\n- Use only provided inputs; do not use external knowledge.\n"

    # Check whether ANY level of `Output Format` heading (# / ## / ###) is
    # already present. If so, trust the existing section and skip the
    # generic re-append. This preserves dataset-specific rules authored by
    # AgentGenerator from the workflow goal or surgically modified by the
    # rewriter for benchmarks that pin final-answer format (DROP multi-span,
    # GSM8K bare number, MATH \\boxed, HumanEval/MBPP raw-Python).
    has_existing_output_format = bool(
        re.search(r"(?m)^#{1,4}\s*Output Format\s*$", prompt_text)
    )
    # Title-mode contract: a `## <output_name>` heading on its own line is
    # the structural contract used by parse_mode="title". When present for
    # any required output, treat the prompt as having an explicit output
    # contract and skip the generic Output Format append (which would
    # otherwise add a competing rule and confuse the model).
    if not has_existing_output_format and req_outputs:
        for _out_name in req_outputs:
            if re.search(rf"(?m)^#{{1,4}}\s+{re.escape(_out_name)}\s*$", prompt_text, re.IGNORECASE):
                has_existing_output_format = True
                break
    if _node_needs_structured_parse(node, is_end_node=is_end_node):
        if not _has_structured_output_contract(prompt_text, req_outputs):
            if has_existing_output_format:
                pass  # dataset-specific structured contract - leave alone
            elif _has_markdown_output_sections(prompt_text):
                prompt_text = _replace_markdown_output_with_json(prompt_text, req_outputs)
            else:
                prompt_text += "\n# Output Format\n"
                prompt_text += "- Return ONLY a valid JSON object with keys:\n"
                for out in req_outputs:
                    prompt_text += f'  - "{out}"\n'
                prompt_text += "- Do not output additional prose, markdown, or analysis.\n"
    elif req_outputs and not has_existing_output_format:
        plain_output_name = req_outputs[0]
        # When the node is an end node AND we have workflow_goal context, try to
        # recover a dataset-specific Output Format block that the rewriter may
        # have stripped. This protects dataset-critical rules (\boxed{},
        # bare number, raw Python, multi-span joins) from being erased across
        # rewrite iterations. Falls back to the generic template when no
        # benchmark is detected or workflow_goal is empty.
        _dataset_output_format = ""
        if is_end_node and workflow_goal:
            _dataset_output_format = _extract_dataset_format_rules(workflow_goal, is_end_node=True, dataset_name=dataset_name)
        if _dataset_output_format:
            prompt_text += "\n" + _dataset_output_format.rstrip() + "\n"
        else:
            prompt_text += "\n# Output Format\n"
            prompt_text += f"- Return ONLY the value for `{plain_output_name}` as plain text.\n"
            if plain_output_name.lower() == "answer":
                prompt_text += "- Return only the shortest correct answer phrase for the question.\n"
            prompt_text += "- Do not wrap the output in JSON, markdown, quotes, labels, or explanation.\n"

    input_names = [inp.name for inp in node.inputs]
    prompt_text = _make_prompt_format_safe(prompt_text.strip(), allowed_input_names=input_names)
    return prompt_text


def _build_prompt_optimizer_instruction(
    node,
    old_prompt: str,
    node_stats: Dict[str, Dict[str, float]],
    failure_prob: float,
    iteration: int,
    component: str = "",
    subtype: str = "",
    style: str = "",
    target_rank: int = 0,
    prompt_history: Optional["PromptHistory"] = None,
    action_history: Optional[ActionOutcomeHistory] = None,
    node_fail_streak: int = 0,
    attempt_idx: int = 0,
    previous_failed_ops: Optional[List[str]] = None,
    diagnostic_context: Optional[Dict[str, Any]] = None,
    preferred_op_family: str = "",
) -> str:
    """
    Prompt optimization policy (structured operations):
    - Reflexion-style: diagnose failures from observation evidence
    - TextGrad-style: objective-driven improvement direction
    - OPRO-style: leverage historical prompt performance
    - EvoPrompt-style: structured ADD/DELETE/MODIFY mutations
    """
    input_desc = "\n".join([f"- {p.name} ({p.type}): {p.description}" for p in node.inputs]) or "- None"
    output_desc = "\n".join([f"- {p.name} ({p.type}): {p.description}" for p in node.outputs]) or "- None"
    input_names = [p.name for p in node.inputs]
    placeholder_warning = ", ".join([f"{{{n}}}" for n in input_names])

    obs_summary = {
        "prompt": node_stats.get("prompt", {}),
        "params": node_stats.get("params", {}),
        "return": node_stats.get("return", {}),
        "failure_prob": round(float(failure_prob), 4),
        "rca_component": component or "Prompt",
        "rca_subtype": subtype or "Prompt",
        "selected_style": style or "UNSPECIFIED",
        "target_rank": int(target_rank),
        "iteration": int(iteration),
    }

    history_section = ""
    if prompt_history:
        history_text = prompt_history.format_history_for_llm(node.name)
        if "No previous" not in history_text:
            history_section = f"""
    ## Historical Performance
    {history_text}

    Analyze the trend: which changes improved performance and which degraded it.
    Learn from past iterations to make better decisions.
    """

    rl_history_section = ""
    if action_history is not None:
        node_style_success = action_history.node_style_success_rate("prompt_explore", node.name, style)
        node_style_reward = action_history.node_style_mean_reward("prompt_explore", node.name, style)
        tracked_component = component or "Prompt"
        tracked_subtype = subtype or tracked_component
        subtype_success = action_history.node_subtype_success_rate("prompt_explore", node.name, tracked_component, tracked_subtype)
        subtype_reward = action_history.node_subtype_mean_reward("prompt_explore", node.name, tracked_component, tracked_subtype)
        last_success_style = action_history.last_successful_style("prompt_explore", node.name) or "None"
        failed_styles = action_history.failed_styles("prompt_explore", node.name)
        op_success = action_history.node_op_success_rate("prompt_explore", node.name, preferred_op_family)
        op_reward = action_history.node_op_mean_reward("prompt_explore", node.name, preferred_op_family)
        last_success_op = action_history.last_successful_op("prompt_explore", node.name) or "None"
        failed_op_families = action_history.failed_op_families("prompt_explore", node.name)
        rl_history_section = f"""
        ## RL Action History
        - Selected style: {style or "UNSPECIFIED"}
        - Preferred operation family: {preferred_op_family or "UNSPECIFIED"}
        - RCA component: {component or "Prompt"}
        - RCA subtype: {subtype or "Prompt"}
        - Target rank in RCA pool: {int(target_rank)}
        - Node-subtype success rate: {subtype_success:.4f}
        - Node-subtype mean reward: {subtype_reward:.4f}
        - Node-style success rate: {node_style_success:.4f}
        - Node-style mean reward: {node_style_reward:.4f}
        - Node-op-family success rate: {op_success:.4f}
        - Node-op-family mean reward: {op_reward:.4f}
        - Last successful style on this node: {last_success_style}
        - Recent failed styles on this node: {failed_styles or ["None"]}
        - Last successful op family on this node: {last_success_op}
        - Recent failed op families on this node: {failed_op_families or ["None"]}
        - Node fail streak: {int(node_fail_streak)}
        """
    diagnostic_context = diagnostic_context or {}
    edge_context = ""
    if diagnostic_context.get("edge_source") or diagnostic_context.get("edge_target"):
        edge_context = f"""
        ## Edge Context
        - Upstream edge source: {diagnostic_context.get("edge_source") or "None"}
        - Downstream edge target: {diagnostic_context.get("edge_target") or node.name}
        """

    attempt_section = ""
    if attempt_idx > 0:
        failed_ops_text = ", ".join(previous_failed_ops[-8:]) if previous_failed_ops else "None"
        attempt_section = f"""
        ## Retry Constraint
        - Current retry attempt index: {attempt_idx}
        - Previously failed operation traces: {failed_ops_text}
        - You MUST propose a meaningfully different strategy from previous failed operations.
        """

    return f"""You are an expert prompt engineer optimizing individual node prompts within a multi-step AI workflow.

## Node Information
- Name: {node.name}
- Description: {node.description}
- Inputs:
{input_desc}
- Outputs:
{output_desc}

## Current Prompt
```
{old_prompt}
```

## Diagnostic Evidence
{json.dumps(obs_summary, ensure_ascii=False, indent=2)}

    Interpretation of evidence scores (0-1, higher is better):
    - prompt.input_binding: Are input placeholders correctly used?
    - prompt.output_contract: Are output variable names mentioned?
    - prompt.grounded: Does the prompt avoid hallucination?
    - prompt.executable: Can this node run with its available inputs/evidence, without impossible retrieval assumptions or unresolved placeholders?
- return.type_ok: Does the output match expected type?
- return.content_ok: Is the output non-empty and meaningful?
- return.task_ok: Does the output correctly answer the question?
- params.not_truncated: Was the output cut off by token limit?
- params.format_parseable: Can the output be parsed correctly?
{history_section}
{rl_history_section}
{edge_context}
{attempt_section}
## Task
Analyze the evidence and optimize the prompt using structured operations.
Key failure patterns to watch for based on the diagnostic scores:
- **Low task_ok**: The node output does not satisfy the task; improve reasoning clarity or tighten the objective.
- **Low output_contract**: Output variable names are missing; add them explicitly to the format section.
- **Low input_binding**: Input placeholders are missing or unused; bind them explicitly in the prompt.
- **Low grounded**: Unsupported claims present; add evidence-grounding constraints.
- **Low format_parseable**: Output format hard to parse; simplify the output format instruction.

Return a JSON array of operations. Each operation must be one of:

1. ADD - Insert new content:
{{"op": "ADD", "position": "before|after|beginning|end", "anchor": "<text to position near, required for before/after>", "content": "<new text>", "reason": "<why>"}}

2. DELETE - Remove content:
{{"op": "DELETE", "target": "<exact text to remove>", "reason": "<why>"}}

3. MODIFY - Replace content:
{{"op": "MODIFY", "target": "<exact text to find>", "replacement": "<new text>", "reason": "<why>"}}

4. REWRITE - Full rewrite (use ONLY when major restructuring needed):
{{"op": "REWRITE", "content": "<complete new prompt>", "reason": "<why full rewrite is necessary>"}}

## Critical Rules
- MUST preserve these input placeholders exactly: {placeholder_warning}
- MUST keep output variable names: {", ".join([p.name for p in node.outputs])}
- Do NOT delete or rewrite the core `### Output Format` contract unless it is clearly missing or inconsistent with the node outputs.
- Prefer editing `### Objective` and `### Instructions` over changing output schema wording.
- Prefer targeted ADD/DELETE/MODIFY over REWRITE
- Use the selected style as the primary editing policy:
  - BINDING_REPAIR: strengthen variable usage and upstream/downstream linking.
  - SCHEMA_HARDEN: make output contract and field naming stricter.
  - GROUNDING_HARDEN: reduce unsupported claims and require evidence-grounded reasoning.
  - CHAIN_SYNTHESIS: clarify multi-hop combination and bridge reasoning.
  - DEDUP_SIMPLIFY: remove repeated or noisy instructions without losing constraints.
  - ANSWER_NORMALIZE: force concise, canonical answer formatting.
- Focus on improving the RCA-indicated failure subtype while preserving valid formatting.
- Use `{preferred_op_family or "MODIFY"}` as the primary operation family for this edit.
- At least one returned operation SHOULD be of family `{preferred_op_family or "MODIFY"}` unless REWRITE is strictly necessary.
- Return ONLY the JSON array. No explanation, no markdown fences.
""".strip()


def _heuristic_prompt_fallback(node, old_prompt: str, iteration: int, style: str = "", subtype: str = "") -> str:
    old_prompt = (old_prompt or "").strip()
    marker = f"# Optimization Patch Iteration {iteration}"
    if marker in old_prompt:
        return old_prompt

    patch = [
        marker,
        "- Decompose this subtask into minimal reasoning steps before output.",
        "- Use only provided inputs and evidence from context.",
        "- Validate output variable names and return only required outputs.",
    ]
    style = (style or "").upper().strip()
    if style == "BINDING_REPAIR":
        patch.append("- Explicitly bind every required input variable and reference upstream outputs by exact variable name.")
    elif style == "SCHEMA_HARDEN":
        patch.append("- Enforce the exact output schema and required field names with no extra prose.")
    elif style == "GROUNDING_HARDEN":
        patch.append("- Refuse unsupported claims; keep all reasoning grounded in provided context only.")
    elif style == "CHAIN_SYNTHESIS":
        patch.append("- Explicitly connect intermediate facts into one coherent reasoning chain before answering.")
    elif style == "DEDUP_SIMPLIFY":
        patch.append("- Remove redundant instructions and keep only the minimum constraints needed for correctness.")
    elif style == "ANSWER_NORMALIZE":
        patch.append("- Normalize the final answer to one concise canonical entity string.")
    if subtype:
        patch.append(f"- Prioritize repairing the RCA-identified failure subtype: {subtype}.")
    return (old_prompt + "\n\n" + "\n".join(patch)).strip() if old_prompt else "\n".join(patch)


def _parse_llm_operations(raw_output: str) -> List[Dict[str, Any]]:
    """Parse LLM output into a list of operation dicts."""
    text = raw_output.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Conservative fallback: do not rewrite full prompt on parse failure.
    # Let caller use heuristic patch instead.
    if len(text) > 20:
        return []

    return []


def _apply_prompt_operations(
    prompt_text: str, operations: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Apply structured operations to a prompt.
    Returns (new_prompt, list_of_applied_operation_descriptions).
    """
    applied: List[str] = []
    result = prompt_text

    for op in operations:
        op_type = (op.get("op") or "").upper().strip()
        reason = op.get("reason", "no reason given")

        if op_type == "REWRITE":
            content = (op.get("content") or "").strip()
            if content:
                result = content
                applied.append(f"REWRITE: {reason}")
                break  # REWRITE replaces everything, ignore subsequent ops

        elif op_type == "ADD":
            content = (op.get("content") or "").strip()
            position = (op.get("position") or "end").lower().strip()
            anchor = op.get("anchor") or ""

            if not content:
                continue

            if position == "beginning":
                result = content + "\n" + result
            elif position == "end":
                result = result + "\n" + content
            elif position == "before" and anchor and anchor in result:
                result = result.replace(anchor, content + "\n" + anchor, 1)
            elif position == "after" and anchor and anchor in result:
                result = result.replace(anchor, anchor + "\n" + content, 1)
            else:
                # Fallback: append to end
                result = result + "\n" + content
            applied.append(f"ADD({position}): {reason}")

        elif op_type == "DELETE":
            target = op.get("target") or ""
            if target and target in result:
                result = result.replace(target, "", 1)
                applied.append(f"DELETE: {reason}")

        elif op_type == "MODIFY":
            target = op.get("target") or ""
            replacement = op.get("replacement") or ""
            if target and target in result:
                result = result.replace(target, replacement, 1)
                applied.append(f"MODIFY: {reason}")

    return result.strip(), applied


def _optimize_prompt_for_node(
    llm,
    workflow_graph: WorkFlowGraph,
    node_name: str,
    node_stats: Dict[str, Dict[str, Dict[str, float]]],
    failure_prob: float,
    iteration: int,
    component: str = "",
    subtype: str = "",
    style: str = "",
    target_rank: int = 0,
    prompt_history: Optional["PromptHistory"] = None,
    action_history: Optional[ActionOutcomeHistory] = None,
    modification_history=None,
    node_fail_streak: int = 0,
    attempt_idx: int = 0,
    previous_failed_ops: Optional[List[str]] = None,
    diagnostic_context: Optional[Dict[str, Any]] = None,
    preferred_op_family: str = "",
    failure_examples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Optimize a node's prompt using structured ADD/DELETE/MODIFY operations.
    Returns (changed, message, operations_applied).
    """
    node = workflow_graph.get_node(node_name)
    old_prompt = _get_node_primary_prompt(node)
    if not old_prompt:
        return False, f"{node_name}: no prompt found, skipped", []

    # --- Agentic optimizer (primary path) ---
    operations = []
    try:
        from evoagentx.optimizers.rl.optimizer_agent import OptimizationContext, run_optimizer_agent
        _ctx = OptimizationContext(
            workflow_graph=workflow_graph,
            node_stats=node_stats,
            prompt_history=prompt_history,
            action_history=action_history,
            modification_history=modification_history,
            get_node_prompt=lambda n: _get_node_primary_prompt(workflow_graph.get_node(n)),
            failure_examples=failure_examples,
        )
        agent_ops = run_optimizer_agent(
            llm=llm,
            context=_ctx,
            node_name=node_name,
            failure_prob=failure_prob,
            iteration=iteration,
            component=component,
            subtype=subtype,
            style=style,
            preferred_op_family=preferred_op_family,
            attempt_idx=attempt_idx,
            previous_failed_ops=previous_failed_ops,
            max_turns=5,
        )
        if agent_ops is not None:
            operations = agent_ops
    except Exception as _agent_err:
        print(f">>> [PromptOpt] node={node_name} agent failed ({_agent_err}), falling back to legacy")

    # --- Legacy fallback: monolithic prompt ---
    if not operations:
        optimize_prompt = _build_prompt_optimizer_instruction(
            node=node,
            old_prompt=old_prompt,
            node_stats=node_stats.get(node_name, {}),
            failure_prob=failure_prob,
            iteration=iteration,
            component=component,
            subtype=subtype,
            style=style,
            target_rank=target_rank,
            prompt_history=prompt_history,
            action_history=action_history,
            node_fail_streak=node_fail_streak,
            attempt_idx=attempt_idx,
            previous_failed_ops=previous_failed_ops,
            diagnostic_context=diagnostic_context,
            preferred_op_family=preferred_op_family,
        )
        raw_response = ""
        try:
            llm_out = llm.generate(prompt=optimize_prompt)
            raw_response = str(getattr(llm_out, "content", llm_out)).strip()
        except Exception as e:
            print(f">>> [PromptOpt] node={node_name} LLM call failed: {e}")
        operations = _parse_llm_operations(raw_response) if raw_response else []
    preferred_family = _normalize_op_family(preferred_op_family)
    fallback_prefix: List[str] = []
    if operations and preferred_family:
        family_ops = _filter_operations_by_family(operations, preferred_family)
        if family_ops:
            operations = family_ops
        else:
            fallback_prefix.append(f"SOFT_OP_FAMILY_FALLBACK({preferred_family})")
            print(
                f">>> [PromptOpt] node={node_name} no {preferred_family} operations from LLM; "
                "falling back to the best available operation family"
            )

    if not operations:
        print(f">>> [PromptOpt] node={node_name} no valid operations from LLM, using heuristic fallback")
        candidate = _heuristic_prompt_fallback(
            node=node,
            old_prompt=old_prompt,
            iteration=iteration,
            style=style,
            subtype=subtype,
        )
        ops_applied = fallback_prefix + ["HEURISTIC_FALLBACK"]
    else:
        # Apply structured operations
        candidate, ops_applied = _apply_prompt_operations(old_prompt, operations)
        if not ops_applied:
            candidate = _heuristic_prompt_fallback(
                node=node,
                old_prompt=old_prompt,
                iteration=iteration,
                style=style,
                subtype=subtype,
            )
            ops_applied = fallback_prefix + ["LLM_OPS_NO_MATCH_FALLBACK"]
            print(f">>> [PromptOpt] node={node_name} LLM operations unmatched, fallback applied")
        else:
            ops_applied = fallback_prefix + ops_applied
            print(f">>> [PromptOpt] node={node_name} applied {len(ops_applied)} operations: {ops_applied}")

    # Post-processing: repair contract and validate
    try:
        _end_nodes = set(workflow_graph.find_end_nodes() or [])
    except Exception:
        _end_nodes = set()
    _is_end = node_name in _end_nodes
    candidate = _repair_prompt_contract(
        node=node,
        prompt_text=candidate,
        is_end_node=_is_end,
        workflow_goal=getattr(workflow_graph, "goal", "") or "",
    )
    is_valid, err_msg = _validate_prompt_format(node=node, prompt_text=candidate)

    if not is_valid:
        # Try to recover with format safety
        input_names = [inp.name for inp in node.inputs]
        candidate = _make_prompt_format_safe(candidate, allowed_input_names=input_names)
        is_valid, err_msg = _validate_prompt_format(node=node, prompt_text=candidate)

    if not is_valid:
        # Rollback: check if we have a best historical prompt
        if prompt_history:
            best_record = prompt_history.get_best_record(node_name)
            if best_record and best_record.prompt_text != old_prompt:
                candidate = best_record.prompt_text
                is_valid, _ = _validate_prompt_format(node=node, prompt_text=candidate)
                if is_valid:
                    ops_applied = [f"ROLLBACK_TO_BEST(iter={best_record.iteration})"]
                    print(f">>> [PromptOpt] node={node_name} rolled back to best historical prompt (iter {best_record.iteration})")
        if not is_valid:
            return False, f"{node_name}: invalid prompt format ({err_msg}), rollback", []

    if candidate == old_prompt:
        return False, f"{node_name}: prompt unchanged after operations", ops_applied

    changed = _set_node_prompt(node=node, new_prompt=candidate)
    return changed, (f"{node_name}: prompt updated ({len(ops_applied)} ops)" if changed else f"{node_name}: prompt not updated"), ops_applied


def _optimize_prompt_with_retries(
    llm,
    workflow_graph: WorkFlowGraph,
    node_name: str,
    node_stats: Dict[str, Dict[str, Dict[str, float]]],
    failure_prob: float,
    iteration: int,
    component: str = "",
    subtype: str = "",
    style: str = "",
    target_rank: int = 0,
    prompt_history: Optional["PromptHistory"] = None,
    action_history: Optional[ActionOutcomeHistory] = None,
    modification_history=None,
    node_fail_streak: int = 0,
    max_attempts: int = 3,
    force: bool = False,
    diagnostic_context: Optional[Dict[str, Any]] = None,
    preferred_op_family: str = "",
    failure_examples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Retry prompt optimization with strategy diversification.
    """
    node_level_stats = node_stats.get(node_name, {})
    prompt_stats = node_level_stats.get("prompt", {})
    if not force and failure_prob < 0.02 and _stats_are_healthy(prompt_stats, threshold=0.95):
        return False, f"{node_name}: prompt signal too weak", []

    failed_ops: List[str] = []
    last_msg = f"{node_name}: no prompt attempt executed"

    for attempt_idx in range(max_attempts):
        ok, msg, ops_applied = _optimize_prompt_for_node(
            llm=llm,
            workflow_graph=workflow_graph,
            node_name=node_name,
            node_stats=node_stats,
            failure_prob=failure_prob,
            iteration=iteration * 10 + attempt_idx,
            component=component,
            subtype=subtype,
            style=style,
            target_rank=target_rank,
            prompt_history=prompt_history,
            action_history=action_history,
            modification_history=modification_history,
            node_fail_streak=node_fail_streak,
            attempt_idx=attempt_idx,
            previous_failed_ops=failed_ops,
            diagnostic_context=diagnostic_context,
            preferred_op_family=preferred_op_family,
            failure_examples=failure_examples,
        )
        if ok:
            return True, f"{msg}; attempts={attempt_idx + 1}", ops_applied

        last_msg = msg
        if ops_applied:
            failed_ops.extend(ops_applied)

    return False, f"{node_name}: failed after {max_attempts} attempts ({last_msg})", []




def _summarize_actionable_rca_strength(
    actionable: List[RcaTarget],
    strong_rca_threshold: float,
) -> Dict[str, float]:
    probs = [max(0.0, _safe_rate(item.failure_prob, 0.0)) for item in actionable[: max(1, min(5, len(actionable)))]]
    if not probs:
        return {
            "top1": 0.0,
            "top2": 0.0,
            "margin": 0.0,
            "ratio": 0.0,
            "normalized_entropy": 1.0,
            "tie_count": 0.0,
            "saturated": 0.0,
            "strong": 0.0,
            "threshold": float(strong_rca_threshold),
        }

    top1 = probs[0]
    top2 = probs[1] if len(probs) > 1 else 0.0
    margin = top1 - top2
    ratio = (top1 / max(top2, 1e-6)) if top2 > 0 else (999.0 if top1 > 0 else 0.0)
    total = sum(probs)
    if total > 0:
        normalized = [p / total for p in probs]
    else:
        normalized = [1.0 / len(probs)] * len(probs)
    entropy = -sum(p * math.log(max(p, 1e-12), 2) for p in normalized)
    max_entropy = math.log(len(normalized), 2) if len(normalized) > 1 else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    tie_tol = max(1e-6, 0.03 * max(top1, 1.0))
    tie_count = sum(1 for p in probs if abs(p - top1) <= tie_tol)
    saturated = 1.0 if (top1 >= 0.95 and tie_count >= min(3, len(probs))) else 0.0
    strong = 1.0 if (
        top1 >= max(strong_rca_threshold, 0.03)
        and margin >= max(0.02, 0.10 * top1)
        and (top2 <= 1e-9 or ratio >= 1.8)
        and normalized_entropy <= 0.92
        and tie_count <= 1
        and not saturated
    ) else 0.0
    return {
        "top1": float(top1),
        "top2": float(top2),
        "margin": float(margin),
        "ratio": float(ratio),
        "normalized_entropy": float(normalized_entropy),
        "tie_count": float(tie_count),
        "saturated": float(saturated),
        "strong": float(strong),
        "threshold": float(strong_rca_threshold),
    }


def _build_stats_fallback_plan(
    node_stats: Dict[str, Dict[str, Dict[str, float]]],
    max_targets: int = 4,
    min_risk: float = 0.05,
) -> List[Dict[str, Any]]:
    """Fallback when RCA ranking gives no actionable Prompt/Params nodes.

    Risk is estimated from observation quality: lower score => higher risk.
    """
    candidates: List[Dict[str, Any]] = []
    for node_name, comps in (node_stats or {}).items():
        for comp_key, comp_name in (("prompt", "Prompt"), ("params", "Params")):
            dims = list((comps.get(comp_key) or {}).values())
            if dims:
                quality = sum(_safe_rate(v, 0.0) for v in dims) / max(1, len(dims))
                risk = 1.0 - quality
            else:
                continue
            if risk >= min_risk:
                candidates.append(
                    {
                        "health_name": f"Fallback{comp_name}_{node_name}",
                        "component": comp_name,
                        "node_name": node_name,
                        "failure_prob": float(risk),
                    }
                )
    candidates.sort(key=lambda x: x["failure_prob"], reverse=True)
    return candidates[:max_targets]


def _build_rca_target_pool(
    *,
    root_causes: List[Tuple[str, float]],
    node_stats: Dict[str, Dict[str, Dict[str, float]]],
    node_fail_streak: Dict[Tuple[str, str], int],
    action_history: Optional[ActionOutcomeHistory],
    max_node_fail_streak: int,
    max_targets: int,
    strong_rca_threshold: float,
    diversify: bool = False,
) -> Tuple[List[RcaTarget], str, float, Dict[str, float]]:
    """
    RCA owns target localization.

    The returned target pool is the only place from which optimization candidates
    may be instantiated. RL is then restricted to selecting among candidates
    derived from this pool, instead of freely choosing arbitrary nodes.
    """
    actionable: List[RcaTarget] = []
    seen = set()
    seen_component_nodes = set()
    for health_name, failure_prob in root_causes or []:
        parsed = _parse_health_node_name(health_name)
        if not parsed:
            continue
        component, subtype, node_name = parsed
        key = (component, node_name, subtype)
        if key in seen:
            continue
        seen.add(key)
        if component in ("Prompt", "Params"):
            component_key = (component, node_name)
            if node_fail_streak.get(component_key, 0) >= max_node_fail_streak:
                continue
        elif component == "Return":
            prompt_streak = node_fail_streak.get(("Prompt", node_name), 0)
            params_streak = node_fail_streak.get(("Params", node_name), 0)
            prompt_attempts = (
                action_history.node_subtype_attempts("prompt_explore", node_name, component, subtype)
                if action_history is not None
                else 0.0
            )
            prompt_success = (
                action_history.node_subtype_success_rate("prompt_explore", node_name, component, subtype)
                if action_history is not None
                else 0.0
            )
            if (
                prompt_streak >= max_node_fail_streak
                and params_streak >= max_node_fail_streak
                and prompt_attempts >= 2.0
                and prompt_success <= 0.05
            ):
                continue
        elif component == "Edge":
            if node_fail_streak.get(("Prompt", node_name), 0) >= max_node_fail_streak:
                continue
        seen_component_nodes.add((component, node_name))
        edge_source, edge_target = _parse_edge_suffix(health_name) if component == "Edge" else (None, None)
        actionable.append(
            RcaTarget(
                health_name=health_name,
                component=component,
                subtype=subtype,
                node_name=node_name,
                failure_prob=float(failure_prob),
                source="rca",
                edge_source=edge_source,
                edge_target=edge_target,
            )
        )

    actionable.sort(key=lambda x: x.failure_prob, reverse=True)
    specific_subtypes = {
        (item.component, item.node_name)
        for item in actionable
        if item.subtype not in (item.component, "Prompt", "Params", "Return", "Edge", "Structure")
    }
    actionable = [
        item
        for item in actionable
        if (item.component, item.node_name) not in specific_subtypes
        or item.subtype not in (item.component, "Prompt", "Params", "Return", "Edge", "Structure")
    ]
    strength = _summarize_actionable_rca_strength(actionable, strong_rca_threshold)
    top_prob = strength["top1"]
    pool_mode = "strong_rca" if strength["strong"] >= 1.0 else "weak_rca"
    target_limit = (max_targets + 2) if (diversify and pool_mode == "weak_rca") else max_targets
    pool_cap = min(max_targets, 2) if pool_mode == "strong_rca" else target_limit
    pool_cap = min(len(actionable), pool_cap) if actionable else pool_cap
    target_pool: List[RcaTarget] = list(actionable[:pool_cap])

    structure_or_edge_candidate = next(
        (
            item
            for item in actionable
            if item.component in ("Structure", "Edge")
            and item.failure_prob >= (
                max(0.10, 0.40 * max(strength.get("top2", 0.0), 0.10))
                if item.component == "Structure"
                else max(0.02, 0.15 * max(strength.get("top2", 0.0), 0.05))
            )
        ),
        None,
    )
    if structure_or_edge_candidate is not None and not any(
        item.component in ("Structure", "Edge") for item in target_pool
    ):
        if len(target_pool) < pool_cap:
            target_pool.append(structure_or_edge_candidate)
        else:
            replace_idx = None
            for idx in range(len(target_pool) - 1, -1, -1):
                if target_pool[idx].component == "Return":
                    replace_idx = idx
                    break
            if replace_idx is None:
                replace_idx = len(target_pool) - 1
            target_pool[replace_idx] = structure_or_edge_candidate

    # Ensure the pool always contains at least one Prompt-type target so the
    # planner can fall back to prompt_edit when structure edits fail.  This is
    # critical for strong_rca mode where pool_cap=2 and both top targets may be
    # Structure, leaving zero Prompt targets and causing every prompt_edit
    # candidate to be rejected with invalid_rca_rank.
    has_prompt_target = any(item.component in ("Prompt", "Params") for item in target_pool)
    if not has_prompt_target:
        prompt_fallback = next(
            (item for item in actionable if item.component in ("Prompt", "Params")),
            None,
        )
        if prompt_fallback is not None:
            target_pool.append(prompt_fallback)

    if pool_mode == "weak_rca" and len(target_pool) < target_limit:
        fallback = _build_stats_fallback_plan(node_stats=node_stats, max_targets=max_targets, min_risk=0.05)
        for item in fallback:
            key = (item["component"], item["node_name"])
            if key in seen_component_nodes:
                continue
            if node_fail_streak.get(key, 0) >= max_node_fail_streak:
                continue
            seen_component_nodes.add(key)
            subtype = "Prompt" if item["component"] == "Prompt" else "Params"
            target_pool.append(
                RcaTarget(
                    health_name=item["health_name"],
                    component=item["component"],
                    subtype=subtype,
                    node_name=item["node_name"],
                    failure_prob=float(item["failure_prob"]),
                    source="stats_fallback",
                )
            )
            if len(target_pool) >= target_limit:
                break

    for idx, item in enumerate(target_pool, start=1):
        item.target_rank = idx
        item.target_pool_size = len(target_pool)
        item.pool_mode = pool_mode

    return target_pool, pool_mode, top_prob, strength





def _io_spec(name: str, type_name: str, description: str, required: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "type": type_name,
        "description": description,
        "required": required,
    }


def _node_spec(
    name: str,
    description: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputs": copy.deepcopy(inputs),
        "outputs": copy.deepcopy(outputs),
        "agents": [],
    }


def _best_workflow_summary(best_workflow: WorkFlowGraph, best_results: Dict[str, float]) -> str:
    meta = _workflow_role_meta(best_workflow)
    complexity = workflow_complexity_metrics(best_workflow)
    summary = {
        "best_metrics": {
            "f1": round(_safe_rate(best_results.get("f1", 0.0)), 4),
            "em": round(_safe_rate(best_results.get("em", 0.0)), 4),
            "acc": round(_safe_rate(best_results.get("acc", 0.0)), 4),
        },
        "utility": round(compute_workflow_utility(workflow_graph=best_workflow, results=best_results, config=RewardConfig()), 4),
        "node_count": int(complexity["node_count"]),
        "edge_count": int(complexity["edge_count"]),
        "dag_depth": int(complexity["dag_depth"]),
        "covered_roles": meta["covered_roles"],
        "workflow_description": best_workflow.get_workflow_description(),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _get_prompt_failure_prob(root_causes: List[Tuple[str, float]], node_name: str) -> float:
    for rc_name, rc_prob in root_causes:
        parsed = _parse_health_node_name(rc_name)
        if parsed and parsed[0] == "Prompt" and parsed[2] == node_name:
            return float(rc_prob)
    return 0.0


def _record_prompt_history_snapshot(
    workflow_graph: WorkFlowGraph,
    root_causes: List[Tuple[str, float]],
    metrics: Dict[str, float],
    prompt_history: PromptHistory,
    iteration: int,
    ops_by_node: Optional[Dict[str, List[str]]] = None,
    allow_new_versions: bool = True,
):
    ops_by_node = ops_by_node or {}
    f1 = _safe_rate(metrics.get("f1", 0.0))
    em = _safe_rate(metrics.get("em", 0.0))

    for node in workflow_graph.nodes:
        current_prompt = _get_node_primary_prompt(node)
        if not current_prompt:
            continue
        node_fp = _get_prompt_failure_prob(root_causes=root_causes, node_name=node.name)
        ops = ops_by_node.get(node.name, [])
        history = prompt_history.get_history(node.name)

        if not history:
            prompt_history.add_record(
                node.name,
                PromptRecord(
                    iteration=iteration,
                    prompt_text=current_prompt,
                    metrics={"f1": f1, "em": em},
                    failure_prob=node_fp,
                    operations_applied=ops,
                ),
            )
            continue

        same_prompt = history[-1].prompt_text == current_prompt
        is_explicit_mutation = bool(ops)
        if allow_new_versions and (is_explicit_mutation or not same_prompt):
            prompt_history.add_record(
                node.name,
                PromptRecord(
                    iteration=iteration,
                    prompt_text=current_prompt,
                    metrics={"f1": f1, "em": em},
                    failure_prob=node_fp,
                    operations_applied=ops,
                ),
            )
        else:
            history[-1].metrics = {"f1": f1, "em": em}
            history[-1].failure_prob = node_fp


def _build_fixed_eval_indices(
    benchmark: HotPotQA,
    eval_mode: str,
    sample_k: int,
    seed: int,
) -> List[int]:
    """Build a deterministic fixed subset once, then reuse it for all iterations."""
    assert eval_mode in ("train", "dev", "test"), f"Invalid eval_mode: {eval_mode}"
    if eval_mode == "train":
        data = benchmark.get_train_data()
    elif eval_mode == "dev":
        data = benchmark.get_dev_data()
    else:
        data = benchmark.get_test_data()

    total = len(data)
    if total <= 0:
        return []

    all_indices = list(range(total))
    if sample_k is None or sample_k <= 0 or sample_k >= total:
        return all_indices

    rng = random.Random(seed)
    return rng.sample(all_indices, k=sample_k)


def _run_single_iteration(
    iteration: int,
    llm,
    workflow_graph: WorkFlowGraph,
    agent_manager: AgentManager,
    benchmark: HotPotQA,
    eval_indices: Optional[List[int]] = None,
    sample_k: Optional[int] = None,
    seed: Optional[int] = None,
    eval_mode: str = "dev",
    num_workers: int = 50,
    calibration_profile: Optional[FactorCalibrationProfile] = None,
    run_rca: bool = True,
) -> Tuple[
    Dict[str, float],
    List[Tuple[str, float]],
    Dict[str, SampleEvidence],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    evidence_buf = EvidenceBuffer()
    failure_examples: List[Dict[str, Any]] = []
    use_aflow_hotpotqa = isinstance(benchmark, AFlowHotPotQA)
    use_math = isinstance(benchmark, (MATH, AFlowMATH))
    use_humaneval = isinstance(benchmark, (HumanEval, AFlowHumanEval))
    use_mbpp = isinstance(benchmark, (MBPP, AFlowMBPP))
    use_gsm8k = isinstance(benchmark, (GSM8K, AFlowGSM8K))
    use_drop = isinstance(benchmark, AFlowDROP)

    def _stringify_context_value(value: Any, max_chars: int = 400) -> str:
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

    def _extract_node_outputs_from_trajectory(trajectory: Any) -> Dict[str, str]:
        node_outputs: Dict[str, str] = {}
        for message in trajectory or []:
            if isinstance(message, dict):
                wf_task = message.get("wf_task")
                content = message.get("content")
            else:
                wf_task = getattr(message, "wf_task", None)
                content = getattr(message, "content", None)
            if not wf_task:
                continue
            text = _stringify_context_value(content, max_chars=300)
            if text:
                node_outputs[str(wf_task)] = text
        return node_outputs

    def _record_failure_example(example: Any, label: Any, prediction: Any, metrics: Dict[str, Any], trajectory: Any):
        try:
            _mdict = metrics or {}
            em = _safe_rate(_mdict.get("em", _mdict.get("pass@1", 0.0)))
            f1 = _safe_rate(_mdict.get("f1", _mdict.get("pass@1", _mdict.get("solve_rate", 0.0))))
            if f1 >= 0.999:
                return
            example_dict = example if isinstance(example, dict) else {}
            failure_examples.append(
                {
                    "question": _stringify_context_value(example_dict.get("question") or example_dict.get("problem") or example_dict.get("context") or example_dict.get("prompt") or example_dict.get("text") or "", max_chars=500),
                    "gold_answer": _stringify_context_value(label if label is not None else example_dict.get("answer", ""), max_chars=250),
                    "predicted_answer": _stringify_context_value(prediction, max_chars=250),
                    "node_outputs": _extract_node_outputs_from_trajectory(trajectory),
                    "f1": float(f1),
                    "em": float(em),
                }
            )
        except Exception as e:
            print(
                f">>> [Iter {iteration}] warning: failure-example extraction failed: {e}"
            )

    def _postprocess_math_answer(output: Any) -> str:
        """Post-process for MATH: preserve \boxed{} and LaTeX."""
        import regex as _regex
        if isinstance(output, dict):
            output = output.get("answer", output.get("result", output))
        if output is None:
            return ""
        content = str(output).strip()
        _xml_m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)
        if _xml_m:
            content = _xml_m.group(1).strip()
        content = content.replace("```json", "").replace("```", "")
        boxed_pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = _regex.findall(boxed_pattern, content, _regex.DOTALL)
        if boxed_matches:
            return "\\boxed{" + boxed_matches[-1].strip() + "}"
        _ans_m = re.search(r"(?:(?:final )?answer\s*(?:is|:|=)\s*)(.+?)\s*[.]?\s*$", content, re.IGNORECASE | re.DOTALL)
        if _ans_m:
            return _ans_m.group(1).strip().rstrip(".")
        _lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        if _lines:
            return _lines[-1]
        return content.strip()

    def collate_func(example: dict) -> dict:
        if use_math:
            prompt = example["problem"]
            return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt}
        if use_humaneval:
            prompt = example["prompt"]
            entry_point = example.get("entry_point", "")
            if entry_point:
                prompt = prompt.rstrip() + "\n\nIMPORTANT: The generated function MUST be named `" + entry_point + "`. Return ONLY the function definition (and any needed helpers/imports). Do NOT include asserts, test cases, check() functions, or explanations."
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
            context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
            prompt = f"Context: {context_str}\n\nQuestion: {example['question']}\n\nAnswer:"
            return {"goal": prompt, "problem": prompt, "question": prompt, "input": prompt, "query": prompt, "task": prompt}

        prompt = "### User Question:\n{}\n\n".format(example["question"])
        context_texts = []
        for item in example["context"]:
            context_texts.append(f"Document [{item[0]}]: {''.join(item[1])}")
        prompt += "### Reference Documents:\n" + "\n".join(context_texts)
        prompt += (
            "\n\n### Instruction:\nBased ONLY on the Reference Documents above, "
            "extract the precise entity name that answers the User Question."
        )
        return {"goal": prompt, "problem": prompt}

    def output_postprocess_func(output: Any) -> str:
        if use_gsm8k:
            # GSM8K: dict/JSON unwrap + answer normalization (strip $, commas, units).
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
            # AgentGenerator template often emits `## Thought ... ## answer <value>`. Extract the last `## answer` / `## final_answer` section body so the GSM8K scoring regex (extract_last_number) does not pick numbers out of the reasoning prose that would override the correct final number.
            _section_pattern = re.compile(r"(?im)^#{1,4}\s*(?:answer|final[_\s-]*answer|final[_\s-]*output)\s*$")
            _matches = list(_section_pattern.finditer(content))
            if _matches:
                _last = _matches[-1]
                _body = content[_last.end():]
                _next_header = re.search(r"(?m)^#{1,4}\s+\S", _body)
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
                _l = re.sub(r'\s*(dollars|cents|percent|%|units?|items?|people|hours?|minutes?|seconds?|days?|weeks?|months?|years?|miles?|km|meters?|feet|inches?|pounds?|kg|grams?|liters?|gallons?|pieces?|pairs?|boxes?|bags?|bottles?|cups?|slices?|tickets?|books?|pages?|students?|children|adults?|apples?|oranges?|balls?|cars?|dogs?|cats?|birds?|fish|eggs?|cookies?|candies|marbles?|coins?|stamps?|stickers?|flowers?|trees?|shirts?|toys?|games?|points?|goals?|runs?|laps?|trips?|times?|ways?|groups?|rows?|columns?|layers?|levels?|steps?|blocks?|miles?|km|mph|kph)\s*$', '', _l, flags=re.IGNORECASE)
                _l = _l.rstrip('.')
                _normalized_lines.append(_l)
            return '\n'.join(_normalized_lines)
        if use_drop:
            # DROP: dict/JSON unwrap + markdown section extraction + prefix stripping (matches AFlow pipeline)
            if isinstance(output, dict):
                output = output.get("answer", output.get("result", output))
            if output is None:
                return ""
            _content = str(output).strip()
            if _content.startswith("{"):
                try:
                    import json as _json
                    _parsed = _json.loads(_content)
                    if isinstance(_parsed, dict):
                        for _k in ("answer", "result", "output", "response", "final_answer"):
                            if _k in _parsed:
                                _content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            # AgentGenerator emits `## Thought ... ## answer <value>` template. Extract the last `## answer` / `## final_answer` section body so parse_mode=str does not leak the reasoning prose.
            _section_pattern = re.compile(r"(?im)^#{1,4}\s*(?:answer|final[_\s-]*answer|final[_\s-]*output)\s*$")
            _matches = list(_section_pattern.finditer(_content))
            if _matches:
                _last = _matches[-1]
                _body = _content[_last.end():]
                _next_header = re.search(r"(?m)^#{1,4}\s+\S", _body)
                if _next_header:
                    _body = _body[:_next_header.start()]
                _content = _body.strip()
            # Secondary fallback: if preamble tokens like `## Thought` / "Reasoning:" remain, keep only the last non-empty line.
            _lines = [ln.strip() for ln in _content.split("\n") if ln.strip()]
            if len(_lines) > 1:
                _lowered_pre = " ".join(_lines[:-1]).lower()
                if any(_kw in _lowered_pre for _kw in ("thought", "reasoning", "step ", "explain", "because", "therefore")):
                    _content = _lines[-1]
            _content = re.sub(r"^\s*(?:Answer|Final answer|The answer is|The final answer is)[: \s]+", "", _content, flags=re.IGNORECASE).strip()
            _content = _content.strip().strip('"').strip("'").strip(".").strip(",").strip()
            return _content
        if use_math:
            return _postprocess_math_answer(output)
        if use_mbpp:
            # MBPP: extract code matching AFlow CodeFormatter pipeline
            import json as _json
            from evoagentx.utils.sanitize import code_extract as _code_extract
            if isinstance(output, dict):
                output = output.get("code", output.get("solution", output.get("answer", output.get("result", output))))
            if output is None:
                return ""
            _content = str(output).strip()
            if _content.startswith("{"):
                try:
                    _parsed = _json.loads(_content)
                    if isinstance(_parsed, dict):
                        for _k in ("code", "solution", "answer", "result", "output"):
                            if _k in _parsed:
                                _content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            # AFlow-style: extract ALL markdown code blocks (not just first)
            _py_blocks = re.findall(r"```python\s*([\s\S]*?)\s*```", _content)
            if _py_blocks:
                _content = "\n\n".join(_py_blocks)
            else:
                _gen_blocks = re.findall(r"```\s*([\s\S]*?)\s*```", _content)
                if _gen_blocks:
                    _content = "\n\n".join(_gen_blocks)
            _extracted = _code_extract(_content)
            if _extracted and _extracted.strip():
                _content = _extracted
            return _content
        if use_humaneval:
            # HumanEval: extract code from LLM output (unchanged)
            import json as _json
            from evoagentx.utils.sanitize import code_extract as _code_extract
            if isinstance(output, dict):
                output = output.get("code", output.get("solution", output.get("answer", output.get("result", output))))
            if output is None:
                return ""
            _content = str(output).strip()
            if _content.startswith("{"):
                try:
                    _parsed = _json.loads(_content)
                    if isinstance(_parsed, dict):
                        for _k in ("code", "solution", "answer", "result", "output"):
                            if _k in _parsed:
                                _content = str(_parsed[_k]).strip()
                                break
                except (ValueError, TypeError):
                    pass
            _fence_m = re.search(r"```(?:python)?\s*\n(.*?)```", _content, re.DOTALL)
            if _fence_m:
                _content = _fence_m.group(1).strip()
            _extracted = _code_extract(_content)
            if _extracted and _extracted.strip():
                _content = _extracted
            return _content
        if isinstance(output, dict):
            output = output.get("answer", output.get("result", output))
        if output is None:
            return ""
        content = str(output).strip()

        # --- JSON unwrapping (handles str-mode end nodes that still output JSON) ---
        if content.startswith("{"):
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    for _jk in ("answer", "result", "output", "response", "final_answer", "inferred_answer"):
                        if _jk in parsed_json:
                            content = str(parsed_json[_jk]).strip()
                            break
                    else:
                        for _jv in parsed_json.values():
                            if isinstance(_jv, str) and _jv.strip():
                                content = _jv.strip()
                                break
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Extract from XML answer tags if present
        _xml_m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)
        if _xml_m:
            content = _xml_m.group(1).strip()
        # Remove markdown fences
        content = content.replace("```json", "").replace("```", "")

        # --- Prefix stripping (safe patterns; no "The"/"It is" removal) ---
        def _strip_prefixes(text):
            text = re.sub(r"^\s*(?:Answer|Final answer|The answer is|The final answer is|The extracted output is|The direct answer is|The output is|Result|Output)[:\s]+", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^\s*(?:The\s+)?(?:answer|result|output)\s+(?:is|was|=)[:\s]*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^(?:Based on\b[^,]{0,120},\s*)", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^(?:According to\b[^,]{0,120},\s*)", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^(?:From (?:the )?(?:workflow|execution|context|results?|information|evidence)[^,]{0,80},\s*)", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^(?:After\b[^,]{0,120},\s*)", "", text, flags=re.IGNORECASE)
            return text.strip()

        content = _strip_prefixes(content)
        # Take first non-empty line
        _lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        if _lines:
            content = _lines[0]
        content = _strip_prefixes(content)

        # --- First-sentence truncation ---
        _ABBREVS = {"dr", "mr", "mrs", "ms", "jr", "sr", "st", "prof", "gen", "col", "sgt", "lt", "capt", "rev", "hon", "vs", "etc", "inc", "ltd", "co", "no"}
        for _sm in re.finditer(r"[.!?]\s+[A-Z]", content):
            _before = content[:_sm.start() + 1].strip()
            if len(_before.split()) >= 2:
                _last_word = _before.rstrip(".").split()[-1].lower()
                if _last_word in _ABBREVS:
                    continue
                content = _before
                break

        # --- Trailing explanation truncation ---
        _comma_expl = re.search(
            r"[,;]\s+(?:because|since|as |which|who|that |but |and |however|this |it |the |where|when|a |an |is |was |are |were )",
            content, re.IGNORECASE
        )
        if _comma_expl:
            content = content[:_comma_expl.start()].strip()

        # Strip quotes, trailing punctuation
        content = content.strip().strip('"').strip("'").strip(".").strip(",").strip()

        # --- AFlow-style normalize_answer ---
        import string as _string
        def _normalize_answer(s):
            s = s.lower()
            exclude = set(_string.punctuation)
            s = "".join(ch for ch in s if ch not in exclude)
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            return " ".join(s.split())

        return _normalize_answer(content)

    def on_sample_evaluated(example_id, example, graph, trajectory, prediction, label, metrics):
        try:
            em = _safe_rate(metrics.get("em", metrics.get("pass@1", 0.0)))
            f1 = _safe_rate(metrics.get("f1", metrics.get("pass@1", metrics.get("solve_rate", 0.0))))
            final_observation = max(0.0, min(1.0, f1 if use_aflow_hotpotqa else (0.5 * em + 0.5 * f1)))
            prompt_obs, return_obs, params_obs, edge_obs, structure_obs = extract_obs_from_trajectory(
                workflow_graph=graph,
                trajectory=trajectory,
                label=str(label) if label is not None else None,
                prediction=str(prediction) if prediction is not None else None,
                metrics=metrics,
            )
            judge_payloads = extract_judge_payloads_from_trajectory(
                workflow_graph=graph,
                trajectory=trajectory,
            )
            evidence_buf.add(
                SampleEvidence(
                    example_id=str(example_id),
                    final_observation=final_observation,
                    metrics=metrics,
                    prediction=str(prediction),
                    label=str(label),
                    prompt_obs=prompt_obs,
                    return_obs=return_obs,
                    params_obs=params_obs,
                    edge_obs=edge_obs,
                    structure_obs=structure_obs,
                    judge_payloads=judge_payloads,
                )
            )
        except Exception as e:
            print(
                f">>> [Iter {iteration}] warning: evidence extraction failed for example_id={example_id}: {e}"
            )
        _record_failure_example(example, label, prediction, metrics or {}, trajectory)

    evaluator = Evaluator(
        llm=llm,
        agent_manager=agent_manager,
        collate_func=collate_func,
        output_postprocess_func=output_postprocess_func,
        verbose=True,
        num_workers=num_workers,
        on_sample_evaluated=on_sample_evaluated,
    )

    print(f"\n>>> [Iter {iteration}] Evaluating workflow ...")
    with suppress_logger_info():
        results = evaluator.evaluate(
            graph=workflow_graph,
            benchmark=benchmark,
            eval_mode=eval_mode,
            indices=eval_indices,
            sample_k=sample_k,
            seed=seed,
            update_agents=True,
        )

    evaluation_records = evaluator.get_all_evaluation_records()
    if evaluation_records:
        recovered_results: Dict[str, float] = {}
        if use_aflow_hotpotqa or use_math or use_humaneval or use_mbpp or use_gsm8k or use_drop:
            if eval_indices:
                total_f1 = 0.0
                for idx in eval_indices:
                    example = benchmark.get_example_by_index(index=int(idx), mode=eval_mode)
                    if example is None:
                        continue
                    _raw_eid = benchmark.get_id(example=example)
                    record = evaluation_records.get(_raw_eid) or evaluation_records.get(str(_raw_eid))
                    _rm = (record or {}).get("metrics") or {}; total_f1 += _safe_rate(_rm.get("f1", _rm.get("pass@1", _rm.get("solve_rate", 0.0))))
                recovered_results = {"f1": total_f1 / max(1, len(eval_indices))}
            else:
                total_f1 = sum(
                    _safe_rate(((record or {}).get("metrics") or {}).get("f1", ((record or {}).get("metrics") or {}).get("pass@1", ((record or {}).get("metrics") or {}).get("solve_rate", 0.0))))
                    for record in evaluation_records.values()
                )
                recovered_results = {"f1": total_f1 / max(1, len(evaluation_records))}
        else:
            metric_keys = set()
            for record in evaluation_records.values():
                metric_keys.update((record.get("metrics") or {}).keys())
            for key in metric_keys:
                values = [
                    _safe_rate((record.get("metrics") or {}).get(key, 0.0))
                    for record in evaluation_records.values()
                ]
                if values:
                    recovered_results[key] = sum(values) / len(values)
        if recovered_results and (
            not results
            or any(abs(_safe_rate(results.get(k, 0.0)) - _safe_rate(v)) > 1e-12 for k, v in recovered_results.items())
        ):
            print(
                f">>> [Iter {iteration}] using benchmark metrics recomputed from evaluation records: "
                f"{recovered_results}"
            )
            results = recovered_results

    if failure_examples:
        # Sort worst-first; downstream _truncate_failure_examples bucket-samples
        # by failure_mode out of this pool, so we keep enough variety here for
        # the bucketing to actually see a distribution rather than the same
        # 5 lowest-F1 cases (which on a saturated incumbent are usually
        # all the same failure mode).
        failure_examples.sort(key=lambda item: (float(item.get("f1", 1.0)), float(item.get("em", 1.0))))
        failure_examples = failure_examples[:25]

    if not run_rca:
        all_ev = evidence_buf.snapshot()
        return results, [], all_ev, {}, evaluation_records, failure_examples

    all_ev = evidence_buf.snapshot()
    if not all_ev:
        if not results:
            print(
                f">>> [Iter {iteration}] warning: evaluator produced no valid metrics and no evidence. "
                "This indicates workflow execution failure or callback-side exceptions on every sample."
            )
        return results, [], all_ev, {}, evaluation_records, failure_examples

    all_ev = enrich_evidence_with_llm_judge(
        evidences=all_ev,
        workflow_graph=workflow_graph,
        llm=llm,
        sample_limit=min(20, len(all_ev)),
        max_workers=max(1, int(num_workers or 1)),
    )
    all_ev = compute_backward_consistency_scores(
        evidences=all_ev,
        workflow_graph=workflow_graph,
    )

    factor_engine = build_multi_sample_factor_graph(
        workflow_graph,
        all_ev,
        health_prior=float(getattr(calibration_profile, "health_prior", 0.85)),
        calibration_profile=calibration_profile,
    )
    factor_engine.run_loopy_belief_propagation(
        max_iter=64,
        tolerance=5e-4,
        damping=0.5,
        patience=12,
        verbose=False,
    )
    root_causes = factor_engine.get_root_causes()
    node_stats = _summarize_node_observations(all_ev)
    return results, root_causes, all_ev, node_stats, evaluation_records, failure_examples

def _refresh_package_rca(
    *,
    package: EvaluationPackage,
    workflow_graph: WorkFlowGraph,
    calibration_profile: Optional[FactorCalibrationProfile],
) -> EvaluationPackage:
    if not getattr(package, "evidences", None):
        return package
    factor_engine = build_multi_sample_factor_graph(
        workflow_graph,
        package.evidences,
        health_prior=float(getattr(calibration_profile, "health_prior", 0.85)),
        calibration_profile=calibration_profile,
    )
    factor_engine.run_loopy_belief_propagation(
        max_iter=64,
        tolerance=5e-4,
        damping=0.5,
        patience=12,
        verbose=False,
    )
    package.root_causes = factor_engine.get_root_causes()
    package.node_stats = _summarize_node_observations(package.evidences)
    package.obs_coverage = _compute_observation_coverage(
        evidences=package.evidences,
        workflow_graph=workflow_graph,
    )
    return package


def _get_or_run_evaluation_package(
    *,
    evaluation_cache: EvaluationCache,
    workflow_graph: WorkFlowGraph,
    llm,
    agent_manager: AgentManager,
    benchmark: HotPotQA,
    eval_indices: Optional[List[int]],
    eval_mode: str,
    iteration: int,
    num_workers: int = 50,
    calibration_profile: Optional[FactorCalibrationProfile] = None,
    run_rca: bool = True,
) -> Tuple[EvaluationPackage, bool]:
    canonical_workflow, _, _ = _canonicalize_workflow_graph(workflow_graph)

    def _runner(eval_workflow: WorkFlowGraph) -> EvaluationPackage:
        before_cost = cost_manager.snapshot()
        results, root_causes, evidences, node_stats, evaluation_records, failure_examples = _run_single_iteration(
            iteration=iteration,
            llm=llm,
            workflow_graph=eval_workflow,
            agent_manager=agent_manager,
            benchmark=benchmark,
            eval_indices=eval_indices,
            sample_k=None,
            seed=None,
            eval_mode=eval_mode,
            num_workers=num_workers,
            calibration_profile=calibration_profile,
            run_rca=run_rca,
        )
        after_cost = cost_manager.snapshot()
        delta = cost_manager.diff(before_cost, after_cost)
        return EvaluationPackage(
            workflow_fingerprint="",
            eval_indices=tuple(eval_indices or []),
            results=results,
            evidences=evidences,
            root_causes=root_causes,
            node_stats=node_stats,
            evaluation_records=evaluation_records,
            failure_examples=failure_examples,
            total_tokens_delta=delta["tokens"],
            total_cost_delta=delta["cost"],
            obs_coverage=_compute_observation_coverage(evidences=evidences, workflow_graph=eval_workflow),
        )

    package, cached = evaluation_cache.get_or_evaluate(
        workflow_graph=canonical_workflow,
        llm=llm,
        eval_indices=eval_indices,
        eval_mode=eval_mode,
        runner=lambda: _runner(canonical_workflow),
    )
    if cached and run_rca:
        package = _refresh_package_rca(
            package=package,
            workflow_graph=canonical_workflow,
            calibration_profile=calibration_profile,
        )
        # If cached package has empty evidences, re-run evaluation to collect them
        if not getattr(package, "evidences", None):
            package = _runner(canonical_workflow)
            evaluation_cache.put(
                workflow_graph=canonical_workflow,
                llm=llm,
                eval_indices=eval_indices,
                eval_mode=eval_mode,
                package=package,
            )
            cached = False






    return package, cached


