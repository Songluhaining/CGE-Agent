import hashlib
import json
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Set

from pydantic import Field
from flagent.Graph.engine import FactorGraphEngine
from flagent.Graph.schema import GraphVariable
from flagent.Graph.factors import (
    DataObservationFactor,
    HealthAggregationFactor,
    HealthGatedStepFactor,
    MultiInputStepFactor,
    ObservationFactor,
    UnaryObservationFactor,
)
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.models.base_model import LLMOutputParser


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        val = float(default)
    return max(0.0, min(1.0, val))


PROMPT_OBS_DIMS = ["input_binding", "output_contract", "grounded", "executable"]
RETURN_OBS_DIMS = [
    "type_ok",
    "content_ok",
    "task_ok",
    "exact_ok",
    "overlap_ok",
    "evidence_alignment",
    "entity_fidelity",
    "answer_supported",
    "answer_normalized",
]
PARAMS_OBS_DIMS = ["not_truncated", "format_parseable"]
EDGE_OBS_DIMS = ["edge_consumed", "entity_overlap", "semantic_transfer", "dependency_preserved"]
STRUCTURE_OBS_DIMS = ["role_coverage", "topology_ordering"]
_ROLE_ORDER = {
    "decompose": 0,
    "extract": 1,
    "evidence": 2,
    "synthesize": 3,
    "answer": 4,
    "generic": 5,
}

PROMPT_CHILD_MAP = {
    "input_binding": ("Prompt", "Binding"),
    "output_contract": ("Prompt", "Contract"),
    "grounded": ("Prompt", "Grounding"),
    "executable": ("Prompt", "Grounding"),
}
PARAMS_CHILD_MAP = {
    "not_truncated": ("Params", "Length"),
    "format_parseable": ("Params", "Parse"),
}
RETURN_CHILD_MAP = {
    "type_ok": ("Return", "Type"),
    "answer_normalized": ("Return", "Type"),
    "content_ok": ("Return", "Evidence"),
    "overlap_ok": ("Return", "Evidence"),
    "evidence_alignment": ("Return", "Evidence"),
    "entity_fidelity": ("Return", "Evidence"),
    "answer_supported": ("Return", "Evidence"),
    "task_ok": ("Return", "Task"),
    "exact_ok": ("Return", "Task"),
}
COMPONENT_CHILDREN = {
    "Prompt": ["Binding", "Contract", "Grounding"],
    "Params": ["Length", "Parse"],
    "Return": ["Type", "Evidence", "Task"],
}

# Edge 子部件已折叠为单一 HealthEdge：所有观测维度直接挂到父级健康节点。
EDGE_CHILD_MAP = {
    "edge_consumed": ("Edge", ""),
    "dependency_preserved": ("Edge", ""),
    "entity_overlap": ("Edge", ""),
    "semantic_transfer": ("Edge", ""),
}


@dataclass
class SampleEvidence:
    example_id: str
    final_observation: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[str] = None
    label: Optional[str] = None

    # 鑺傜偣绾ч潤鎬佽娴嬶紙铏界劧鏆傛椂鎸傚湪鏍锋湰閲岋紝浣嗘瀯鍥炬椂浼氬幓閲嶏級
    prompt_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 鏍锋湰绾ц繑鍥炶娴?
    return_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # params obs
    params_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # edge-level runtime obs: "src__TO__tgt" -> dim -> float
    edge_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # structure-level runtime/static obs shared by the workflow execution
    structure_obs: Dict[str, float] = field(default_factory=dict)
    # node_name -> dims judged by LLM
    llm_judged_dims: Dict[str, Set[str]] = field(default_factory=dict)
    # node_name -> lightweight payload used by llm-judge / backward consistency
    judge_payloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class JudgeScoreOutput(LLMOutputParser):
    coverage: Optional[float] = Field(default=None)
    independence: Optional[float] = Field(default=None)
    ordering: Optional[float] = Field(default=None)
    relevance: Optional[float] = Field(default=None)
    accuracy: Optional[float] = Field(default=None)
    completeness: Optional[float] = Field(default=None)
    coherence: Optional[float] = Field(default=None)
    evidence_use: Optional[float] = Field(default=None)
    correctness: Optional[float] = Field(default=None)
    usefulness: Optional[float] = Field(default=None)


class EvidenceBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, SampleEvidence] = {}

    def add(self, ev: SampleEvidence):
        with self._lock:
            self._data[str(ev.example_id)] = ev

    def snapshot(self) -> Dict[str, SampleEvidence]:
        with self._lock:
            return dict(self._data)


def _sanitize_id(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(s))


def _sample_sid(example_id: str) -> str:
    raw = str(example_id)
    safe = _sanitize_id(raw)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{h}"


def _edge_key(source: str, target: str) -> str:
    return f"{source}__TO__{target}"


def _edge_data_var(source: str, target: str, sid: str) -> str:
    return f"DataEdge_{source}__TO__{target}__{sid}"


def _profile_family_entry(
    calibration_profile: Optional[Any],
    family: str,
    dim_name: str,
    default: Dict[str, float],
    source: Optional[str] = None,
) -> Dict[str, float]:
    profile = getattr(calibration_profile, family, {}) if calibration_profile is not None else {}
    entry = (profile or {}).get(dim_name, {})
    if source and isinstance(entry.get("source_profiles"), dict):
        entry = entry.get("source_profiles", {}).get(source, entry)
    return {
        "good_match_prob": float(entry.get("good_match_prob", default["good_match_prob"])),
        "bad_match_prob": float(entry.get("bad_match_prob", default["bad_match_prob"])),
        "slip_prob": float(entry.get("slip_prob", default.get("slip_prob", 0.05))),
    }


def _final_obs_entry(calibration_profile: Optional[Any], dim_name: str, default: Dict[str, float]) -> Dict[str, float]:
    profile = getattr(calibration_profile, "final_obs", {}) if calibration_profile is not None else {}
    entry = (profile or {}).get(dim_name, {})
    return {
        "good_match_prob": float(entry.get("good_match_prob", default["good_match_prob"])),
        "bad_match_prob": float(entry.get("bad_match_prob", default["bad_match_prob"])),
    }


def _step_slip(calibration_profile: Optional[Any], component: str, default: float) -> float:
    profile = getattr(calibration_profile, "step_slip", {}) if calibration_profile is not None else {}
    return float((profile or {}).get(component.lower(), default))


def _default_prompt_obs_profile(dim_name: str) -> Dict[str, float]:
    return {"good_match_prob": 0.58, "bad_match_prob": 0.46, "slip_prob": 0.18}


def _default_params_obs_profile(dim_name: str) -> Dict[str, float]:
    if dim_name == "format_parseable":
        return {"good_match_prob": 0.95, "bad_match_prob": 0.10, "slip_prob": 0.03}
    return {"good_match_prob": 0.90, "bad_match_prob": 0.16, "slip_prob": 0.05}


def _default_return_obs_profile(dim_name: str, is_final_node: bool, source: str = "heuristic") -> Dict[str, float]:
    if source == "ground_truth":
        return {"good_match_prob": 0.95, "bad_match_prob": 0.08, "slip_prob": 0.03}
    if source == "llm_judged":
        return {"good_match_prob": 0.82, "bad_match_prob": 0.22, "slip_prob": 0.09}
    if source == "deterministic" or dim_name in {"type_ok", "answer_normalized"}:
        return {"good_match_prob": 0.88, "bad_match_prob": 0.18, "slip_prob": 0.08}
    if is_final_node:
        return {"good_match_prob": 0.82, "bad_match_prob": 0.28, "slip_prob": 0.10}
    return {"good_match_prob": 0.58, "bad_match_prob": 0.46, "slip_prob": 0.18}


def _default_edge_obs_profile(dim_name: str) -> Dict[str, float]:
    return {"good_match_prob": 0.74, "bad_match_prob": 0.34, "slip_prob": 0.12}


def _default_structure_obs_profile(dim_name: str) -> Dict[str, float]:
    return {"good_match_prob": 0.84, "bad_match_prob": 0.20, "slip_prob": 0.06}


def _parse_first_json_dict(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        candidates = parse_json_from_text(text)
    except Exception:
        candidates = []
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            try:
                data = json.loads(candidate.replace("\n", " "))
            except Exception:
                continue
        if isinstance(data, dict):
            return data
    return None


def _deterministic_sample_ids(evidences: Dict[str, "SampleEvidence"], sample_limit: int) -> List[str]:
    if sample_limit <= 0:
        return []
    ranked = sorted(
        evidences.items(),
        key=lambda item: (
            _safe_rate(getattr(item[1], "final_observation", 0.0), 0.0),
            _safe_rate((getattr(item[1], "metrics", {}) or {}).get("f1", 0.0), 0.0),
            str(item[0]),
        ),
    )
    if len(ranked) <= sample_limit:
        return [example_id for example_id, _ in ranked]
    if sample_limit == 1:
        return [ranked[0][0]]
    selected: List[str] = []
    step = (len(ranked) - 1) / float(sample_limit - 1)
    used = set()
    for idx in range(sample_limit):
        pos = int(round(idx * step))
        pos = max(0, min(len(ranked) - 1, pos))
        example_id = ranked[pos][0]
        if example_id in used:
            continue
        used.add(example_id)
        selected.append(example_id)
    if len(selected) < sample_limit:
        for example_id, _ in ranked:
            if example_id in used:
                continue
            selected.append(example_id)
            used.add(example_id)
            if len(selected) >= sample_limit:
                break
    return selected


def _build_llm_judge_prompt(
    *,
    node_name: str,
    payload: Dict[str, Any],
    label: Optional[str],
) -> Optional[str]:
    roles = payload.get("roles") or []
    prompt_text = _obs_norm_text(payload.get("prompt_text", ""))
    outputs = payload.get("outputs") or {}
    outputs_json = json.dumps(outputs, ensure_ascii=False, default=str)
    label_text = _obs_norm_text(label)
    if "answer" in roles or payload.get("is_final_node"):
        return None

    if "decompose" in roles:
        return (
            "You are judging a multi-hop QA workflow step.\n"
            "Task: score the decomposition quality from 0 to 10.\n"
            "Return JSON with keys coverage, independence, ordering.\n"
            "Scoring rubric:\n"
            "- coverage: do the sub-questions jointly cover what is needed to solve the original problem?\n"
            "- independence: are the sub-questions minimally overlapping and individually answerable?\n"
            "- ordering: is the dependency order logical for multi-hop reasoning?\n"
            f"Node: {node_name}\n"
            f"Prompt:\n{prompt_text}\n"
            f"Outputs:\n{outputs_json}\n"
            "Return only JSON."
        )
    if "extract" in roles or "evidence" in roles:
        return (
            "You are judging a multi-hop QA evidence step.\n"
            "Task: score the extracted evidence quality from 0 to 10.\n"
            "Return JSON with keys relevance, accuracy, completeness.\n"
            "Scoring rubric:\n"
            "- relevance: extracted content matches the sub-question or upstream need\n"
            "- accuracy: extracted entities / relations look factually faithful to the prompt context\n"
            "- completeness: extracted content is sufficient for downstream chaining\n"
            f"Node: {node_name}\n"
            f"Prompt:\n{prompt_text}\n"
            f"Outputs:\n{outputs_json}\n"
            "Return only JSON."
        )
    if "synthesize" in roles:
        return (
            "You are judging a multi-hop QA synthesis step.\n"
            "Task: score the synthesis quality from 0 to 10.\n"
            "Return JSON with keys coherence, evidence_use, completeness.\n"
            "Scoring rubric:\n"
            "- coherence: reasoning is logically connected\n"
            "- evidence_use: evidence is explicitly and correctly used\n"
            "- completeness: the chain is sufficient for answering the final question\n"
            f"Node: {node_name}\n"
            f"Prompt:\n{prompt_text}\n"
            f"Outputs:\n{outputs_json}\n"
            f"Ground truth answer (for reference only if helpful): {label_text}\n"
            "Return only JSON."
        )
    return (
        "You are judging an intermediate workflow step.\n"
        "Score the step from 0 to 10 for correctness and usefulness to the downstream task.\n"
        "Return JSON with keys correctness, usefulness.\n"
        f"Node: {node_name}\n"
        f"Prompt:\n{prompt_text}\n"
        f"Outputs:\n{outputs_json}\n"
        "Return only JSON."
    )


def _normalize_judge_score(value: Any) -> float:
    score = _safe_rate(value, 0.0)
    if score > 1.0:
        score = score / 10.0
    return _clamp01(score)


def _apply_llm_judge_to_node(ev: "SampleEvidence", node_name: str, payload: Dict[str, Any], judged: Dict[str, Any]):
    roles = payload.get("roles") or []
    node_return = ev.return_obs.setdefault(node_name, {})
    judged_dims = ev.llm_judged_dims.setdefault(node_name, set())

    if "decompose" in roles:
        coverage = _normalize_judge_score(judged.get("coverage"))
        independence = _normalize_judge_score(judged.get("independence"))
        ordering = _normalize_judge_score(judged.get("ordering"))
        node_return["question_coverage"] = coverage
        node_return["dependency_order"] = ordering
        node_return["task_ok"] = _clamp01((coverage + independence + ordering) / 3.0)
        judged_dims.update({"question_coverage", "dependency_order", "task_ok"})
        return
    if "extract" in roles or "evidence" in roles:
        relevance = _normalize_judge_score(judged.get("relevance"))
        accuracy = _normalize_judge_score(judged.get("accuracy"))
        completeness = _normalize_judge_score(judged.get("completeness"))
        node_return["evidence_alignment"] = _clamp01((relevance + completeness) / 2.0)
        node_return["entity_fidelity"] = accuracy
        node_return["task_ok"] = _clamp01((relevance + accuracy + completeness) / 3.0)
        judged_dims.update({"evidence_alignment", "entity_fidelity", "task_ok"})
        return
    if "synthesize" in roles:
        coherence = _normalize_judge_score(judged.get("coherence"))
        evidence_use = _normalize_judge_score(judged.get("evidence_use"))
        completeness = _normalize_judge_score(judged.get("completeness"))
        node_return["bridge_consistency"] = _clamp01((coherence + evidence_use) / 2.0)
        node_return["chain_completeness"] = completeness
        node_return["task_ok"] = _clamp01((coherence + evidence_use + completeness) / 3.0)
        judged_dims.update({"bridge_consistency", "chain_completeness", "task_ok"})
        return
    correctness = _normalize_judge_score(judged.get("correctness"))
    usefulness = _normalize_judge_score(judged.get("usefulness"))
    node_return["task_ok"] = _clamp01((correctness + usefulness) / 2.0)
    judged_dims.add("task_ok")


def enrich_evidence_with_llm_judge(
    evidences: Dict[str, "SampleEvidence"],
    workflow_graph: Any,
    llm: Any,
    sample_limit: int = 20,
    max_workers: int = 1,
) -> Dict[str, "SampleEvidence"]:
    if not evidences or llm is None or not hasattr(llm, "single_generate"):
        return evidences

    node_map = {getattr(node, "name", ""): node for node in getattr(workflow_graph, "nodes", []) or []}
    selected_ids = _deterministic_sample_ids(evidences, sample_limit=max(0, int(sample_limit)))
    judge_jobs: List[Tuple[str, str, Dict[str, Any], str]] = []
    for example_id in selected_ids:
        ev = evidences.get(example_id)
        if ev is None:
            continue
        for node_name, payload in (ev.judge_payloads or {}).items():
            if payload.get("is_final_node"):
                continue
            judge_prompt = _build_llm_judge_prompt(node_name=node_name, payload=payload, label=ev.label)
            if judge_prompt:
                judge_jobs.append((example_id, node_name, payload, judge_prompt))

    def _run_single_judge(job: Tuple[str, str, Dict[str, Any], str]) -> Optional[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
        example_id, node_name, payload, judge_prompt = job
        try:
            parsed = llm.generate(
                prompt=judge_prompt,
                parser=JudgeScoreOutput,
                parse_mode="json",
                stream=False,
                output_response=False,
            )
            judged = {
                key: value
                for key, value in (
                    (field_name, getattr(parsed, field_name, None))
                    for field_name in JudgeScoreOutput.get_attrs()
                )
                if value is not None
            }
            if not judged and isinstance(getattr(parsed, "content", None), str):
                judged = _parse_first_json_dict(parsed.content) or {}
            if not judged:
                return None
            return example_id, node_name, payload, judged
        except Exception:
            return None

    max_workers = max(1, int(max_workers or 1))
    judged_results: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    if judge_jobs:
        if max_workers <= 1 or len(judge_jobs) <= 1:
            for job in judge_jobs:
                judged_result = _run_single_judge(job)
                if judged_result is not None:
                    judged_results.append(judged_result)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(judge_jobs))) as executor:
                futures = [executor.submit(_run_single_judge, job) for job in judge_jobs]
                for future in as_completed(futures):
                    judged_result = future.result()
                    if judged_result is not None:
                        judged_results.append(judged_result)

    for example_id, node_name, payload, judged in judged_results:
        ev = evidences.get(example_id)
        if ev is None:
            continue
        _apply_llm_judge_to_node(ev=ev, node_name=node_name, payload=payload, judged=judged)

    for example_id in selected_ids:
        ev = evidences.get(example_id)
        if ev is None:
            continue
        node_outputs = {
            node_name: (payload.get("outputs") or {})
            for node_name, payload in (ev.judge_payloads or {}).items()
            if node_name in node_map
        }
        _migrate_chain_runtime_evidence(
            workflow_graph=workflow_graph,
            node_map=node_map,
            node_outputs=node_outputs,
            return_obs=ev.return_obs,
            edge_obs=ev.edge_obs,
            structure_obs=ev.structure_obs,
        )
    return evidences


def compute_backward_consistency_scores(
    evidences: Dict[str, "SampleEvidence"],
    workflow_graph: Any,
) -> Dict[str, "SampleEvidence"]:
    edges = [
        (getattr(edge, "source", None), getattr(edge, "target", None))
        for edge in (getattr(workflow_graph, "edges", []) or [])
        if getattr(edge, "source", None) and getattr(edge, "target", None)
    ]
    outgoing: Dict[str, List[str]] = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append(_edge_key(source, target))

    for ev in evidences.values():
        label_norm = _normalize_answer(ev.label or "")
        if not label_norm:
            continue
        label_terms = _keyword_set(label_norm)
        final_f1 = _safe_rate((ev.metrics or {}).get("f1", 0.0), 0.0)
        for node_name, payload in (ev.judge_payloads or {}).items():
            if payload.get("is_final_node"):
                continue
            output_text = " ".join(_flatten_output_texts(payload.get("outputs") or {}))
            output_norm = _normalize_answer(output_text)
            if not output_norm:
                continue
            keyword_support = _jaccard(label_terms, _keyword_set(output_norm)) if label_terms else 0.0
            exact_support = 1.0 if label_norm and label_norm in output_norm else 0.0
            support = max(keyword_support, exact_support)
            node_return = ev.return_obs.setdefault(node_name, {})
            current_task = _safe_rate(node_return.get("task_ok", 0.0), 0.0)
            # Sigmoid smoothing replaces hard F1 < 0.6 threshold for backward consistency
            import math
            _failure_weight = 1.0 / (1.0 + math.exp(15.0 * (final_f1 - 0.6)))  # ~1.0 when f1<<0.6, ~0 when f1>>0.6
            _success_weight = 1.0 - _failure_weight
            if support >= 0.45:
                for edge_key in outgoing.get(node_name, []):
                    edge_entry = ev.edge_obs.setdefault(edge_key, {})
                    # Blend edge penalty by failure weight (stronger penalty when f1 is low)
                    penalty_transfer = _clamp01(1.0 - 0.75 * support) * _failure_weight + 1.0 * _success_weight
                    penalty_preserved = _clamp01(1.0 - 0.65 * support) * _failure_weight + 1.0 * _success_weight
                    edge_entry["semantic_transfer"] = min(
                        _safe_rate(edge_entry.get("semantic_transfer", 1.0), 1.0),
                        penalty_transfer,
                    )
                    edge_entry["dependency_preserved"] = min(
                        _safe_rate(edge_entry.get("dependency_preserved", 1.0), 1.0),
                        penalty_preserved,
                    )
            # Blend task_ok update using sigmoid weights
            low_task = _clamp01(0.35 + 0.45 * support)
            high_task = _clamp01(0.45 + 0.45 * support) if support >= 0.30 else current_task
            blended_task = _failure_weight * min(current_task, low_task) + _success_weight * max(current_task, high_task)
            node_return["task_ok"] = _clamp01(blended_task)
    # Fix #3: Add instruction-output consistency observation dimension
    # Measures whether each node's output aligns with its declared task description
    _node_map = {getattr(n, "name", ""): n for n in (getattr(workflow_graph, "nodes", []) or [])}
    for ev in evidences.values():
        for node_name, payload in (ev.judge_payloads or {}).items():
            node_return = ev.return_obs.setdefault(node_name, {})
            # Extract node description from the workflow graph for consistency check
            node_obj = _node_map.get(node_name) if _node_map else None
            if node_obj is None:
                continue
            node_desc = getattr(node_obj, "description", "") or ""
            if not node_desc.strip():
                continue
            output_text = " ".join(_flatten_output_texts(payload.get("outputs") or {}))
            output_norm = _normalize_answer(output_text)
            if not output_norm:
                node_return["instruction_output_consistency"] = 0.0
                continue
            desc_terms = _keyword_set(_normalize_answer(node_desc))
            output_terms = _keyword_set(output_norm)
            if desc_terms:
                consistency = _jaccard(desc_terms, output_terms)
                node_return["instruction_output_consistency"] = _clamp01(consistency)
            else:
                node_return["instruction_output_consistency"] = 0.5  # neutral if no description keywords

    return evidences


def build_multi_sample_factor_graph(
    workflow_graph,
    evidences: Dict[str, SampleEvidence],
    health_prior: float = 0.85,
    calibration_profile: Optional[Any] = None,
) -> FactorGraphEngine:
    """
    鍥犳灉缁撴瀯锛?

    1) 鍏ㄥ眬鍋ュ悍鑺傜偣锛堣法鏍锋湰鍏变韩锛?
       - HealthPrompt_{node}
       - HealthParams_{node}
       - HealthReturn_{node}

    2) 鏍锋湰绉佹湁鍊艰妭鐐?
       - DataPrompt_{node}__{sid}
       - DataParams_{node}__{sid}
       - DataOut_{node}__{sid}

    3) 涓诲洜鏋滈摼
       - 涓婃父 DataOut --> 褰撳墠 DataPrompt
       - HealthPrompt --> 褰撳墠 DataPrompt
       - HealthParams --> 褰撳墠 DataParams
       - DataPrompt + DataParams + HealthReturn --> 褰撳墠 DataOut

    4) 瑙傛祴缁撴瀯
       - Prompt 瑙傛祴锛堥潤鎬侊級锛歄bsPrompt_* 鍙繛 HealthPrompt
       - Return 瑙傛祴锛堟牱鏈骇锛夛細ObsReturn_* 杩?HealthReturn 鍜?DataOut
       - Final 瑙傛祴锛氭渶缁堣妭鐐?DataOut 鏄惁姝ｇ‘
    """
    engine = FactorGraphEngine()
    calibrated_health_prior = float(getattr(calibration_profile, "health_prior", health_prior))
    nodes = workflow_graph.list_nodes()
    edges = [
        (getattr(edge, "source", None), getattr(edge, "target", None))
        for edge in (getattr(workflow_graph, "edges", []) or [])
        if getattr(edge, "source", None) and getattr(edge, "target", None)
    ]

    # ============================================================
    # 1) 鍒涘缓鍏ㄥ眬鍋ュ悍鑺傜偣锛堝叡浜級
    # ============================================================
    for node_name in nodes:
        for comp in ["Prompt", "Params", "Return"]:
            var_name = f"Health{comp}_{node_name}"
            if var_name not in engine.variables:
                hv = GraphVariable(name=var_name, is_health_node=True)
                hv.set_prior(calibrated_health_prior)
                engine.add_variable(hv)
            child_vars: List[str] = []
            for subtype in COMPONENT_CHILDREN.get(comp, []):
                child_name = f"Health{comp}{subtype}_{node_name}"
                child_vars.append(child_name)
                if child_name not in engine.variables:
                    chv = GraphVariable(name=child_name, is_health_node=True)
                    chv.set_prior(calibrated_health_prior)
                    engine.add_variable(chv)
            agg_name = f"HealthAgg{comp}_{node_name}"
            if agg_name not in engine.factors:
                engine.add_factor(
                    HealthAggregationFactor(
                        name=agg_name,
                        component_vars=child_vars,
                        output_var=var_name,
                        slip_prob=0.02,
                        guess_prob=0.05,
                    )
                )

    # Edge 层：折叠为单一 HealthEdge_{edge}，不再区分 Binding/Semantic 子部件。
    for source_name, target_name in edges:
        edge_suffix = _edge_key(source_name, target_name)
        parent_name = f"HealthEdge_{edge_suffix}"
        if parent_name not in engine.variables:
            hv = GraphVariable(name=parent_name, is_health_node=True)
            hv.set_prior(calibrated_health_prior)
            engine.add_variable(hv)

    # ============================================================
    # 3) 鎵炬渶缁堣妭鐐癸紙鍗曞嚭鍙ｏ級
    # ============================================================
    end_nodes = workflow_graph.find_end_nodes()
    if not end_nodes:
        raise ValueError("workflow_graph.find_end_nodes() returned empty; cannot locate final node.")
    final_node = end_nodes[0]

    # ============================================================
    # 4) 瀵规瘡涓牱鏈垱寤虹鏈夊€艰妭鐐?+ 鍥犲瓙
    # ============================================================
    for example_id, ev in evidences.items():
        sid = _sample_sid(example_id)

        # --------------------------------------------------------
        # 4.1 鍒涘缓鏍锋湰绉佹湁鍊艰妭鐐?
        # --------------------------------------------------------
        for node_name in nodes:
            for var_name in [
                f"DataPrompt_{node_name}__{sid}",
                f"DataParams_{node_name}__{sid}",
                f"DataOut_{node_name}__{sid}",
            ]:
                if var_name not in engine.variables:
                    engine.add_variable(GraphVariable(name=var_name, is_health_node=False))
        for source_name, target_name in edges:
            edge_var_name = _edge_data_var(source_name, target_name, sid)
            if edge_var_name not in engine.variables:
                engine.add_variable(GraphVariable(name=edge_var_name, is_health_node=False))

        # --------------------------------------------------------
        # 4.2 涓烘瘡涓?node 鍒涘缓涓诲洜鏋滃洜瀛?
        # --------------------------------------------------------
        for node_name in nodes:
            preds = workflow_graph.get_node_predecessors(node_name)
            for pred_name in preds:
                edge_factor_name = f"EdgeFactor_{_edge_key(pred_name, node_name)}__{sid}"
                if edge_factor_name not in engine.factors:
                    engine.add_factor(
                        MultiInputStepFactor(
                            name=edge_factor_name,
                            input_vars=[f"DataOut_{pred_name}__{sid}"],
                            health_var=f"HealthEdge_{_edge_key(pred_name, node_name)}",
                            output_var=_edge_data_var(pred_name, node_name, sid),
                            slip_prob=_step_slip(calibration_profile, "edge", 0.03),
                            guess_prob=0.005,
                        )
                    )

            # ---------- Prompt 閮ㄥ垎 ----------
            # 涓婃父 DataOut + HealthPrompt -> DataPrompt
            if not preds:
                start_name = f"Start_{node_name}__{sid}"
                if start_name not in engine.variables:
                    engine.add_variable(
                        GraphVariable(
                            name=start_name,
                            is_observed=True,
                            observed_value=1,
                        )
                    )
                prompt_inputs = [start_name]
            else:
                prompt_inputs = [_edge_data_var(p, node_name, sid) for p in preds]

            prompt_factor_name = f"PromptFactor_{node_name}__{sid}"
            if prompt_factor_name not in engine.factors:
                prompt_factor = MultiInputStepFactor(
                    name=prompt_factor_name,
                    input_vars=prompt_inputs,
                    health_var=f"HealthPrompt_{node_name}",
                    output_var=f"DataPrompt_{node_name}__{sid}",
                    slip_prob=_step_slip(calibration_profile, "prompt", 0.08),
                    guess_prob=0.001,
                )
                engine.add_factor(prompt_factor)

            # ---------- Params 部分：HealthParams -> DataParams 的直接二元因子 ----------
            # 原先使用常量 1 作为 MultiInputStepFactor 的输入形成退化结构，
            # 改为 HealthGatedStepFactor 直接建立健康->数据的门控关系。
            params_factor_name = f"ParamsFactor_{node_name}__{sid}"
            if params_factor_name not in engine.factors:
                params_factor = HealthGatedStepFactor(
                    name=params_factor_name,
                    health_var=f"HealthParams_{node_name}",
                    data_var=f"DataParams_{node_name}__{sid}",
                    slip_prob=_step_slip(calibration_profile, "params", 0.05),
                    guess_prob=0.001,
                )
                engine.add_factor(params_factor)

            # ---------- Return 閮ㄥ垎 ----------
            # DataPrompt + DataParams + HealthReturn -> DataOut
            return_factor_name = f"ReturnFactor_{node_name}__{sid}"
            if return_factor_name not in engine.factors:
                return_factor = MultiInputStepFactor(
                    name=return_factor_name,
                    input_vars=[
                        f"DataPrompt_{node_name}__{sid}",
                        f"DataParams_{node_name}__{sid}",
                    ],
                    health_var=f"HealthReturn_{node_name}",
                    output_var=f"DataOut_{node_name}__{sid}",
                    # Keep return link informative but avoid over-hard blame assignment.
                    slip_prob=_step_slip(calibration_profile, "return", 0.12),
                    guess_prob=0.03,
                )
                engine.add_factor(return_factor)

            # ----------------------------------------------------
            # 4.3 鏍锋湰绾?Return 瑙傛祴
            # 鍏ㄩ儴缁熶竴鎴?ObservationFactor锛?
            #   HealthReturn_{node} + DataOut_{node}__{sid} -> ObsReturn
            # 杩欐牱涓夌被 return 瑙傛祴閮戒笌鈥滄湰娆℃牱鏈緭鍑虹姸鎬佲€濈粦瀹?
            # ----------------------------------------------------
            node_return_obs = getattr(ev, "return_obs", {}).get(node_name, {})

            for metric_name, metric_val in node_return_obs.items():
                if metric_name not in RETURN_OBS_DIMS:
                    continue

                obs_name = f"ObsReturn_{metric_name}_{node_name}__{sid}"
                if obs_name not in engine.variables:
                    obs_var = GraphVariable(
                        name=obs_name,
                        is_health_node=False,
                        is_observed=True,
                        observed_value=_clamp01(metric_val),
                    )
                    engine.add_variable(obs_var)

                factor_name = f"ReturnObsFactor_{metric_name}_{node_name}__{sid}"
                if factor_name not in engine.factors:
                    target_comp, target_subtype = RETURN_CHILD_MAP.get(metric_name, ("Return", "Task"))
                    is_final_return_obs = node_name == final_node
                    judged_dims = set((getattr(ev, "llm_judged_dims", {}) or {}).get(node_name, set()) or [])
                    if is_final_return_obs and metric_name in {"exact_ok", "overlap_ok"}:
                        return_source = "ground_truth"
                    elif metric_name in judged_dims:
                        return_source = "llm_judged"
                    elif metric_name in {"type_ok", "answer_normalized"}:
                        return_source = "deterministic"
                    else:
                        return_source = "heuristic"
                    return_entry = _profile_family_entry(
                        calibration_profile,
                        "return_obs",
                        metric_name,
                        _default_return_obs_profile(metric_name, is_final_return_obs, source=return_source),
                        source=return_source,
                    )
                    factor = ObservationFactor(
                        name=factor_name,
                        health_var=f"Health{target_comp}{target_subtype}_{node_name}",
                        data_var=f"DataOut_{node_name}__{sid}",
                        obs_var=obs_name,
                        # Weaken deterministic pull from heuristic return observations.
                        slip_prob=return_entry["slip_prob"],
                        bad_match_prob=return_entry["bad_match_prob"],
                    )
                    engine.add_factor(factor)

            # ----------------------------------------------------
            # 4.3.5 Params obs
            #   HealthParams_{node} + DataParams_{node}__{sid} -> ObsParams
            # ----------------------------------------------------
            node_params_obs = getattr(ev, "params_obs", {}).get(node_name, {})

            for metric_name, metric_val in node_params_obs.items():
                if metric_name not in PARAMS_OBS_DIMS:
                    continue

                obs_name = f"ObsParams_{metric_name}_{node_name}__{sid}"
                if obs_name not in engine.variables:
                    obs_var = GraphVariable(
                        name=obs_name,
                        is_health_node=False,
                        is_observed=True,
                        observed_value=_clamp01(metric_val),
                    )
                    engine.add_variable(obs_var)

                factor_name = f"ParamsObsFactor_{metric_name}_{node_name}__{sid}"
                if factor_name not in engine.factors:
                    target_comp, target_subtype = PARAMS_CHILD_MAP.get(metric_name, ("Params", "Parse"))
                    params_entry = _profile_family_entry(
                        calibration_profile,
                        "params_obs",
                        metric_name,
                        _default_params_obs_profile(metric_name),
                    )
                    factor = ObservationFactor(
                        name=factor_name,
                        health_var=f"Health{target_comp}{target_subtype}_{node_name}",
                        data_var=f"DataParams_{node_name}__{sid}",
                        obs_var=obs_name,
                        slip_prob=params_entry["slip_prob"],
                        bad_match_prob=params_entry["bad_match_prob"],
                    )
                    engine.add_factor(factor)

            # ----------------------------------------------------
            # 4.3.6 Prompt obs (sample-level)
            #   HealthPrompt{Sub}_{node} + DataPrompt_{node}__{sid} -> ObsPrompt
            # ----------------------------------------------------
            node_prompt_obs = getattr(ev, "prompt_obs", {}).get(node_name, {})

            for metric_name, metric_val in node_prompt_obs.items():
                if metric_name not in PROMPT_OBS_DIMS:
                    continue

                obs_name = f"ObsPrompt_{metric_name}_{node_name}__{sid}"
                if obs_name not in engine.variables:
                    obs_var = GraphVariable(
                        name=obs_name,
                        is_health_node=False,
                        is_observed=True,
                        observed_value=_clamp01(metric_val),
                    )
                    engine.add_variable(obs_var)

                factor_name = f"PromptObsFactor_{metric_name}_{node_name}__{sid}"
                if factor_name not in engine.factors:
                    target_comp, target_subtype = PROMPT_CHILD_MAP.get(metric_name, ("Prompt", "Prompt"))
                    prompt_entry = _profile_family_entry(
                        calibration_profile,
                        "prompt_obs",
                        metric_name,
                        _default_prompt_obs_profile(metric_name),
                    )
                    factor = ObservationFactor(
                        name=factor_name,
                        health_var=f"Health{target_comp}{target_subtype}_{node_name}",
                        data_var=f"DataPrompt_{node_name}__{sid}",
                        obs_var=obs_name,
                        slip_prob=prompt_entry.get("slip_prob", 0.1),
                        bad_match_prob=prompt_entry["bad_match_prob"],
                    )
                    engine.add_factor(factor)

        for source_name, target_name in edges:
            edge_key = _edge_key(source_name, target_name)
            node_edge_obs = getattr(ev, "edge_obs", {}).get(edge_key, {})
            for metric_name, metric_val in node_edge_obs.items():
                if metric_name not in EDGE_OBS_DIMS:
                    continue
                obs_name = f"ObsEdge_{metric_name}_{edge_key}__{sid}"
                if obs_name not in engine.variables:
                    engine.add_variable(
                        GraphVariable(
                            name=obs_name,
                            is_health_node=False,
                            is_observed=True,
                            observed_value=_clamp01(metric_val),
                        )
                    )
                factor_name = f"EdgeObsFactor_{metric_name}_{edge_key}__{sid}"
                if factor_name not in engine.factors:
                    target_comp, target_subtype = EDGE_CHILD_MAP.get(metric_name, ("Edge", ""))
                    edge_entry = _profile_family_entry(
                        calibration_profile,
                        "edge_obs",
                        metric_name,
                        _default_edge_obs_profile(metric_name),
                    )
                    engine.add_factor(
                        ObservationFactor(
                            name=factor_name,
                            health_var=f"Health{target_comp}{target_subtype}_{edge_key}",
                            data_var=_edge_data_var(source_name, target_name, sid),
                            obs_var=obs_name,
                            slip_prob=edge_entry["slip_prob"],
                            bad_match_prob=edge_entry["bad_match_prob"],
                        )
                    )

        # --------------------------------------------------------
        # 4.4 鏈€缁堣娴嬶細鏈€缁堣妭鐐硅緭鍑烘槸鍚︽纭?
        # --------------------------------------------------------
        em_obs_value = float(max(0.0, min(1.0, float((ev.metrics or {}).get("em", ev.final_observation) or 0.0))))
        f1_obs_value = float(max(0.0, min(1.0, float((ev.metrics or {}).get("f1", ev.final_observation) or 0.0))))

        obs_final_em_name = f"ObsFinalEM_{final_node}__{sid}"
        if obs_final_em_name not in engine.variables:
            engine.add_variable(
                GraphVariable(
                    name=obs_final_em_name,
                    is_health_node=False,
                    is_observed=True,
                    observed_value=em_obs_value,
                )
            )

        final_em_factor_name = f"FinalEMObsFactor_{final_node}__{sid}"
        if final_em_factor_name not in engine.factors:
            final_em_entry = _final_obs_entry(
                calibration_profile,
                "em",
                {"good_match_prob": 0.92, "bad_match_prob": 0.08},
            )
            engine.add_factor(
                DataObservationFactor(
                    name=final_em_factor_name,
                    data_var=f"DataOut_{final_node}__{sid}",
                    obs_var=obs_final_em_name,
                    good_match_prob=final_em_entry["good_match_prob"],
                    bad_match_prob=final_em_entry["bad_match_prob"],
                )
            )

        obs_final_f1_name = f"ObsFinalF1_{final_node}__{sid}"
        if obs_final_f1_name not in engine.variables:
            engine.add_variable(
                GraphVariable(
                    name=obs_final_f1_name,
                    is_health_node=False,
                    is_observed=True,
                    observed_value=f1_obs_value,
                )
            )

        final_f1_factor_name = f"FinalF1ObsFactor_{final_node}__{sid}"
        if final_f1_factor_name not in engine.factors:
            final_f1_entry = _final_obs_entry(
                calibration_profile,
                "f1",
                {"good_match_prob": 0.86, "bad_match_prob": 0.14},
            )
            engine.add_factor(
                DataObservationFactor(
                    name=final_f1_factor_name,
                    data_var=f"DataOut_{final_node}__{sid}",
                    obs_var=obs_final_f1_name,
                    good_match_prob=final_f1_entry["good_match_prob"],
                    bad_match_prob=final_f1_entry["bad_match_prob"],
                )
            )

    return engine


_OBS_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")
_OBS_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_OBS_REFUSAL_PREFIXES = (
    "error",
    "i apologize",
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "sorry,",
)
_OBS_RETRIEVAL_CUES = (
    "knowledge base",
    "database",
    "retrieve",
    "retrieval",
    "search",
    "look up",
    "evidence",
    "passage",
    "reference",
    "document",
    "wiki",
    "wikipedia",
)
_OBS_EVIDENCE_INPUT_CUES = (
    "document",
    "documents",
    "context",
    "reference",
    "references",
    "passage",
    "passages",
    "evidence",
    "facts",
    "retrieved_data",
    "support",
    "article",
    "articles",
)
_OBS_GROUNDING_CUES = (
    "based on",
    "using the",
    "read and understand",
    "provided",
    "given",
    "from the input",
    "from the above",
    "according to",
    "refer to",
    "use the following",
)


def _obs_norm_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _obs_lower_text(value: Any) -> str:
    return _obs_norm_text(value).lower()


def _is_response_message(msg: Any) -> bool:
    msg_type = getattr(msg, "msg_type", None)
    if msg_type is None:
        return False
    msg_type_value = getattr(msg_type, "value", msg_type)
    return _obs_lower_text(msg_type_value) == "response"


def _flatten_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return _obs_norm_text(prompt)
    if isinstance(prompt, dict):
        try:
            return json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        except Exception:
            return _obs_norm_text(prompt)
    if isinstance(prompt, list):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, dict):
                role = _obs_norm_text(item.get("role", ""))
                content = _obs_norm_text(item.get("content", ""))
                if content:
                    parts.append(f"{role}: {content}" if role else content)
            else:
                text = _obs_norm_text(item)
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return _obs_norm_text(prompt)


def _split_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_obs_norm_text(x) for x in value if _obs_norm_text(x)]
    if isinstance(value, dict):
        return [_obs_norm_text(v) for _, v in sorted(value.items()) if _obs_norm_text(v)]

    text = _obs_norm_text(value)
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items: List[str] = []
    for line in lines:
        if line.startswith("- ") or line.startswith("* "):
            items.append(line[2:].strip())
        else:
            items.append(line)

    if len(items) == 1 and "," in items[0]:
        parts = [part.strip() for part in items[0].split(",") if part.strip()]
        if len(parts) >= 2:
            return parts
    return items


def _looks_like_refusal(value: Any) -> bool:
    text = _obs_lower_text(value)
    if not text:
        return True
    if text.startswith(_OBS_REFUSAL_PREFIXES):
        return True
    return "cannot comply" in text or "unable to" in text


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = _OBS_ARTICLES_RE.sub(" ", no_punc)
    return " ".join(no_articles.split())


def _answer_f1(prediction: str, label: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    label_tokens = _normalize_answer(label).split()
    if not pred_tokens or not label_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(label_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(label_tokens)
    return (2 * precision * recall) / (precision + recall)


def _extract_prompt_template_from_node(node: Any) -> str:
    if not getattr(node, "agents", None):
        return ""

    def _find_template(obj: Any) -> str:
        if isinstance(obj, dict):
            prompt = obj.get("prompt")
            if isinstance(prompt, str) and _OBS_PLACEHOLDER_RE.search(prompt):
                return prompt
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    found = _find_template(value)
                    if found:
                        return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find_template(item)
                if found:
                    return found
        return ""

    for agent_spec in node.agents:
        if isinstance(agent_spec, str):
            continue
        if isinstance(agent_spec, dict):
            found = _find_template(agent_spec)
            if found:
                return found
        elif hasattr(agent_spec, "actions"):
            for action in (getattr(agent_spec, "actions", None) or []):
                prompt = getattr(action, "prompt", None)
                if isinstance(prompt, str) and _OBS_PLACEHOLDER_RE.search(prompt):
                    return prompt
    return ""


def _expected_type_ok(expected_type: str, value: Any) -> int:
    type_text = _obs_lower_text(expected_type)
    if not type_text:
        return 1 if value is not None else 0

    if type_text.startswith("list") or "array" in type_text:
        return 1 if isinstance(value, list) else 0
    if type_text.startswith("dict") or "object" in type_text:
        return 1 if isinstance(value, dict) else 0
    if "bool" in type_text:
        return 1 if isinstance(value, bool) else 0
    if "int" in type_text:
        return 1 if isinstance(value, int) and not isinstance(value, bool) else 0
    if "float" in type_text or "number" in type_text:
        return 1 if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    if type_text in ("string", "str", "text"):
        return 1 if isinstance(value, str) and value.strip() else 0
    return 1 if value is not None else 0


def _nonempty_value_ok(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, dict)):
        return 1 if len(value) > 0 else 0
    return 1


def _json_cast_for_expected_type(value: Any, expected_type: str) -> Any:
    if not isinstance(value, str):
        return value
    type_text = _obs_lower_text(expected_type)
    if not (
        type_text.startswith("list")
        or type_text.startswith("dict")
        or "array" in type_text
        or "object" in type_text
    ):
        return value
    try:
        parsed = json.loads(value)
    except Exception:
        return value
    return parsed


def _get_structured_outputs_from_message(msg: Any, node: Any) -> Dict[str, Any]:
    content = getattr(msg, "content", None)
    outputs: Dict[str, Any] = {}

    for param in getattr(node, "outputs", []):
        name = param.name
        value = None

        if content is None:
            outputs[name] = None
            continue

        if isinstance(content, dict) and name in content:
            value = content.get(name)
        elif hasattr(content, name):
            value = getattr(content, name)
        else:
            try:
                if hasattr(content, "to_dict"):
                    data = content.to_dict()
                    if isinstance(data, dict) and name in data:
                        value = data[name]
            except Exception:
                pass

            if value is None:
                try:
                    if hasattr(content, "get_structured_data"):
                        data = content.get_structured_data()
                        if isinstance(data, dict) and name in data:
                            value = data[name]
                except Exception:
                    pass

            if value is None and isinstance(content, str) and len(getattr(node, "outputs", [])) == 1:
                value = content.strip()
            elif value is None and len(getattr(node, "outputs", [])) == 1:
                serialized = _obs_norm_text(content)
                if serialized and serialized not in ("None", "{}"):
                    value = serialized

        value = _json_cast_for_expected_type(value, getattr(param, "type", ""))
        outputs[name] = value

    return outputs


def _node_has_external_evidence_input(node: Any) -> bool:
    texts: List[str] = []
    for inp in getattr(node, "inputs", []):
        texts.append(_obs_lower_text(getattr(inp, "name", "")))
        texts.append(_obs_lower_text(getattr(inp, "description", "")))
    joined = " ".join(texts)
    return any(cue in joined for cue in _OBS_EVIDENCE_INPUT_CUES)


def _node_has_bundled_context_input(node: Any, prompt_text: str) -> bool:
    """
    Some single-node QA workflows carry all evidence inside a generic input like
    `goal`/`problem` rather than exposing a dedicated `context` field. Treat them
    as executable when the rendered prompt clearly contains bundled documents.
    """
    input_names = {
        _obs_lower_text(getattr(inp, "name", ""))
        for inp in getattr(node, "inputs", [])
        if getattr(inp, "required", True)
    }
    has_generic_container = bool(input_names & {"goal", "problem", "question", "input"})
    prompt_lower = _obs_lower_text(prompt_text)
    has_context_cue = any(
        cue in prompt_lower
        for cue in (
            "reference documents",
            "documents above",
            "provided documents",
            "provided context",
            "reference context",
            "context:",
        )
    )
    return has_generic_container and has_context_cue


def _score_prompt_observation(node: Any, prompt_text: str, template_text: str) -> Dict[str, float]:
    prompt_lower = _obs_lower_text(prompt_text)
    template_lower = _obs_lower_text(template_text)
    desc_lower = _obs_lower_text(getattr(node, "description", ""))

    input_names = [inp.name for inp in getattr(node, "inputs", []) if getattr(inp, "required", True)]
    output_names = [out.name for out in getattr(node, "outputs", []) if getattr(out, "required", True)]

    unresolved_placeholders = _OBS_PLACEHOLDER_RE.findall(prompt_text)
    unresolved_set = set(unresolved_placeholders)
    observable_text = " ".join([prompt_lower, template_lower, desc_lower])

    if not input_names:
        input_binding = 1.0
    else:
        template_placeholders = set(_OBS_PLACEHOLDER_RE.findall(template_text)) if template_text else set()
        placeholder_hits = sum(1 for name in input_names if name in template_placeholders)
        literal_hits = sum(
            1
            for name in input_names
            if re.search(rf"\b{re.escape(name.lower())}\b", observable_text)
        )
        coverage = max(
            placeholder_hits / max(1.0, float(len(input_names))),
            0.75 * (literal_hits / max(1.0, float(len(input_names)))),
        )
        if unresolved_set:
            coverage *= 0.85
        if len(prompt_text) < 50:
            coverage *= 0.85
        input_binding = coverage

    if not output_names:
        output_contract = 1.0
    else:
        hits = sum(1 for name in output_names if re.search(rf"\b{re.escape(name.lower())}\b", observable_text))
        explicit_format_hits = sum(
            1
            for cue in ("json", "schema", "format", "return", "output", "dictionary", "list")
            if cue in observable_text
        )
        if len(output_names) == 1 and output_names[0].lower() == "answer":
            output_contract = 0.55 * (1.0 if hits >= 1 else 0.0) + 0.45 * min(explicit_format_hits / 3.0, 1.0)
        else:
            output_contract = 0.70 * (hits / max(1.0, float(len(output_names)))) + 0.30 * min(explicit_format_hits / 3.0, 1.0)

    grounding_hits = sum(1 for cue in _OBS_GROUNDING_CUES if cue in prompt_lower)
    input_mentions = sum(1 for inp in input_names if inp.lower() in observable_text)
    grounded = (
        0.55 * min(grounding_hits / max(1.0, float(min(len(_OBS_GROUNDING_CUES), 5))), 1.0)
        + 0.30 * min(input_mentions / max(1.0, float(len(input_names) or 1)), 1.0)
        + 0.15 * min(len(prompt_text) / 180.0, 1.0)
    )

    retrieval_demand = any(cue in observable_text for cue in _OBS_RETRIEVAL_CUES)
    has_evidence_input = _node_has_external_evidence_input(node)
    has_bundled_context = _node_has_bundled_context_input(node=node, prompt_text=prompt_text)
    instruction_hits = sum(
        1 for cue in ("must", "only", "return", "output", "do not", "based only", "json") if cue in prompt_lower
    )
    executable = 0.50 + 0.25 * min(instruction_hits / 4.0, 1.0) + 0.25 * min(len(prompt_text) / 150.0, 1.0)
    if retrieval_demand and not (has_evidence_input or has_bundled_context):
        executable -= 0.55
    if unresolved_set:
        executable -= 0.35
    if len(prompt_text) < 20:
        executable -= 0.25

    return {
        "input_binding": _clamp01(input_binding),
        "output_contract": _clamp01(output_contract),
        "grounded": _clamp01(grounded),
        "executable": _clamp01(executable),
    }


def _score_params_observation(node: Any, msg: Any, outputs: Dict[str, Any], prompt_text: str) -> Dict[str, float]:
    required_output_names = [out.name for out in getattr(node, "outputs", []) if getattr(out, "required", True)]
    content = getattr(msg, "content", None)
    content_text = _obs_norm_text(content)

    trailing_cues = (
        prompt_text.rstrip().endswith("...") and not prompt_text.rstrip().endswith("...."),
        bool(re.search(r"[\[{(]\s*$", prompt_text)),
        bool(re.search(r"[:,]\s*$", prompt_text)),
    )
    has_unresolved_placeholders = bool(_OBS_PLACEHOLDER_RE.search(prompt_text))
    truncation_penalty = 0.0
    if trailing_cues[0]:
        truncation_penalty += 0.35
    if trailing_cues[1]:
        truncation_penalty += 0.30
    if trailing_cues[2]:
        truncation_penalty += 0.30
    if has_unresolved_placeholders:
        truncation_penalty += 0.35
    if len(prompt_text) < 20:
        truncation_penalty += 0.25
    not_truncated = _clamp01(1.0 - truncation_penalty)

    if required_output_names:
        parsed_ok = sum(_nonempty_value_ok(outputs.get(name)) for name in required_output_names) / max(1.0, float(len(required_output_names)))
    else:
        parsed_ok = 1.0 if _nonempty_value_ok(content) == 1 else 0.0
    if _looks_like_refusal(content_text):
        parsed_ok *= 0.15
    format_parseable = _clamp01(parsed_ok)

    return {
        "not_truncated": not_truncated,
        "format_parseable": format_parseable,
    }


def _infer_node_role(node: Any, output_names: List[str]) -> str:
    out_set = {name.lower() for name in output_names}
    desc = _obs_lower_text(getattr(node, "description", ""))

    if "answer" in out_set:
        return "final_answer"
    if any(name in out_set for name in ("query", "search_query", "subquestion")):
        return "query_generation"
    if any(name in out_set for name in ("key_entities", "entities", "relationships", "entity_relations")):
        return "relation_extraction"
    if any(name in out_set for name in ("retrieved_data", "facts", "evidence", "passages", "context")):
        return "retrieval"
    if "answer" in desc:
        return "final_answer"
    return "generic"


def _infer_node_roles(node: Any, output_names: List[str]) -> List[str]:
    text = " ".join(
        [
            _obs_lower_text(getattr(node, "name", "")),
            _obs_lower_text(getattr(node, "description", "")),
            " ".join(_obs_lower_text(name) for name in output_names),
        ]
    )
    roles = set()

    if any(cue in text for cue in ("decompose", "break down", "analyze", "understand", "read_user_query", "subquestion")):
        roles.add("decompose")
    if any(cue in text for cue in ("extract", "entity", "entities", "relation", "key information", "key_information")):
        roles.add("extract")
    if any(cue in text for cue in ("retrieve", "retrieval", "search", "evidence", "facts", "passage", "context", "document")):
        roles.add("evidence")
    if any(cue in text for cue in ("synthesize", "reason", "combine", "infer", "chain", "multi_hop", "multi-hop")):
        roles.add("synthesize")
    if "answer" in text or "final" in text:
        roles.add("answer")

    primary = _infer_node_role(node, output_names)
    if primary == "query_generation":
        roles.add("decompose")
    elif primary == "relation_extraction":
        roles.add("extract")
    elif primary == "retrieval":
        roles.add("evidence")
    elif primary == "final_answer":
        roles.add("answer")

    return sorted(roles) if roles else ["generic"]


def _workflow_role_coverage(workflow_graph: Any) -> Dict[str, Any]:
    covered = set()
    per_node: Dict[str, List[str]] = {}
    for node in getattr(workflow_graph, "nodes", []) or []:
        output_names = [getattr(out, "name", "") for out in getattr(node, "outputs", []) or []]
        roles = _infer_node_roles(node, output_names)
        per_node[node.name] = roles
        covered.update(role for role in roles if role != "generic")
    return {"count": len(covered), "covered_roles": sorted(covered), "per_node": per_node}


def _score_answer_task_ok(answer: Any, label: Optional[str], metrics: Optional[Dict[str, Any]]) -> float:
    answer_text = _obs_norm_text(answer)
    if not answer_text:
        return 0.0

    lowered = answer_text.lower()
    if lowered in {"yes", "no", "noanswer"}:
        return 0.90

    if isinstance(metrics, dict):
        em = float(metrics.get("em", 0.0) or 0.0)
        f1 = float(metrics.get("f1", 0.0) or 0.0)
        if em >= 1.0:
            return 1.0
        if f1 >= 0.8:
            return 0.90
        if f1 >= 0.6:
            return 0.75

    if isinstance(label, str) and label.strip():
        if _normalize_answer(answer_text) == _normalize_answer(label):
            return 1.0
        label_f1 = _answer_f1(answer_text, label)
        if label_f1 >= 0.8:
            return 0.90
        if label_f1 >= 0.6:
            return 0.75

    token_len = len(answer_text.split())
    if token_len > 8:
        return 0.10
    if any(phrase in lowered for phrase in (" because ", " therefore ", " the answer is ", " currently ")):
        return 0.15
    return 0.50


def _flatten_output_texts(outputs: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for value in (outputs or {}).values():
        if isinstance(value, dict):
            texts.extend([_obs_norm_text(v) for v in value.values() if _obs_norm_text(v)])
        elif isinstance(value, list):
            texts.extend([_obs_norm_text(v) for v in value if _obs_norm_text(v)])
        else:
            text = _obs_norm_text(value)
            if text:
                texts.append(text)
    return texts


_EDGE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "be", "this", "that", "these", "those", "from", "into", "then", "than",
    "using", "based", "provided", "context", "evidence", "answer", "question",
}


def _keyword_set(text: str) -> set:
    lowered = _obs_lower_text(text)
    parts = re.findall(r"[a-z0-9_]+", lowered)
    return {p for p in parts if len(p) >= 3 and p not in _EDGE_STOPWORDS}


def _extract_candidate_entities(texts: List[str]) -> set:
    entities = set()
    for text in texts:
        raw = _obs_norm_text(text)
        for match in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,})\b", raw):
            cleaned = match.strip()
            if cleaned:
                entities.add(cleaned.lower())
    return entities


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1.0, len(a | b))


def _score_edge_observation(
    source_node: Any,
    target_node: Any,
    source_outputs: Dict[str, Any],
    target_prompt_text: str,
    target_outputs: Dict[str, Any],
) -> Dict[str, float]:
    source_texts = _flatten_output_texts(source_outputs)
    target_texts = _flatten_output_texts(target_outputs)
    source_joined = " ".join(source_texts)
    target_prompt_joined = _obs_norm_text(target_prompt_text)
    target_combined = " ".join([target_prompt_joined] + target_texts)

    source_keywords = _keyword_set(source_joined)
    target_prompt_keywords = _keyword_set(target_prompt_joined)
    target_all_keywords = _keyword_set(target_combined)
    source_entities = _extract_candidate_entities(source_texts)
    target_entities = _extract_candidate_entities([target_prompt_joined] + target_texts)
    source_output_names = {
        str(name).lower().strip()
        for name in (source_outputs or {}).keys()
        if str(name).strip()
    }
    target_input_names = {
        str(getattr(inp, "name", "")).lower().strip()
        for inp in (getattr(target_node, "inputs", []) or [])
        if str(getattr(inp, "name", "")).strip()
    }
    prompt_lower = _obs_lower_text(target_prompt_joined)
    direct_interface_overlap = len(source_output_names & target_input_names)
    interface_binding = min(
        direct_interface_overlap / max(1.0, float(len(source_output_names) or 1)),
        1.0,
    )
    source_items = []
    for value in (source_outputs or {}).values():
        source_items.extend(_split_items(value))
    exact_value_hits = 0
    for item in source_items[:8]:
        lowered_item = _obs_lower_text(item)
        if len(lowered_item) >= 6 and lowered_item in _obs_lower_text(target_combined):
            exact_value_hits += 1
    value_transfer = min(exact_value_hits / max(1.0, float(min(len(source_items), 4) or 1)), 1.0)

    overlap_prompt = _jaccard(source_keywords, target_prompt_keywords)
    overlap_all = _jaccard(source_keywords, target_all_keywords)
    entity_overlap = len(source_entities & target_entities)
    output_name_hits = sum(1 for name in source_output_names if name and name in prompt_lower)

    source_roles = _infer_node_roles(source_node, list(source_outputs.keys()))
    target_roles = _infer_node_roles(target_node, list(target_outputs.keys()))
    dependency_items = [item.lower() for item in dict.fromkeys(_split_items(next(iter(source_outputs.values()), []))) if item]
    dependency_hits = sum(1 for item in dependency_items if item and item in _obs_lower_text(target_prompt_joined))

    output_name_ratio = min(output_name_hits / max(1.0, float(len(source_output_names) or 1)), 1.0)
    entity_overlap_ratio = min(entity_overlap / max(1.0, float(len(source_entities) or 1)), 1.0) if source_entities else 0.0
    edge_consumed = max(
        output_name_ratio,
        interface_binding,
        overlap_prompt,
        entity_overlap_ratio,
        0.90 * value_transfer,
    )
    semantic_transfer = max(
        min(overlap_all / 0.25, 1.0),
        min(overlap_prompt / 0.22, 1.0),
        entity_overlap_ratio,
        value_transfer,
    )
    if "decompose" in source_roles:
        dependency_preserved = min(dependency_hits / max(1.0, float(len(dependency_items) or 1)), 1.0) if dependency_items else 0.0
    else:
        dependency_preserved = max(
            interface_binding,
            value_transfer,
            min(overlap_prompt / 0.25, 1.0),
        )
    entity_overlap_score = max(
        entity_overlap_ratio,
        min(overlap_all / 0.30, 1.0),
        0.85 * value_transfer,
    )
    if "extract" in source_roles or "evidence" in source_roles:
        semantic_transfer = _clamp01(0.55 * semantic_transfer + 0.45 * entity_overlap_score)
    if "synthesize" in target_roles or "answer" in target_roles:
        semantic_transfer = _clamp01(0.45 * semantic_transfer + 0.55 * max(entity_overlap_score, min(overlap_all / 0.18, 1.0)))

    return {
        "edge_consumed": _clamp01(edge_consumed),
        "entity_overlap": _clamp01(entity_overlap_score),
        "semantic_transfer": _clamp01(semantic_transfer),
        "dependency_preserved": _clamp01(dependency_preserved),
    }


def _score_structure_observation(
    workflow_graph: Any,
    seen_nodes_in_order: List[str],
) -> Dict[str, float]:
    role_meta = _workflow_role_coverage(workflow_graph)
    edges = [
        (getattr(edge, "source", None), getattr(edge, "target", None))
        for edge in (getattr(workflow_graph, "edges", []) or [])
        if getattr(edge, "source", None) and getattr(edge, "target", None)
    ]
    pos = {name: idx for idx, name in enumerate(seen_nodes_in_order)}
    if seen_nodes_in_order:
        pos = {}
        for idx, name in enumerate(seen_nodes_in_order):
            if name not in pos:
                pos[name] = idx
    ordering_ok = 1
    for source_name, target_name in edges:
        if source_name in pos and target_name in pos and pos[source_name] > pos[target_name]:
            ordering_ok = 0
            break
    consumed_intermediate_outputs = _count_consumed_intermediate_outputs(workflow_graph)
    role_positions: Dict[str, int] = {}
    for node in getattr(workflow_graph, "nodes", []) or []:
        roles = _infer_node_roles(node, [getattr(out, "name", "") for out in getattr(node, "outputs", []) or []])
        node_rank = min(_ROLE_ORDER.get(role, 5) for role in roles) if roles else 5
        role_positions[getattr(node, "name", "")] = node_rank
    for source_name, target_name in edges:
        if source_name in role_positions and target_name in role_positions:
            if role_positions[source_name] > role_positions[target_name]:
                ordering_ok = 0
                break
    node_count = len(getattr(workflow_graph, "nodes", []) or [])
    edge_count = len(edges)
    role_coverage = min(role_meta["count"] / 3.0, 1.0) * min(consumed_intermediate_outputs / max(1.0, float(max(edge_count - 1, 1))), 1.0)
    if node_count < 3:
        role_coverage *= 0.5
    topology_ordering = (0.65 if ordering_ok else 0.0) + 0.35 * min(consumed_intermediate_outputs / max(1.0, float(max(edge_count, 1))), 1.0)
    return {
        "role_coverage": _clamp01(role_coverage),
        "topology_ordering": _clamp01(topology_ordering),
    }


def _count_consumed_intermediate_outputs(workflow_graph: Any) -> int:
    count = 0
    node_map = {
        getattr(node, "name", ""): node
        for node in (getattr(workflow_graph, "nodes", []) or [])
        if getattr(node, "name", "")
    }
    for edge in (getattr(workflow_graph, "edges", []) or []):
        source_name = getattr(edge, "source", None)
        target_name = getattr(edge, "target", None)
        if not source_name or not target_name:
            continue
        source_node = node_map.get(source_name)
        target_node = node_map.get(target_name)
        if source_node is None or target_node is None:
            continue
        source_outputs = {
            str(getattr(param, "name", "")).lower()
            for param in (getattr(source_node, "outputs", []) or [])
            if str(getattr(param, "name", "")).strip()
        }
        target_inputs = {
            str(getattr(param, "name", "")).lower()
            for param in (getattr(target_node, "inputs", []) or [])
            if str(getattr(param, "name", "")).strip()
        }
        count += len({name for name in (source_outputs & target_inputs) if name and name != "answer"})
    return count


def _migrate_chain_runtime_evidence(
    *,
    workflow_graph: Any,
    node_map: Dict[str, Any],
    node_outputs: Dict[str, Dict[str, Any]],
    return_obs: Dict[str, Dict[str, float]],
    edge_obs: Dict[str, Dict[str, float]],
    structure_obs: Dict[str, float],
):
    outgoing_edges: Dict[str, List[Tuple[str, str]]] = {}
    for edge in (getattr(workflow_graph, "edges", []) or []):
        source_name = getattr(edge, "source", None)
        target_name = getattr(edge, "target", None)
        if not source_name or not target_name:
            continue
        if source_name not in node_outputs or target_name not in node_map:
            continue
        outgoing_edges.setdefault(source_name, []).append((source_name, target_name))

    for node_name, outputs in node_outputs.items():
        node = node_map.get(node_name)
        if node is None:
            continue
        roles = _infer_node_roles(node, list(outputs.keys()))
        node_return_obs = return_obs.setdefault(node_name, {})
        attached_edges = [_edge_key(src, tgt) for src, tgt in outgoing_edges.get(node_name, [])]

        if "decompose" in roles:
            question_coverage = node_return_obs.pop("question_coverage", None)
            dependency_order = node_return_obs.pop("dependency_order", None)
            if dependency_order is not None:
                structure_obs["topology_ordering"] = min(
                    _safe_rate(structure_obs.get("topology_ordering", 1.0), 1.0),
                    _safe_rate(dependency_order),
                )
            for edge_key in attached_edges:
                edge_entry = edge_obs.setdefault(edge_key, {})
                if question_coverage is not None:
                    edge_entry["edge_consumed"] = max(_safe_rate(edge_entry.get("edge_consumed", 0.0)), _safe_rate(question_coverage))
                if dependency_order is not None:
                    edge_entry["dependency_preserved"] = max(_safe_rate(edge_entry.get("dependency_preserved", 0.0)), _safe_rate(dependency_order))

        if "extract" in roles or "evidence" in roles:
            evidence_alignment = node_return_obs.pop("evidence_alignment", None)
            entity_fidelity = node_return_obs.pop("entity_fidelity", None)
            for edge_key in attached_edges:
                edge_entry = edge_obs.setdefault(edge_key, {})
                if evidence_alignment is not None:
                    edge_entry["semantic_transfer"] = max(_safe_rate(edge_entry.get("semantic_transfer", 0.0)), _safe_rate(evidence_alignment))
                if entity_fidelity is not None:
                    edge_entry["entity_overlap"] = max(_safe_rate(edge_entry.get("entity_overlap", 0.0)), _safe_rate(entity_fidelity))

        if "synthesize" in roles:
            bridge_consistency = node_return_obs.pop("bridge_consistency", None)
            chain_completeness = node_return_obs.pop("chain_completeness", None)
            if bridge_consistency is not None:
                structure_obs["topology_ordering"] = min(_safe_rate(structure_obs.get("topology_ordering", 1.0), 1.0), _safe_rate(bridge_consistency))
            for edge_key in attached_edges:
                edge_entry = edge_obs.setdefault(edge_key, {})
                if bridge_consistency is not None:
                    edge_entry["entity_overlap"] = max(_safe_rate(edge_entry.get("entity_overlap", 0.0)), _safe_rate(bridge_consistency))
                    edge_entry["semantic_transfer"] = max(_safe_rate(edge_entry.get("semantic_transfer", 0.0)), _safe_rate(bridge_consistency))
                if chain_completeness is not None:
                    edge_entry["dependency_preserved"] = max(_safe_rate(edge_entry.get("dependency_preserved", 0.0)), _safe_rate(chain_completeness))


def _score_task_specific_return_dims(
    node: Any,
    outputs: Dict[str, Any],
    is_final_node: bool,
    label: Optional[str],
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    del label
    roles = _infer_node_roles(node, list(outputs.keys()))
    texts = _flatten_output_texts(outputs)
    items = []
    for value in (outputs or {}).values():
        items.extend(_split_items(value))
    unique_items = [item for item in dict.fromkeys(items) if item]
    joined = " ".join(texts).lower()
    output_names = {name.lower() for name in outputs.keys()}
    result: Dict[str, float] = {}

    if "decompose" in roles:
        if len(unique_items) < 2:
            question_coverage = 0.0
        elif len(unique_items) == 2:
            question_coverage = 0.60
        elif 3 <= len(unique_items) <= 5:
            question_coverage = 0.85
        elif len(unique_items) == 6:
            question_coverage = 0.50
        else:
            question_coverage = 0.20
        dependency_order = 0.0 if len(unique_items) < 2 else len(set(unique_items)) / max(1.0, float(len(unique_items)))
        result["question_coverage"] = _clamp01(question_coverage)
        result["dependency_order"] = _clamp01(dependency_order)

    if "extract" in roles or "evidence" in roles:
        evidence_like = [text for text in texts if len(text.split()) >= 4]
        titleish = sum(1 for text in texts if any(ch.isupper() for ch in text[: min(24, len(text))]))
        result["evidence_alignment"] = _clamp01(min(len(evidence_like) / 2.0, 1.0))
        result["entity_fidelity"] = _clamp01(max(0.60 if titleish >= 1 else 0.0, 0.50 if any(name in output_names for name in ("entities", "facts", "evidence")) else 0.0))

    if "synthesize" in roles:
        connective_hits = sum(1 for cue in ("because", "therefore", "thus", "first", "then", "finally") if cue in joined)
        sufficient_text = any(8 <= len(text.split()) <= 120 for text in texts)
        bridge_consistency = min(connective_hits / 2.0, 1.0) * (1.0 if sufficient_text else 0.30)
        chain_completeness = max(bridge_consistency, min(len(unique_items) / 3.0, 1.0) * (1.0 if sufficient_text else 0.20))
        result["bridge_consistency"] = _clamp01(bridge_consistency)
        result["chain_completeness"] = _clamp01(chain_completeness)

    if "answer" in roles or is_final_node:
        answer = outputs.get("answer")
        if answer is None and outputs:
            answer = next(iter(outputs.values()))
        answer_text = _obs_norm_text(answer)
        em = _safe_rate((metrics or {}).get("em", 0.0))
        f1 = _safe_rate((metrics or {}).get("f1", 0.0))
        token_len = len(answer_text.split())
        # Detect compact LaTeX answer formats (e.g. \boxed{135}, ^2$)
        # that legitimately contain braces/$ but are well-normalized.
        _ans_lower = answer_text.lower()
        _has_latex_wrapper = bool(re.search(r"\\boxed\s*\{.+\}\s*$", answer_text.strip()))  # BOXED_REGEX_FIX_V1: was r"\boxed" (word boundary) -> r"\\boxed" (literal backslash)
        _has_latex_math = (
            _has_latex_wrapper
            or bool(re.match(r"^\$[^$]+\$$", answer_text.strip()))
        )
        _verbose_markers = ("because", "therefore", "the answer is", "```")
        _brace_markers = ("{", "}")
        normalized = (
            token_len > 0
            and (token_len <= 8 or _has_latex_math)
            and not any(m in _ans_lower for m in _verbose_markers)
            and (_has_latex_math or not any(m in _ans_lower for m in _brace_markers))
        )
        result["answer_supported"] = _clamp01(max(em, min(f1 / 0.8, 1.0), 0.45 if (answer_text and token_len <= 8) else 0.0))
        result["answer_normalized"] = 1.0 if normalized else 0.15

    return result


def _score_return_task_ok(
    node: Any,
    outputs: Dict[str, Any],
    msg: Any,
    is_final_node: bool,
    label: Optional[str],
    metrics: Optional[Dict[str, Any]],
) -> float:
    output_names = list(outputs.keys())
    role = _infer_node_role(node, output_names)

    if role == "query_generation":
        key = "query" if "query" in outputs else output_names[0]
        query = _obs_norm_text(outputs.get(key))
        token_len = len(query.split())
        if _looks_like_refusal(query):
            return 0.0
        if 3 <= token_len <= 40:
            return 1.0
        if 2 <= token_len <= 60:
            return 0.55
        return 0.10

    if role == "relation_extraction":
        entities = _split_items(outputs.get("key_entities") if "key_entities" in outputs else outputs.get("entities"))
        relations = _split_items(
            outputs.get("relationships") if "relationships" in outputs else outputs.get("entity_relations")
        )
        if relations:
            return 1.0 if (len(entities) >= 2 and len(relations) >= 1) else (0.45 if entities or relations else 0.0)
        return 1.0 if len(entities) >= 2 else (0.35 if len(entities) == 1 else 0.0)

    if role == "retrieval":
        retrieval_key = None
        for key in ("retrieved_data", "facts", "evidence", "passages", "context"):
            if key in outputs:
                retrieval_key = key
                break
        if retrieval_key is None:
            # Fallback: some workflows use custom field names for retrieved evidence.
            fallback_texts = [_obs_norm_text(v) for v in outputs.values()]
            return _clamp01(sum(1.0 for t in fallback_texts if len(t) >= 20 and not _looks_like_refusal(t)) / max(1.0, float(len(fallback_texts) or 1)))
        value = outputs.get(retrieval_key)
        if isinstance(value, dict):
            evidence_texts = [_obs_norm_text(v) for v in value.values()]
            return _clamp01(sum(1.0 for t in evidence_texts if len(t) >= 20 and not _looks_like_refusal(t)) / max(1.0, float(len(evidence_texts) or 1)))
        if isinstance(value, list):
            evidence_texts = [_obs_norm_text(v) for v in value]
            return _clamp01(sum(1.0 for t in evidence_texts if len(t) >= 20 and not _looks_like_refusal(t)) / max(1.0, float(len(evidence_texts) or 1)))
        text = _obs_norm_text(value)
        return 1.0 if (len(text) >= 20 and not _looks_like_refusal(text)) else (0.20 if text else 0.0)

    if role == "final_answer" or is_final_node:
        answer = outputs.get("answer")
        if answer is None and output_names:
            answer = outputs.get(output_names[0])
        return _score_answer_task_ok(answer=answer, label=label, metrics=metrics)

    all_values = [outputs.get(name) for name in output_names]
    if not all_values:
        return 0.0
    filled = sum(_nonempty_value_ok(v) for v in all_values) / max(1.0, float(len(all_values)))
    if _looks_like_refusal(getattr(msg, "content", "")):
        filled *= 0.15
    flat_texts = _flatten_output_texts(outputs)
    avg_len = (
        sum(len(text.split()) for text in flat_texts) / max(1.0, float(len(flat_texts)))
        if flat_texts
        else 0.0
    )
    keyword_diversity = _jaccard(_keyword_set(" ".join(flat_texts)), _keyword_set(_obs_norm_text(getattr(msg, "content", ""))))
    richness = min(avg_len / 16.0, 1.0)
    return _clamp01(0.45 * filled + 0.35 * richness + 0.20 * keyword_diversity)


def _score_return_observation(
    node: Any,
    msg: Any,
    outputs: Dict[str, Any],
    is_final_node: bool,
    label: Optional[str],
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    type_ok = 1.0
    content_ok = 1.0
    for param in getattr(node, "outputs", []):
        value = outputs.get(param.name)
        type_ok *= 1.0 if _expected_type_ok(getattr(param, "type", ""), value) else 0.0
        content_ok *= 1.0 if _nonempty_value_ok(value) else 0.0

    if _looks_like_refusal(getattr(msg, "content", "")):
        content_ok *= 0.15

    task_ok = _score_return_task_ok(
        node=node,
        outputs=outputs,
        msg=msg,
        is_final_node=is_final_node,
        label=label,
        metrics=metrics,
    )
    result = {
        "type_ok": _clamp01(type_ok),
        "content_ok": _clamp01(content_ok),
        "task_ok": _clamp01(task_ok),
    }
    output_names = [getattr(param, "name", "") for param in getattr(node, "outputs", []) or []]
    if is_final_node or "answer" in {name.lower() for name in output_names}:
        em = float((metrics or {}).get("em", 0.0) or 0.0)
        f1 = float((metrics or {}).get("f1", 0.0) or 0.0)
        result["exact_ok"] = 1.0 if em >= 1.0 else 0.0
        result["overlap_ok"] = _clamp01(f1 / 0.8)
    result.update(
        _score_task_specific_return_dims(
            node=node,
            outputs=outputs,
            is_final_node=is_final_node,
            label=label,
            metrics=metrics,
        )
    )
    return result


def extract_local_obs_from_trajectory(
    workflow_graph: Any,
    trajectory: Any,
    label: Optional[str] = None,
    prediction: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, float],
]:
    del prediction  # Reserved for future use.

    node_map = {node.name: node for node in getattr(workflow_graph, "nodes", [])}
    final_nodes = set()
    if hasattr(workflow_graph, "find_end_nodes"):
        try:
            final_nodes = set(workflow_graph.find_end_nodes() or [])
        except Exception:
            final_nodes = set()

    prompt_obs: Dict[str, Dict[str, float]] = {}
    return_obs: Dict[str, Dict[str, float]] = {}
    params_obs: Dict[str, Dict[str, float]] = {}
    edge_obs: Dict[str, Dict[str, float]] = {}
    structure_obs: Dict[str, float] = {}

    if not isinstance(trajectory, list):
        return prompt_obs, return_obs, params_obs, edge_obs, structure_obs

    node_outputs: Dict[str, Dict[str, Any]] = {}
    node_prompts: Dict[str, str] = {}
    seen_nodes_in_order: List[str] = []

    for msg in trajectory:
        if not _is_response_message(msg):
            continue

        node_name = getattr(msg, "wf_task", None)
        if node_name not in node_map:
            continue

        node = node_map[node_name]
        prompt_text = _flatten_prompt(getattr(msg, "prompt", ""))
        template_text = _extract_prompt_template_from_node(node)
        outputs = _get_structured_outputs_from_message(msg, node)

        prompt_obs[node_name] = _score_prompt_observation(
            node=node,
            prompt_text=prompt_text,
            template_text=template_text,
        )
        return_obs[node_name] = _score_return_observation(
            node=node,
            msg=msg,
            outputs=outputs,
            is_final_node=node_name in final_nodes,
            label=label,
            metrics=metrics,
        )
        params_obs[node_name] = _score_params_observation(
            node=node,
            msg=msg,
            outputs=outputs,
            prompt_text=prompt_text,
        )
        node_outputs[node_name] = outputs
        node_prompts[node_name] = prompt_text
        seen_nodes_in_order.append(node_name)

    for edge in (getattr(workflow_graph, "edges", []) or []):
        source_name = getattr(edge, "source", None)
        target_name = getattr(edge, "target", None)
        if not source_name or not target_name:
            continue
        if source_name not in node_outputs or target_name not in node_prompts:
            continue
        source_node = node_map.get(source_name)
        target_node = node_map.get(target_name)
        if source_node is None or target_node is None:
            continue
        edge_obs[_edge_key(source_name, target_name)] = _score_edge_observation(
            source_node=source_node,
            target_node=target_node,
            source_outputs=node_outputs.get(source_name, {}),
            target_prompt_text=node_prompts.get(target_name, ""),
            target_outputs=node_outputs.get(target_name, {}),
        )

    structure_obs = _score_structure_observation(
        workflow_graph=workflow_graph,
        seen_nodes_in_order=seen_nodes_in_order,
    )
    _migrate_chain_runtime_evidence(
        workflow_graph=workflow_graph,
        node_map=node_map,
        node_outputs=node_outputs,
        return_obs=return_obs,
        edge_obs=edge_obs,
        structure_obs=structure_obs,
    )

    return prompt_obs, return_obs, params_obs, edge_obs, structure_obs


def extract_judge_payloads_from_trajectory(
    workflow_graph: Any,
    trajectory: Any,
) -> Dict[str, Dict[str, Any]]:
    node_map = {node.name: node for node in getattr(workflow_graph, "nodes", [])}
    final_nodes = set()
    if hasattr(workflow_graph, "find_end_nodes"):
        try:
            final_nodes = set(workflow_graph.find_end_nodes() or [])
        except Exception:
            final_nodes = set()

    payloads: Dict[str, Dict[str, Any]] = {}
    if not isinstance(trajectory, list):
        return payloads

    for msg in trajectory:
        if not _is_response_message(msg):
            continue
        node_name = getattr(msg, "wf_task", None)
        if node_name not in node_map:
            continue
        node = node_map[node_name]
        outputs = _get_structured_outputs_from_message(msg, node)
        payloads[node_name] = {
            "prompt_text": _flatten_prompt(getattr(msg, "prompt", "")),
            "outputs": outputs,
            "roles": _infer_node_roles(node, list(outputs.keys())),
            "input_names": [getattr(inp, "name", "") for inp in getattr(node, "inputs", []) or []],
            "output_names": [getattr(out, "name", "") for out in getattr(node, "outputs", []) or []],
            "is_final_node": node_name in final_nodes,
        }
    return payloads
