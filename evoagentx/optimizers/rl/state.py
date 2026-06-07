import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class NodeStateSummary:
    node_name: str
    prompt_quality: float = 0.0
    params_quality: float = 0.0
    return_quality: float = 0.0
    best_prompt_f1: float = 0.0
    best_prompt_em: float = 0.0
    prompt_success_rate: float = 0.0
    params_success_rate: float = 0.0
    fail_streak: float = 0.0
    subtype_failure_probs: Dict[str, float] = field(default_factory=dict)
    subtype_success_rates: Dict[str, float] = field(default_factory=dict)
    subtype_mean_rewards: Dict[str, float] = field(default_factory=dict)
    style_success_rates: Dict[str, float] = field(default_factory=dict)
    style_mean_rewards: Dict[str, float] = field(default_factory=dict)
    op_success_rates: Dict[str, float] = field(default_factory=dict)
    op_mean_rewards: Dict[str, float] = field(default_factory=dict)
    observation_variance: float = 0.0


@dataclass
class RLStateSummary:
    step_index: int
    max_steps: int
    baseline_f1: float
    baseline_em: float
    baseline_acc: float
    obs_coverage: float
    node_count: int
    edge_count: int
    dag_depth: int
    role_count: int
    budget_ratio: float
    no_improve_ratio: float
    top_actionable_failure_prob: float
    rca_entropy: float
    weak_rca: bool
    last_action_f1_delta: float = 0.0
    rca_delta_from_init: float = 0.0
    structure_observation_variance: float = 0.0
    edge_observation_variance: float = 0.0
    top_rca_targets: List[Tuple[str, str, str, float]] = field(default_factory=list)
    nodes: Dict[str, NodeStateSummary] = field(default_factory=dict)


def top_actionable_failure_prob(root_causes: List[Tuple[str, float]]) -> float:
    best = 0.0
    for health_name, failure_prob in root_causes or []:
        if not isinstance(health_name, str):
            continue
        if health_name.startswith("HealthReturn"):
            continue
        best = max(best, _safe_rate(failure_prob, 0.0))
    return best


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
        if isinstance(name, str) and name.startswith(prefix):
            suffix = name[len(prefix) :]
            if component == "Edge" and "__TO__" in suffix:
                return component, subtype, suffix.split("__TO__", 1)[1]
            return component, subtype, suffix
    return None


def _rca_entropy(root_causes: List[Tuple[str, float]]) -> float:
    probs = [_safe_rate(p, 0.0) for _, p in (root_causes or [])]
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in probs]
    return -sum(p * math.log(max(p, 1e-12)) for p in normalized)


def _workflow_depth(workflow_graph: Any) -> int:
    nodes = [getattr(node, "name", "") for node in (getattr(workflow_graph, "nodes", []) or []) if getattr(node, "name", "")]
    if not nodes:
        return 0
    incoming: Dict[str, int] = {name: 0 for name in nodes}
    outgoing: Dict[str, List[str]] = {name: [] for name in nodes}
    for edge in (getattr(workflow_graph, "edges", []) or []):
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source not in outgoing or target not in incoming:
            continue
        outgoing[source].append(target)
        incoming[target] += 1
    queue = [name for name, deg in incoming.items() if deg == 0]
    if not queue:
        return max(1, len(nodes))
    depth: Dict[str, int] = {name: 1 for name in queue}
    idx = 0
    seen = 0
    while idx < len(queue):
        current = queue[idx]
        idx += 1
        seen += 1
        for nxt in outgoing.get(current, []):
            depth[nxt] = max(depth.get(nxt, 1), depth.get(current, 1) + 1)
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)
    if seen < len(nodes):
        return max(max(depth.values(), default=1), len(nodes))
    return max(depth.values(), default=1)


def _variance(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def _node_observation_variances(evidences: Dict[str, Any]) -> Dict[str, float]:
    per_node: Dict[str, List[float]] = {}
    for ev in (evidences or {}).values():
        per_sample: Dict[str, List[float]] = {}
        for attr_name in ("prompt_obs", "params_obs", "return_obs"):
            section = getattr(ev, attr_name, {}) or {}
            for node_name, dims in section.items():
                for dim_value in (dims or {}).values():
                    per_sample.setdefault(str(node_name), []).append(_safe_rate(dim_value, 0.0))
        for node_name, values in per_sample.items():
            per_node.setdefault(node_name, []).append(sum(values) / max(1.0, float(len(values))))
    return {node_name: _variance(values) for node_name, values in per_node.items()}


def _nested_section_variance(evidences: Dict[str, Any], attr_name: str) -> float:
    sample_values: List[float] = []
    for ev in (evidences or {}).values():
        section = getattr(ev, attr_name, {}) or {}
        values: List[float] = []
        if attr_name == "structure_obs":
            values = [_safe_rate(value, 0.0) for value in section.values()]
        else:
            for dims in section.values():
                values.extend(_safe_rate(value, 0.0) for value in (dims or {}).values())
        if values:
            sample_values.append(sum(values) / max(1.0, float(len(values))))
    return _variance(sample_values)


def build_workflow_state(
    *,
    workflow_graph: Any,
    base_pkg: Any,
    prompt_history: Any,
    action_history: Any,
    node_fail_streak: Dict[Tuple[str, str], int],
    iter_idx: int,
    max_opt_iterations: int,
    no_improve_count: int,
    strong_rca_threshold: float,
    role_meta: Optional[Dict[str, Any]] = None,
    last_action_f1_delta: float = 0.0,
    init_top_actionable_failure_prob: float = 0.0,
) -> RLStateSummary:
    role_meta = role_meta or {}
    node_stats = getattr(base_pkg, "node_stats", {}) or {}
    root_causes = getattr(base_pkg, "root_causes", []) or []
    evidences = getattr(base_pkg, "evidences", {}) or {}
    node_obs_variances = _node_observation_variances(evidences)
    edge_obs_variance = _nested_section_variance(evidences, "edge_obs")
    structure_obs_variance = _nested_section_variance(evidences, "structure_obs")
    node_subtype_failure_probs: Dict[str, Dict[str, float]] = {}
    top_targets: List[Tuple[str, str, str, float]] = []
    for health_name, failure_prob in root_causes:
        parsed = _parse_health_node_name(health_name)
        if not parsed:
            continue
        component, subtype, node_name = parsed
        node_subtype_failure_probs.setdefault(node_name, {})
        node_subtype_failure_probs[node_name][f"{component}:{subtype}"] = max(
            _safe_rate(failure_prob, 0.0),
            node_subtype_failure_probs[node_name].get(f"{component}:{subtype}", 0.0),
        )
        if component != "Return":
            top_targets.append((component, subtype, node_name, _safe_rate(failure_prob, 0.0)))
    top_targets.sort(key=lambda x: x[3], reverse=True)
    nodes: Dict[str, NodeStateSummary] = {}
    for node in getattr(workflow_graph, "nodes", []):
        prompt_stats = (node_stats.get(node.name) or {}).get("prompt", {}) or {}
        params_stats = (node_stats.get(node.name) or {}).get("params", {}) or {}
        return_stats = (node_stats.get(node.name) or {}).get("return", {}) or {}
        prompt_rec = prompt_history.get_best_record(node.name) if prompt_history is not None else None
        style_success_rates: Dict[str, float] = {}
        style_mean_rewards: Dict[str, float] = {}
        op_success_rates: Dict[str, float] = {}
        op_mean_rewards: Dict[str, float] = {}
        subtype_success_rates: Dict[str, float] = {}
        subtype_mean_rewards: Dict[str, float] = {}
        if action_history is not None:
            subtype_keys = set(node_subtype_failure_probs.get(node.name, {}).keys())
            subtype_keys.update(
                {
                    "Prompt:Binding",
                    "Prompt:Contract",
                    "Prompt:Grounding",
                    "Params:Length",
                    "Params:Parse",
                    "Return:Type",
                    "Return:Evidence",
                    "Return:Task",
                    "Edge:Binding",
                    "Edge:Semantic",
                }
            )
            for label in ("prompt_explore", "params_explore"):
                for comp_subtype in subtype_keys:
                    component, subtype = comp_subtype.split(":", 1)
                    key = f"{label}:{component}:{subtype}"
                    subtype_success_rates[key] = action_history.node_subtype_success_rate(label, node.name, component, subtype)
                    subtype_mean_rewards[key] = action_history.node_subtype_mean_reward(label, node.name, component, subtype)
            for label in ("prompt_explore", "params_explore"):
                for style in (
                    "BINDING_REPAIR",
                    "SCHEMA_HARDEN",
                    "GROUNDING_HARDEN",
                    "CHAIN_SYNTHESIS",
                    "DEDUP_SIMPLIFY",
                    "ANSWER_NORMALIZE",
                    "MORE_TOKENS",
                    "LOWER_TEMPERATURE",
                    "STRICT_JSON",
                    "LOWER_TOP_P",
                    "BALANCED_SAMPLING",
                ):
                    key = f"{label}:{style}"
                    style_success_rates[key] = action_history.node_style_success_rate(label, node.name, style)
                    style_mean_rewards[key] = action_history.node_style_mean_reward(label, node.name, style)
            for op_family in ("ADD", "DELETE", "MODIFY"):
                key = f"prompt_explore:{op_family}"
                op_success_rates[key] = action_history.node_op_success_rate("prompt_explore", node.name, op_family)
                op_mean_rewards[key] = action_history.node_op_mean_reward("prompt_explore", node.name, op_family)
        nodes[node.name] = NodeStateSummary(
            node_name=node.name,
            prompt_quality=sum(_safe_rate(v, 0.0) for v in prompt_stats.values()) / max(1, len(prompt_stats)),
            params_quality=sum(_safe_rate(v, 0.0) for v in params_stats.values()) / max(1, len(params_stats)),
            return_quality=sum(_safe_rate(v, 0.0) for v in return_stats.values()) / max(1, len(return_stats)),
            best_prompt_f1=_safe_rate(getattr(prompt_rec, "metrics", {}).get("f1", 0.0) if prompt_rec else 0.0),
            best_prompt_em=_safe_rate(getattr(prompt_rec, "metrics", {}).get("em", 0.0) if prompt_rec else 0.0),
            prompt_success_rate=action_history.node_success_rate("prompt_explore", node.name) if action_history else 0.0,
            params_success_rate=action_history.node_success_rate("params_explore", node.name) if action_history else 0.0,
            fail_streak=float(
                max(
                    node_fail_streak.get(("Prompt", node.name), 0),
                    node_fail_streak.get(("Params", node.name), 0),
                )
            ),
            subtype_failure_probs=node_subtype_failure_probs.get(node.name, {}),
            subtype_success_rates=subtype_success_rates,
            subtype_mean_rewards=subtype_mean_rewards,
            style_success_rates=style_success_rates,
            style_mean_rewards=style_mean_rewards,
            op_success_rates=op_success_rates,
            op_mean_rewards=op_mean_rewards,
            observation_variance=node_obs_variances.get(node.name, 0.0),
        )
    max_steps = max(1, int(max_opt_iterations))
    top_prob = top_actionable_failure_prob(root_causes)
    return RLStateSummary(
        step_index=int(iter_idx),
        max_steps=max_steps,
        baseline_f1=_safe_rate(getattr(base_pkg, "results", {}).get("f1", 0.0)),
        baseline_em=_safe_rate(getattr(base_pkg, "results", {}).get("em", 0.0)),
        baseline_acc=_safe_rate(getattr(base_pkg, "results", {}).get("acc", 0.0)),
        obs_coverage=_safe_rate(getattr(base_pkg, "obs_coverage", 0.0)),
        node_count=len(getattr(workflow_graph, "nodes", []) or []),
        edge_count=len(getattr(workflow_graph, "edges", []) or []),
        dag_depth=_workflow_depth(workflow_graph),
        role_count=int(role_meta.get("count", 0)),
        budget_ratio=float(max(0.0, min(1.0, iter_idx / max_steps))),
        no_improve_ratio=float(max(0.0, min(1.0, no_improve_count / max_steps))),
        top_actionable_failure_prob=top_prob,
        rca_entropy=_rca_entropy(root_causes),
        weak_rca=bool(top_prob < strong_rca_threshold),
        last_action_f1_delta=float(last_action_f1_delta),
        rca_delta_from_init=float(top_prob - _safe_rate(init_top_actionable_failure_prob, 0.0)),
        structure_observation_variance=structure_obs_variance,
        edge_observation_variance=edge_obs_variance,
        top_rca_targets=top_targets[:5],
        nodes=nodes,
    )
