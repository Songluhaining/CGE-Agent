from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class RewardConfig:
    terminal_f1_weight: float = 0.7
    terminal_em_weight: float = 0.3
    lambda_complexity: float = 0.08
    complexity_node_weight: float = 0.25
    complexity_edge_weight: float = 0.20
    complexity_depth_weight: float = 0.55
    lambda_cost: float = 0.05
    invalid_penalty: float = 0.25
    acceptance_bonus: float = 0.0
    utility_accept_margin: float = 0.01


def _workflow_edges(workflow_graph: Any) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for edge in (getattr(workflow_graph, "edges", []) or []):
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source and target:
            edges.append((str(source), str(target)))
    return edges


def _workflow_depth(workflow_graph: Any) -> int:
    nodes = [getattr(node, "name", "") for node in (getattr(workflow_graph, "nodes", []) or []) if getattr(node, "name", "")]
    if not nodes:
        return 0
    edges = _workflow_edges(workflow_graph)
    incoming: Dict[str, int] = {name: 0 for name in nodes}
    outgoing: Dict[str, List[str]] = {name: [] for name in nodes}
    for source, target in edges:
        if source not in outgoing or target not in incoming:
            continue
        outgoing[source].append(target)
        incoming[target] += 1
    queue = [name for name, deg in incoming.items() if deg == 0]
    if not queue:
        return max(1, len(nodes))
    depth: Dict[str, int] = {name: 1 for name in queue}
    seen = 0
    idx = 0
    while idx < len(queue):
        current = queue[idx]
        idx += 1
        seen += 1
        current_depth = depth.get(current, 1)
        for nxt in outgoing.get(current, []):
            depth[nxt] = max(depth.get(nxt, 1), current_depth + 1)
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)
    if seen < len(nodes):
        return max(max(depth.values(), default=1), len(nodes))
    return max(depth.values(), default=1)


def workflow_complexity_metrics(workflow_graph: Any) -> Dict[str, float]:
    node_count = float(len(getattr(workflow_graph, "nodes", []) or []))
    edge_count = float(len(_workflow_edges(workflow_graph)))
    depth = float(_workflow_depth(workflow_graph))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "dag_depth": depth,
    }


def compute_workflow_complexity_score(
    workflow_graph: Any,
    config: RewardConfig = RewardConfig(),
) -> float:
    metrics = workflow_complexity_metrics(workflow_graph)
    node_term = metrics["node_count"] / 6.0
    edge_term = metrics["edge_count"] / 8.0
    depth_term = metrics["dag_depth"] / 5.0
    return (
        config.complexity_node_weight * node_term
        + config.complexity_edge_weight * edge_term
        + config.complexity_depth_weight * depth_term
    )


def compute_workflow_utility(
    *,
    workflow_graph: Any,
    results: Dict[str, Any],
    config: RewardConfig = RewardConfig(),
) -> float:
    f1 = _safe_rate((results or {}).get("f1", 0.0))
    em = _safe_rate((results or {}).get("em", 0.0))
    quality = config.terminal_f1_weight * f1 + config.terminal_em_weight * em
    complexity = compute_workflow_complexity_score(workflow_graph, config)
    return quality - config.lambda_complexity * complexity


def compute_policy_reward(
    *,
    base_pkg: Any,
    cand_pkg: Any,
    base_workflow: Any,
    cand_workflow: Any,
    accepted: bool,
    invalid: bool = False,
    config: RewardConfig = RewardConfig(),
) -> float:
    if invalid:
        return -float(config.invalid_penalty)

    base_utility = compute_workflow_utility(
        workflow_graph=base_workflow,
        results=getattr(base_pkg, "results", {}) or {},
        config=config,
    )
    cand_utility = compute_workflow_utility(
        workflow_graph=cand_workflow,
        results=getattr(cand_pkg, "results", {}) or {},
        config=config,
    )
    utility_gain = cand_utility - base_utility
    token_penalty = min(1.0, max(0.0, float(getattr(cand_pkg, "token_delta", 0)) / 50000.0))
    cost_penalty = min(1.0, max(0.0, float(getattr(cand_pkg, "cost_delta", 0.0)) / 5.0))
    reward = utility_gain - config.lambda_cost * (0.7 * token_penalty + 0.3 * cost_penalty)
    if accepted and config.acceptance_bonus:
        reward += config.acceptance_bonus
    return reward
