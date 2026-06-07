import numpy as np
from typing import Dict, List, Tuple

from .factors import GraphFactor
from .schema import GraphVariable, GraphEdge


def _safe_normalize(msg: np.ndarray) -> np.ndarray:
    """
    仅做最基本的归一化，不做人为“抬平”。
    这样可以避免强证据在高连接度节点上被洗回 [0.5, 0.5]。
    """
    msg = np.asarray(msg, dtype=float)

    if msg.shape != (2,) or not np.all(np.isfinite(msg)):
        return np.array([0.5, 0.5], dtype=float)

    msg = np.clip(msg, 0.0, None)
    s = float(np.sum(msg))
    if s <= 0.0:
        return np.array([0.5, 0.5], dtype=float)

    return msg / s


def _soft_observation(val: float, eps: float = 1e-3) -> np.ndarray:
    """
    Observation to message conversion:
    - if val in {0,1}: near-hard evidence with epsilon smoothing
    - if val in (0,1): soft evidence interpreted as P(True)=val
    """
    try:
        p_true = float(val)
    except Exception:
        return np.array([0.5, 0.5], dtype=float)

    if p_true <= 0.0:
        return np.array([1.0 - eps, eps], dtype=float)
    if p_true >= 1.0:
        return np.array([eps, 1.0 - eps], dtype=float)
    return np.array([1.0 - p_true, p_true], dtype=float)


def _msg_to_log(msg: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    """
    把概率消息转成 log 概率。
    只在取 log 前做极小下界裁剪，防止 log(0)。
    """
    msg = _safe_normalize(msg)
    msg = np.clip(msg, floor, 1.0)
    return np.log(msg)


def _normalize_from_log(log_msg: np.ndarray) -> np.ndarray:
    """
    从 log 概率恢复成普通概率，并做稳定归一化。
    """
    log_msg = np.asarray(log_msg, dtype=float)

    if log_msg.shape != (2,) or not np.all(np.isfinite(log_msg)):
        return np.array([0.5, 0.5], dtype=float)

    m = float(np.max(log_msg))
    probs = np.exp(log_msg - m)
    return _safe_normalize(probs)


def _apply_damping(old_msg: np.ndarray, new_msg: np.ndarray, damping: float) -> np.ndarray:
    """
    damping = 0.0 表示不阻尼
    damping = 0.5 表示新旧各占一半
    """
    damping = float(damping)
    if damping <= 0.0:
        return _safe_normalize(new_msg)
    if damping >= 1.0:
        return _safe_normalize(old_msg)

    mixed = damping * np.asarray(old_msg, dtype=float) + (1.0 - damping) * np.asarray(new_msg, dtype=float)
    return _safe_normalize(mixed)


class FactorGraphEngine:
    def __init__(self):
        self.variables: Dict[str, GraphVariable] = {}
        self.factors: Dict[str, GraphFactor] = {}
        self.edges: List[GraphEdge] = []

        # 辅助索引：var_name -> [connected_edges]
        self.var_to_edges: Dict[str, List[GraphEdge]] = {}
        # 辅助索引：factor_name -> [connected_edges]
        self.factor_to_edges: Dict[str, List[GraphEdge]] = {}

    def add_variable(self, var: GraphVariable):
        self.variables[var.name] = var
        self.var_to_edges[var.name] = []

    def add_factor(self, factor: GraphFactor):
        self.factors[factor.name] = factor
        self.factor_to_edges[factor.name] = []

        # 自动创建边
        for var_name in factor.connected_vars:
            edge_id = f"{var_name}-{factor.name}"
            edge = GraphEdge(id=edge_id, variable_name=var_name, factor_name=factor.name)
            self.edges.append(edge)

            if var_name in self.var_to_edges:
                self.var_to_edges[var_name].append(edge)
            if factor.name in self.factor_to_edges:
                self.factor_to_edges[factor.name].append(edge)

    def _reset_inference_state(self):
        """
        每次运行 LBP 前重置边消息与变量 belief，避免旧状态残留。
        """
        for edge in self.edges:
            edge.reset()

        for var in self.variables.values():
            if var.is_observed:
                var._belief = _soft_observation(var.observed_value)
            else:
                var._belief = var._prior.copy()

    def run_loopy_belief_propagation(
        self,
        max_iter: int = 50,
        tolerance: float = 1e-4,
        damping: float = 0.2,
        patience: int = 15,
        verbose: bool = True,
    ):
        """
        Run loopy belief propagation (LBP).

        Key details:
        1. Use log-space aggregation on the variable side (v->f / belief).
        2. Keep factor-side updates in probability space for stability.
        3. Support damping to reduce oscillation on loopy graphs.
        """
        self._reset_inference_state()

        if verbose:
            print(f"--- Starting LBP (max_iter={max_iter}, damping={damping}) ---")

        best_delta = float("inf")
        no_improve_rounds = 0

        for iteration in range(max_iter):
            max_delta = 0.0

            # ============================================================
            # Step 1: Variable -> Factor
            # Use log-space here to avoid underflow on high-degree health nodes.
            # ============================================================
            for var_name, var in self.variables.items():
                connected_edges = self.var_to_edges.get(var_name, [])

                if not connected_edges:
                    # Isolated variable.
                    if var.is_observed:
                        var._belief = _soft_observation(var.observed_value)
                    else:
                        var._belief = _safe_normalize(var._prior.copy())
                    continue

                # Observed variables send fixed observation messages.
                if var.is_observed:
                    fixed_msg = _soft_observation(var.observed_value)
                    for edge in connected_edges:
                        updated = fixed_msg.copy()
                        delta = float(np.max(np.abs(edge._msg_v2f - updated)))
                        max_delta = max(max_delta, delta)
                        edge._msg_v2f = updated
                    continue

                # Unobserved variable: msg(v->f) is prior times incoming messages.
                prior_log = _msg_to_log(var._prior)

                edge_logs = {}
                total_log = prior_log.copy()
                for edge in connected_edges:
                    log_in = _msg_to_log(edge._msg_f2v)
                    edge_logs[edge.id] = log_in
                    total_log += log_in

                for target_edge in connected_edges:
                    # Exclude the target factor's reverse message.
                    out_log = total_log - edge_logs[target_edge.id]
                    new_msg = _normalize_from_log(out_log)
                    new_msg = _apply_damping(target_edge._msg_v2f, new_msg, damping)

                    delta = float(np.max(np.abs(target_edge._msg_v2f - new_msg)))
                    max_delta = max(max_delta, delta)
                    target_edge._msg_v2f = new_msg

            # ============================================================
            # Step 2: Factor -> Variable
            # Factor degree is usually small, so probability space is fine.
            # ============================================================
            for factor_name, factor in self.factors.items():
                connected_edges = self.factor_to_edges.get(factor_name, [])
                if not connected_edges:
                    continue

                incoming_msgs = {
                    edge.variable_name: edge._msg_v2f
                    for edge in connected_edges
                }

                for target_edge in connected_edges:
                    target_var = target_edge.variable_name
                    raw_msg = factor.compute_message_to_variable(target_var, incoming_msgs)
                    new_msg = _safe_normalize(raw_msg)
                    new_msg = _apply_damping(target_edge._msg_f2v, new_msg, damping)

                    delta = float(np.max(np.abs(target_edge._msg_f2v - new_msg)))
                    max_delta = max(max_delta, delta)
                    target_edge._msg_f2v = new_msg

            # ============================================================
            # Step 3: Update marginals
            # Again use log-space to avoid washing out beliefs.
            # ============================================================
            for var_name, var in self.variables.items():
                if var.is_observed:
                    var._belief = _soft_observation(var.observed_value)
                    continue

                connected_edges = self.var_to_edges.get(var_name, [])
                if not connected_edges:
                    var._belief = _safe_normalize(var._prior.copy())
                    continue

                belief_log = _msg_to_log(var._prior)
                for edge in connected_edges:
                    belief_log += _msg_to_log(edge._msg_f2v)

                var._belief = _normalize_from_log(belief_log)

            if verbose:
                print(f"Iter {iteration + 1}: Max Delta = {max_delta:.6f}")

            if max_delta < tolerance:
                if verbose:
                    print("--- Converged ---")
                break

            if max_delta < best_delta - 1e-12:
                best_delta = max_delta
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= patience:
                if verbose:
                    print(f"--- Early stop: no improvement for {patience} rounds (best={best_delta:.6f}) ---")
                break

    def get_root_causes(self) -> List[Tuple[str, float]]:
        """
        获取所有健康节点的故障概率，并按降序排序。
        返回: [(HealthNodeName, P(Failure)), ...]
        """
        results = []
        for name, var in self.variables.items():
            if var.is_health_node:
                prob_failure = float(var._belief[0])  # P(False / Failure)
                results.append((name, prob_failure))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
