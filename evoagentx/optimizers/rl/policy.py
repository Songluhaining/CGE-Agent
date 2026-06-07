import math
import random
from typing import Any, Dict, List, Tuple

from .state import build_workflow_state


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class REINFORCEActionPolicy:
    """
    Lightweight online policy over discrete optimization actions.
    It consumes causal diagnostics through the structured state summary,
    then learns which action family / node target is most useful.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        forced_exploit_steps: int = 2,
        exploration_temperature: float = 0.85,
    ):
        self.learning_rate = learning_rate
        self.forced_exploit_steps = max(0, int(forced_exploit_steps))
        self.exploration_temperature = max(0.25, float(exploration_temperature))
        self.weights: Dict[str, Dict[str, float]] = {}

    def _warm_start_weights(self, label: str) -> Dict[str, float]:
        weights = {
            "bias": 0.0,
            "failure_prob": 1.8,
            "subtype_failure_prob": 0.8,
            "target_rank": -0.6,
            "target_pool_size": -0.05,
            "from_rca": 0.9,
            "rca_locked": 0.3,
            "style_success": 0.4,
            "style_reward": 0.3,
            "op_success": 0.3,
            "op_reward": 0.2,
            "label_success": 0.15,
            "action_type_success": 0.2,
            "node_success": 0.15,
            "observation_variance": 0.25,
            "fail_streak": -0.35,
            "no_improve_ratio": 0.1,
            "last_action_f1_delta": 0.15,
            "rca_delta_from_init": 0.2,
            "weak_rca": -0.05,
        }
        if label == "structure_explore":
            weights.update({"bias": 0.1, "failure_prob": 2.0, "from_rca": 1.0, "target_rank": -0.7})
        elif label == "prompt_explore":
            weights.update({"bias": 0.08, "op_success": 0.4, "op_reward": 0.3})
        elif label == "params_explore":
            weights.update({"bias": -0.05, "failure_prob": 1.2})
        elif label == "skip":
            weights.update(
                {
                    "bias": -0.45,
                    "failure_prob": -1.0,
                    "subtype_failure_prob": -0.5,
                    "from_rca": -0.8,
                    "target_rank": 0.1,
                    "no_improve_ratio": 0.4,
                    "weak_rca": 0.1,
                }
            )
        return weights

    def _ensure_label_weights(self, label: str) -> Dict[str, float]:
        if label not in self.weights:
            self.weights[label] = dict(self._warm_start_weights(label))
        return self.weights[label]

    @staticmethod
    def _history_attempts(history: Any) -> float:
        if history is None or not hasattr(history, "by_label"):
            return 0.0
        total = 0.0
        for bucket in getattr(history, "by_label", {}).values():
            total += _safe_rate((bucket or {}).get("attempts", 0.0), 0.0)
        return float(total)

    def _feature_map(
        self,
        *,
        candidate: Any,
        state: Any,
        history: Any,
        node_fail_streak: Dict[Tuple[str, str], int],
    ) -> Dict[str, float]:
        label = candidate.get("label", "unknown")
        node_name = candidate.get("node_name") or ""
        style = candidate.get("style") or ""
        op_family = candidate.get("op_family") or ""
        failure_prob = _safe_rate(candidate.get("failure_prob", 0.0))
        component_key = candidate.get("component", "")
        if label == "prompt_explore":
            component_key = "Prompt"
        elif label == "params_explore":
            component_key = "Params"
        node_state = state.nodes.get(node_name)
        style_success = 0.0
        style_reward = 0.0
        subtype_prob = 0.0
        subtype_success = 0.0
        subtype_reward = 0.0
        if node_state is not None:
            style_key = f"{label}:{style}" if style else ""
            if style_key:
                style_success = _safe_rate(node_state.style_success_rates.get(style_key, 0.0))
                style_reward = _safe_rate(node_state.style_mean_rewards.get(style_key, 0.0))
            op_key = f"{label}:{op_family}" if op_family else ""
            op_success = _safe_rate(node_state.op_success_rates.get(op_key, 0.0)) if op_key else 0.0
            op_reward = _safe_rate(node_state.op_mean_rewards.get(op_key, 0.0)) if op_key else 0.0
            subtype_key = f"{candidate.get('component', '')}:{candidate.get('subtype', '')}"
            subtype_prob = _safe_rate(node_state.subtype_failure_probs.get(subtype_key, 0.0))
            subtype_hist_key = f"{label}:{candidate.get('component', '')}:{candidate.get('subtype', '')}"
            subtype_success = _safe_rate(node_state.subtype_success_rates.get(subtype_hist_key, 0.0))
            subtype_reward = _safe_rate(node_state.subtype_mean_rewards.get(subtype_hist_key, 0.0))
        else:
            op_success = 0.0
            op_reward = 0.0
        observation_variance = 0.0
        if candidate.get("component") == "Structure":
            observation_variance = _safe_rate(getattr(state, "structure_observation_variance", 0.0))
        elif candidate.get("component") == "Edge":
            observation_variance = _safe_rate(getattr(state, "edge_observation_variance", 0.0))
        elif node_state is not None:
            observation_variance = _safe_rate(getattr(node_state, "observation_variance", 0.0))
        return {
            "bias": 1.0,
            "baseline_f1": state.baseline_f1,
            "baseline_em": state.baseline_em,
            "last_action_f1_delta": _safe_rate(getattr(state, "last_action_f1_delta", 0.0)),
            "rca_delta_from_init": _safe_rate(getattr(state, "rca_delta_from_init", 0.0)),
            "failure_prob": failure_prob,
            "subtype_failure_prob": subtype_prob,
            "subtype_success": subtype_success,
            "subtype_reward": subtype_reward,
            "target_rank": _safe_rate(candidate.get("target_rank", 0.0)),
            "target_pool_size": _safe_rate(candidate.get("target_pool_size", 0.0)),
            "rca_locked": 1.0 if candidate.get("pool_mode") == "strong_rca" else 0.0,
            "from_rca": 1.0 if candidate.get("source") == "rca" else 0.0,
            "style_success": style_success,
            "style_reward": style_reward,
            "op_success": op_success,
            "op_reward": op_reward,
            "obs_coverage": state.obs_coverage,
            "weak_rca": 1.0 if state.weak_rca else 0.0,
            "rca_entropy": state.rca_entropy,
            "node_count": float(state.node_count),
            "edge_count": float(state.edge_count),
            "dag_depth": float(state.dag_depth),
            "role_count": float(state.role_count),
            "budget_ratio": state.budget_ratio,
            "no_improve_ratio": state.no_improve_ratio,
            "label_success": history.label_success_rate(label) if history else 0.0,
            "action_type_success": history.label_success_rate(label) if history else 0.0,
            "node_success": history.node_success_rate(label, node_name) if history else 0.0,
            "node_prompt_quality": getattr(node_state, "prompt_quality", 0.0) if node_state else 0.0,
            "node_params_quality": getattr(node_state, "params_quality", 0.0) if node_state else 0.0,
            "node_return_quality": getattr(node_state, "return_quality", 0.0) if node_state else 0.0,
            "observation_variance": observation_variance,
            "fail_streak": float(node_fail_streak.get((component_key, node_name), 0)),
        }

    def _score(self, label: str, features: Dict[str, float]) -> float:
        label_weights = self._ensure_label_weights(label)
        return sum(label_weights.get(name, 0.0) * value for name, value in features.items())

    @staticmethod
    def _softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
        if not scores:
            return []
        max_score = max(scores)
        temp = max(0.25, float(temperature))
        exps = [math.exp((s - max_score) / temp) for s in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(scores)] * len(scores)
        return [v / total for v in exps]

    def select(
        self,
        *,
        candidates: List[Any],
        workflow_graph: Any,
        base_pkg: Any,
        prompt_history: Any,
        step_index: int,
        no_improve_count: int,
        max_opt_iterations: int,
        history: Any,
        node_fail_streak: Dict[Tuple[str, str], int],
        strong_rca_threshold: float,
        role_meta: Dict[str, Any],
        last_action_f1_delta: float = 0.0,
        init_top_actionable_failure_prob: float = 0.0,
    ) -> Tuple[int, Any, List[float], Any]:
        state = build_workflow_state(
            workflow_graph=workflow_graph,
            base_pkg=base_pkg,
            prompt_history=prompt_history,
            action_history=history,
            node_fail_streak=node_fail_streak,
            iter_idx=step_index,
            max_opt_iterations=max_opt_iterations,
            no_improve_count=no_improve_count,
            strong_rca_threshold=strong_rca_threshold,
            role_meta=role_meta,
            last_action_f1_delta=last_action_f1_delta,
            init_top_actionable_failure_prob=init_top_actionable_failure_prob,
        )
        scored: List[Any] = []
        for candidate in candidates:
            features = self._feature_map(
                candidate=candidate,
                state=state,
                history=history,
                node_fail_streak=node_fail_streak,
            )
            candidate["_policy_features"] = features
            scored.append(candidate)
        scores = [self._score(c.get("label", "unknown"), c["_policy_features"]) for c in scored]
        probs = self._softmax(scores, temperature=self.exploration_temperature)
        history_attempts = self._history_attempts(history)
        non_skip_indices = [i for i, cand in enumerate(scored) if cand.get("label") != "skip"]
        if non_skip_indices and (step_index <= self.forced_exploit_steps or history_attempts < 2.0):
            idx = max(
                non_skip_indices,
                key=lambda i: (
                    scores[i],
                    _safe_rate(scored[i].get("failure_prob", 0.0)),
                    -_safe_rate(scored[i].get("target_rank", 0.0)),
                    1.0 if scored[i].get("source") == "rca" else 0.0,
                ),
            )
        else:
            draw = random.random()
            cum = 0.0
            idx = len(scored) - 1
            for i, prob in enumerate(probs):
                cum += prob
                if draw <= cum:
                    idx = i
                    break
        return idx, scored[idx], probs, state

    def update(
        self,
        *,
        candidates: List[Any],
        chosen_idx: int,
        probs: List[float],
        reward: float,
    ):
        if not candidates or chosen_idx < 0 or chosen_idx >= len(candidates):
            return
        for idx, candidate in enumerate(candidates):
            label = candidate.get("label", "unknown")
            label_weights = self._ensure_label_weights(label)
            features = candidate.get("_policy_features", {})
            coeff = (1.0 if idx == chosen_idx else 0.0) - probs[idx]
            for name, value in features.items():
                label_weights[name] = label_weights.get(name, 0.0) + self.learning_rate * reward * coeff * value
