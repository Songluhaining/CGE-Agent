import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class EvaluationPackage:
    workflow_fingerprint: str
    eval_indices: Tuple[int, ...]
    results: Dict[str, float]
    evidences: Dict[str, Any]
    root_causes: list
    node_stats: Dict[str, Any]
    evaluation_records: Dict[str, Any] = field(default_factory=dict)
    failure_examples: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens_delta: int = 0
    total_cost_delta: float = 0.0
    obs_coverage: float = 0.0

    @property
    def token_delta(self) -> int:
        return int(self.total_tokens_delta)

    @property
    def cost_delta(self) -> float:
        return float(self.total_cost_delta)


def workflow_fingerprint(
    workflow_graph: Any,
    llm: Any,
    eval_indices: Optional[Iterable[int]],
    eval_mode: str,
) -> str:
    try:
        wf_dict = workflow_graph.to_dict()
    except Exception:
        wf_dict = {"goal": getattr(workflow_graph, "goal", ""), "nodes": [], "edges": []}
    if isinstance(wf_dict, dict):
        wf_dict = copy.deepcopy(wf_dict)
        wf_dict.pop("graph", None)
    payload = {
        "workflow": wf_dict,
        "eval_mode": eval_mode,
        "eval_indices": list(eval_indices or []),
        "llm_model": getattr(getattr(llm, "config", None), "model", ""),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EvaluationCache:
    """
    Cache complete evaluation packages keyed by workflow content and eval setup.
    """

    def __init__(self):
        self._cache: Dict[str, EvaluationPackage] = {}

    def get(
        self,
        workflow_graph: Any,
        llm: Any,
        eval_indices: Optional[Iterable[int]],
        eval_mode: str,
    ) -> Optional[EvaluationPackage]:
        key = workflow_fingerprint(workflow_graph, llm, eval_indices, eval_mode)
        return self._cache.get(key)

    def put(
        self,
        workflow_graph: Any,
        llm: Any,
        eval_indices: Optional[Iterable[int]],
        eval_mode: str,
        package: EvaluationPackage,
    ) -> EvaluationPackage:
        key = workflow_fingerprint(workflow_graph, llm, eval_indices, eval_mode)
        package.workflow_fingerprint = key
        package.eval_indices = tuple(int(i) for i in (eval_indices or []))
        self._cache[key] = package
        return package

    def get_or_evaluate(
        self,
        *,
        workflow_graph: Any,
        llm: Any,
        eval_indices: Optional[Iterable[int]],
        eval_mode: str,
        runner: Callable[[], EvaluationPackage],
    ) -> Tuple[EvaluationPackage, bool]:
        cached = self.get(
            workflow_graph=workflow_graph,
            llm=llm,
            eval_indices=eval_indices,
            eval_mode=eval_mode,
        )
        if cached is not None:
            return cached, True

        package = runner()
        stored = self.put(
            workflow_graph=workflow_graph,
            llm=llm,
            eval_indices=eval_indices,
            eval_mode=eval_mode,
            package=package,
        )
        return stored, False

    def values(self):
        return list(self._cache.values())

    def clear(self):
        self._cache.clear()
