from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RcaTarget:
    component: str
    subtype: str
    node_name: Optional[str]
    failure_prob: float
    source: str = "rca"
    target_rank: int = 0
    target_pool_size: int = 0
    pool_mode: str = "weak_rca"
    health_name: Optional[str] = None
    edge_source: Optional[str] = None
    edge_target: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default


@dataclass
class ActionSpec:
    label: str
    component: str
    subtype: str
    style: str
    failure_prob: float
    node_name: Optional[str] = None
    target_rank: int = 0
    target_pool_size: int = 0
    pool_mode: str = "weak_rca"
    source: str = "rca"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _policy_features: Dict[str, float] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None and key not in self.metadata and not hasattr(self, key):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value


@dataclass
class OptimizationCandidate:
    label: str
    workflow: Any
    component: str
    node_name: Optional[str] = None
    subtype: str = ""
    style: str = ""
    failure_prob: float = 0.0
    attempted_keys: List[Tuple[str, str]] = field(default_factory=list)
    prompt_ops_by_node: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _policy_features: Dict[str, float] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None and key not in self.metadata and not hasattr(self, key):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value


@dataclass
class EpisodeStep:
    iteration: int
    baseline_f1: float
    candidate_label: Optional[str]
    candidate_f1: Optional[float]
    reward: float
    accepted: bool
