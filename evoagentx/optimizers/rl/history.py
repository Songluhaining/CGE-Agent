from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PromptRecord:
    iteration: int
    prompt_text: str
    metrics: Dict[str, float]
    failure_prob: float
    operations_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "prompt_text": self.prompt_text,
            "metrics": dict(self.metrics),
            "failure_prob": float(self.failure_prob),
            "operations_applied": list(self.operations_applied),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptRecord":
        return cls(
            iteration=int(d["iteration"]),
            prompt_text=str(d["prompt_text"]),
            metrics=dict(d.get("metrics", {})),
            failure_prob=float(d.get("failure_prob", 0.0)),
            operations_applied=list(d.get("operations_applied", [])),
        )


class PromptHistory:
    def __init__(self):
        self._history: Dict[str, List[PromptRecord]] = {}

    def add_record(self, node_name: str, record: PromptRecord):
        self._history.setdefault(node_name, []).append(record)

    def get_history(self, node_name: str) -> List[PromptRecord]:
        return self._history.get(node_name, [])

    def get_best_record(self, node_name: str) -> Optional[PromptRecord]:
        records = self.get_history(node_name)
        if not records:
            return None
        return max(
            records,
            key=lambda x: (float(x.metrics.get("f1", 0.0)), float(x.metrics.get("em", 0.0))),
        )

    def get_all_node_names(self) -> List[str]:
        return sorted(self._history.keys())

    def format_history_for_llm(self, node_name: str, max_records: int = 5) -> str:
        records = self.get_history(node_name)
        if not records:
            return "No previous history."
        selected = records[-max_records:]
        lines: List[str] = []
        prev_f1: Optional[float] = None
        for record in selected:
            f1 = float(record.metrics.get("f1", 0.0))
            em = float(record.metrics.get("em", 0.0))
            failure_prob = float(record.failure_prob)
            trend = ""
            if prev_f1 is not None:
                delta = f1 - prev_f1
                if delta > 1e-3:
                    direction = "up"
                elif delta < -1e-3:
                    direction = "down"
                else:
                    direction = "flat"
                trend = f" (f1 {direction} {delta:+.4f})"
            prev_f1 = f1
            ops_summary = ", ".join(str(op) for op in record.operations_applied[:3]) if record.operations_applied else "initial"
            lines.append(
                f"- Iter {record.iteration}: f1={f1:.4f}, em={em:.4f}, "
                f"failure_prob={failure_prob:.4f}{trend}, changes=[{ops_summary}]"
            )

        best = self.get_best_record(node_name)
        if best is not None:
            best_ops = ", ".join(str(op) for op in best.operations_applied[:2]) if best.operations_applied else "initial"
            lines.append(
                f"Best so far: iter={best.iteration}, f1={float(best.metrics.get('f1', 0.0)):.4f}, "
                f"em={float(best.metrics.get('em', 0.0)):.4f}, changes=[{best_ops}]"
            )

        if len(records) >= 3:
            first_f1 = float(records[0].metrics.get("f1", 0.0))
            last_f1 = float(records[-1].metrics.get("f1", 0.0))
            if last_f1 > first_f1 + 0.01:
                overall = "improving"
            elif last_f1 < first_f1 - 0.01:
                overall = "degrading"
            else:
                overall = "stagnant"
            lines.append(f"Overall trend: {overall} (f1: {first_f1:.4f} -> {last_f1:.4f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            node_name: [r.to_dict() for r in records]
            for node_name, records in self._history.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptHistory":
        obj = cls()
        for node_name, records in d.items():
            obj._history[node_name] = [PromptRecord.from_dict(r) for r in records]
        return obj


class ActionOutcomeHistory:
    def __init__(self):
        self.by_label: Dict[str, Dict[str, float]] = {}
        self.by_node_label: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.by_node_style_label: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self.by_node_op_label: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self.by_label_component_subtype: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self.by_node_component_subtype_label: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
        self.last_successful_style_by_node_label: Dict[Tuple[str, str], str] = {}
        self.last_successful_op_by_node_label: Dict[Tuple[str, str], str] = {}
        self.failed_styles_by_node_label: Dict[Tuple[str, str], List[str]] = {}
        self.failed_ops_by_node_label: Dict[Tuple[str, str], List[str]] = {}

    @staticmethod
    def _update_bucket(bucket: Dict[str, float], reward: float, accepted: bool):
        bucket["attempts"] = bucket.get("attempts", 0.0) + 1.0
        bucket["accepts"] = bucket.get("accepts", 0.0) + (1.0 if accepted else 0.0)
        prev_mean = bucket.get("mean_reward", 0.0)
        n = bucket["attempts"]
        bucket["mean_reward"] = prev_mean + (reward - prev_mean) / max(1.0, n)

    def record(
        self,
        label: str,
        node_name: Optional[str],
        reward: float,
        accepted: bool,
        style: Optional[str] = None,
        component: Optional[str] = None,
        subtype: Optional[str] = None,
        selected_op_family: Optional[str] = None,
    ):
        self._update_bucket(self.by_label.setdefault(label, {}), reward, accepted)
        if component and subtype:
            self._update_bucket(
                self.by_label_component_subtype.setdefault((label, component, subtype), {}),
                reward,
                accepted,
            )
        if node_name:
            self._update_bucket(self.by_node_label.setdefault((label, node_name), {}), reward, accepted)
            if component and subtype:
                self._update_bucket(
                    self.by_node_component_subtype_label.setdefault((label, node_name, component, subtype), {}),
                    reward,
                    accepted,
                )
            if style:
                self._update_bucket(
                    self.by_node_style_label.setdefault((label, node_name, style), {}),
                    reward,
                    accepted,
                )
                key = (label, node_name)
                failed_styles = self.failed_styles_by_node_label.setdefault(key, [])
                if accepted:
                    self.last_successful_style_by_node_label[key] = style
                    self.failed_styles_by_node_label[key] = [s for s in failed_styles if s != style]
                else:
                    if style not in failed_styles:
                        failed_styles.append(style)
            if selected_op_family:
                op_family = str(selected_op_family).upper().strip()
                self._update_bucket(
                    self.by_node_op_label.setdefault((label, node_name, op_family), {}),
                    reward,
                    accepted,
                )
                key = (label, node_name)
                failed_ops = self.failed_ops_by_node_label.setdefault(key, [])
                if accepted:
                    self.last_successful_op_by_node_label[key] = op_family
                    self.failed_ops_by_node_label[key] = [s for s in failed_ops if s != op_family]
                else:
                    if op_family not in failed_ops:
                        failed_ops.append(op_family)

    def label_success_rate(self, label: str) -> float:
        bucket = self.by_label.get(label, {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def label_attempts(self, label: str) -> float:
        return float(self.by_label.get(label, {}).get("attempts", 0.0))

    def node_success_rate(self, label: str, node_name: Optional[str]) -> float:
        if not node_name:
            return 0.0
        bucket = self.by_node_label.get((label, node_name), {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def node_style_attempts(self, label: str, node_name: Optional[str], style: Optional[str]) -> float:
        if not node_name or not style:
            return 0.0
        return float(self.by_node_style_label.get((label, node_name, style), {}).get("attempts", 0.0))

    def node_style_success_rate(self, label: str, node_name: Optional[str], style: Optional[str]) -> float:
        if not node_name or not style:
            return 0.0
        bucket = self.by_node_style_label.get((label, node_name, style), {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def node_style_mean_reward(self, label: str, node_name: Optional[str], style: Optional[str]) -> float:
        if not node_name or not style:
            return 0.0
        bucket = self.by_node_style_label.get((label, node_name, style), {})
        return float(bucket.get("mean_reward", 0.0))

    def node_op_attempts(self, label: str, node_name: Optional[str], op_family: Optional[str]) -> float:
        if not node_name or not op_family:
            return 0.0
        return float(self.by_node_op_label.get((label, node_name, str(op_family).upper().strip()), {}).get("attempts", 0.0))

    def node_op_success_rate(self, label: str, node_name: Optional[str], op_family: Optional[str]) -> float:
        if not node_name or not op_family:
            return 0.0
        bucket = self.by_node_op_label.get((label, node_name, str(op_family).upper().strip()), {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def node_op_mean_reward(self, label: str, node_name: Optional[str], op_family: Optional[str]) -> float:
        if not node_name or not op_family:
            return 0.0
        bucket = self.by_node_op_label.get((label, node_name, str(op_family).upper().strip()), {})
        return float(bucket.get("mean_reward", 0.0))

    def label_component_subtype_attempts(self, label: str, component: str, subtype: str) -> float:
        return float(self.by_label_component_subtype.get((label, component, subtype), {}).get("attempts", 0.0))

    def label_component_subtype_success_rate(self, label: str, component: str, subtype: str) -> float:
        bucket = self.by_label_component_subtype.get((label, component, subtype), {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def label_component_subtype_mean_reward(self, label: str, component: str, subtype: str) -> float:
        bucket = self.by_label_component_subtype.get((label, component, subtype), {})
        return float(bucket.get("mean_reward", 0.0))

    def node_subtype_attempts(self, label: str, node_name: Optional[str], component: str, subtype: str) -> float:
        if not node_name:
            return 0.0
        return float(self.by_node_component_subtype_label.get((label, node_name, component, subtype), {}).get("attempts", 0.0))

    def node_subtype_success_rate(self, label: str, node_name: Optional[str], component: str, subtype: str) -> float:
        if not node_name:
            return 0.0
        bucket = self.by_node_component_subtype_label.get((label, node_name, component, subtype), {})
        attempts = bucket.get("attempts", 0.0)
        if attempts <= 0:
            return 0.0
        return float(bucket.get("accepts", 0.0) / attempts)

    def node_subtype_mean_reward(self, label: str, node_name: Optional[str], component: str, subtype: str) -> float:
        if not node_name:
            return 0.0
        bucket = self.by_node_component_subtype_label.get((label, node_name, component, subtype), {})
        return float(bucket.get("mean_reward", 0.0))

    def last_successful_style(self, label: str, node_name: Optional[str]) -> Optional[str]:
        if not node_name:
            return None
        return self.last_successful_style_by_node_label.get((label, node_name))

    def last_successful_op(self, label: str, node_name: Optional[str]) -> Optional[str]:
        if not node_name:
            return None
        return self.last_successful_op_by_node_label.get((label, node_name))

    def failed_styles(self, label: str, node_name: Optional[str], max_items: int = 5) -> List[str]:
        if not node_name:
            return []
        return list(self.failed_styles_by_node_label.get((label, node_name), []))[-max_items:]

    def failed_op_families(self, label: str, node_name: Optional[str], max_items: int = 5) -> List[str]:
        if not node_name:
            return []
        return list(self.failed_ops_by_node_label.get((label, node_name), []))[-max_items:]


@dataclass
class ModificationRecord:
    iteration: int
    candidate_id: str
    target_component: str
    target_subtype: str
    target_node_name: str
    rca_rank: int
    edit_kind: str
    style: str
    op_family: str = ""
    structure_variant: str = ""
    rationale: str = ""
    history_reference: str = ""
    expected_effect: str = ""
    materialization_status: str = ""
    validation_status: str = ""
    duplicate_kind: str = ""
    baseline_f1: float = 0.0
    baseline_em: float = 0.0
    baseline_utility: float = 0.0
    baseline_estimated_full_f1: float = 0.0
    candidate_f1: float = 0.0
    candidate_em: float = 0.0
    candidate_utility: float = 0.0
    candidate_estimated_full_f1: float = 0.0
    utility_delta: float = 0.0
    estimated_full_f1_delta: float = 0.0
    reward_like_delta: float = 0.0
    accepted: bool = False
    node_metric_deltas: dict = None  # {node_name: {"f1_delta": float, "em_delta": float}}

    def __post_init__(self):
        if self.node_metric_deltas is None:
            self.node_metric_deltas = {}

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["node_metric_deltas"] = dict(self.node_metric_deltas or {})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModificationRecord":
        kwargs = dict(d)
        kwargs["node_metric_deltas"] = kwargs.get("node_metric_deltas") or {}
        return cls(**kwargs)


class ModificationHistory:
    def __init__(self):
        self.records: List[ModificationRecord] = []

    def add_record(self, record: ModificationRecord):
        self.records.append(record)

    def recent_records(self, max_records: int = 5) -> List[ModificationRecord]:
        return list(self.records[-max_records:])

    def best_accepted_records(self, max_records: int = 3) -> List[ModificationRecord]:
        accepted = [record for record in self.records if record.accepted]
        accepted.sort(
            key=lambda record: (
                float(record.estimated_full_f1_delta),
                float(record.candidate_estimated_full_f1),
                float(record.utility_delta),
                float(record.candidate_f1),
                float(record.candidate_em),
            ),
            reverse=True,
        )
        return accepted[:max_records]

    def recent_failures(self, max_records: int = 5) -> List[ModificationRecord]:
        failures = [
            record
            for record in self.records
            if (not record.accepted)
            and (
                record.materialization_status not in {"accepted", "evaluated"}
                or record.validation_status not in {"valid", ""}
                or record.duplicate_kind
                or float(record.estimated_full_f1_delta or record.utility_delta) <= 0.0
            )
        ]
        return failures[-max_records:]

    def summarize_recent_attempts_by_node(self, max_nodes: int = 8, max_records_per_node: int = 3) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for record in reversed(self.records):
            node_name = record.target_node_name or "__STRUCTURE__"
            bucket = grouped.setdefault(node_name, [])
            if len(bucket) >= max_records_per_node:
                continue
            bucket.append(
                f"iter={record.iteration}, kind={record.edit_kind}, style={record.style or '-'}, "
                f"op={record.op_family or '-'}, variant={record.structure_variant or '-'}, "
                f"status={record.materialization_status or record.validation_status or 'unknown'}, "
                f"score_delta={float(record.estimated_full_f1_delta):+.4f}, utility_delta={float(record.utility_delta):+.4f}, accepted={bool(record.accepted)}"
            )
            if len(grouped) >= max_nodes and all(len(v) >= max_records_per_node for v in grouped.values()):
                break
        return grouped

    def summarize_failure_patterns(self, max_patterns: int = 5) -> List[str]:
        patterns: Dict[str, int] = {}
        for record in self.records:
            if record.accepted:
                continue
            parts = [
                record.edit_kind or "-",
                record.style or "-",
                record.op_family or "-",
                record.structure_variant or "-",
                record.materialization_status or "-",
                record.validation_status or "-",
                record.duplicate_kind or "-",
            ]
            key = " | ".join(parts)
            patterns[key] = patterns.get(key, 0) + 1
        ranked = sorted(patterns.items(), key=lambda item: item[1], reverse=True)
        return [f"{pattern} -> {count}x" for pattern, count in ranked[:max_patterns]]

    def summarize_pattern_performance(
        self,
        *,
        accepted: Optional[bool] = None,
        max_patterns: int = 6,
    ) -> List[Dict[str, object]]:
        buckets: Dict[Tuple[str, str, str, str, str, str], Dict[str, float]] = {}
        for record in self.records:
            key = (
                record.edit_kind or "",
                record.style or "",
                record.op_family or "",
                record.structure_variant or "",
                record.target_component or "",
                record.target_subtype or "",
            )
            bucket = buckets.setdefault(
                key,
                {
                    "attempts": 0.0,
                    "accepts": 0.0,
                    "invalid_or_duplicate": 0.0,
                    "mean_utility_delta": 0.0,
                    "mean_estimated_full_f1_delta": 0.0,
                },
            )
            bucket["attempts"] += 1.0
            bucket["accepts"] += 1.0 if record.accepted else 0.0
            if record.validation_status not in {"", "valid"} or record.duplicate_kind:
                bucket["invalid_or_duplicate"] += 1.0
            attempts = bucket["attempts"]
            prev_mean = bucket["mean_utility_delta"]
            bucket["mean_utility_delta"] = prev_mean + (float(record.utility_delta) - prev_mean) / max(1.0, attempts)
            prev_score_mean = bucket["mean_estimated_full_f1_delta"]
            bucket["mean_estimated_full_f1_delta"] = (
                prev_score_mean
                + (float(record.estimated_full_f1_delta) - prev_score_mean) / max(1.0, attempts)
            )

        rows: List[Dict[str, object]] = []
        for key, bucket in buckets.items():
            attempts = float(bucket.get("attempts", 0.0))
            accepts = float(bucket.get("accepts", 0.0))
            failures = attempts - accepts
            if accepted is True and accepts <= 0.0:
                continue
            if accepted is False and failures <= 0.0:
                continue
            edit_kind, style, op_family, structure_variant, target_component, target_subtype = key
            invalid_or_duplicate = float(bucket.get("invalid_or_duplicate", 0.0))
            rows.append(
                {
                    "edit": {
                        "kind": edit_kind,
                        "style": style,
                        "op_family": op_family,
                        "structure_variant": structure_variant,
                    },
                    "target": {
                        "component": target_component,
                        "subtype": target_subtype,
                    },
                    "attempts": attempts,
                    "accept_rate": accepts / max(1.0, attempts),
                    "invalid_or_duplicate_rate": invalid_or_duplicate / max(1.0, attempts),
                    "mean_estimated_full_f1_delta": float(bucket.get("mean_estimated_full_f1_delta", 0.0)),
                    "mean_utility_delta": float(bucket.get("mean_utility_delta", 0.0)),
                }
            )

        if accepted is True:
            rows.sort(
                key=lambda row: (
                    float(row.get("mean_estimated_full_f1_delta", 0.0)),
                    float(row.get("mean_utility_delta", 0.0)),
                    float(row.get("accept_rate", 0.0)),
                    float(row.get("attempts", 0.0)),
                ),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: (
                    -float(row.get("invalid_or_duplicate_rate", 0.0)),
                    float(row.get("mean_estimated_full_f1_delta", 0.0)),
                    float(row.get("mean_utility_delta", 0.0)),
                    -float(row.get("attempts", 0.0)),
                )
            )
        return rows[:max_patterns]

    def summarize_node_edit_outcomes(self, max_nodes: int = 8, max_entries_per_node: int = 4) -> Dict[str, List[Dict[str, object]]]:
        grouped: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, float]]] = {}
        for record in self.records:
            node_name = record.target_node_name or "__STRUCTURE__"
            node_bucket = grouped.setdefault(node_name, {})
            key = (
                record.edit_kind or "",
                record.style or "",
                record.op_family or "",
                record.structure_variant or "",
            )
            bucket = node_bucket.setdefault(
                key,
                {
                    "attempts": 0.0,
                    "accepts": 0.0,
                    "invalid_or_duplicate": 0.0,
                    "mean_utility_delta": 0.0,
                    "mean_estimated_full_f1_delta": 0.0,
                },
            )
            bucket["attempts"] += 1.0
            bucket["accepts"] += 1.0 if record.accepted else 0.0
            if record.validation_status not in {"", "valid"} or record.duplicate_kind:
                bucket["invalid_or_duplicate"] += 1.0
            attempts = bucket["attempts"]
            prev_mean = bucket["mean_utility_delta"]
            bucket["mean_utility_delta"] = prev_mean + (float(record.utility_delta) - prev_mean) / max(1.0, attempts)
            prev_score_mean = bucket["mean_estimated_full_f1_delta"]
            bucket["mean_estimated_full_f1_delta"] = (
                prev_score_mean
                + (float(record.estimated_full_f1_delta) - prev_score_mean) / max(1.0, attempts)
            )

        ranked_nodes = sorted(
            grouped.items(),
            key=lambda item: max((float(bucket.get("attempts", 0.0)) for bucket in item[1].values()), default=0.0),
            reverse=True,
        )
        result: Dict[str, List[Dict[str, object]]] = {}
        for node_name, pattern_buckets in ranked_nodes[:max_nodes]:
            rows: List[Dict[str, object]] = []
            for key, bucket in pattern_buckets.items():
                edit_kind, style, op_family, structure_variant = key
                attempts = float(bucket.get("attempts", 0.0))
                accepts = float(bucket.get("accepts", 0.0))
                invalid_or_duplicate = float(bucket.get("invalid_or_duplicate", 0.0))
                rows.append(
                    {
                        "edit": {
                            "kind": edit_kind,
                            "style": style,
                            "op_family": op_family,
                            "structure_variant": structure_variant,
                        },
                        "attempts": attempts,
                        "accept_rate": accepts / max(1.0, attempts),
                        "invalid_or_duplicate_rate": invalid_or_duplicate / max(1.0, attempts),
                        "mean_estimated_full_f1_delta": float(bucket.get("mean_estimated_full_f1_delta", 0.0)),
                        "mean_utility_delta": float(bucket.get("mean_utility_delta", 0.0)),
                    }
                )
            rows.sort(
                key=lambda row: (
                    float(row.get("mean_estimated_full_f1_delta", 0.0)),
                    float(row.get("mean_utility_delta", 0.0)),
                    float(row.get("accept_rate", 0.0)),
                    float(row.get("attempts", 0.0)),
                ),
                reverse=True,
            )
            result[node_name] = rows[:max_entries_per_node]
        return result

    def summarize_for_llm(
        self,
        *,
        recent_limit: int = 5,
        best_limit: int = 3,
        failure_limit: int = 5,
    ) -> Dict[str, object]:
        def _record_to_dict(record: ModificationRecord) -> Dict[str, object]:
            return {
                "iteration": int(record.iteration),
                "candidate_id": record.candidate_id,
                "target": {
                    "component": record.target_component,
                    "subtype": record.target_subtype,
                    "node_name": record.target_node_name,
                    "rca_rank": int(record.rca_rank),
                },
                "edit": {
                    "kind": record.edit_kind,
                    "style": record.style,
                    "op_family": record.op_family,
                    "structure_variant": record.structure_variant,
                },
                "statuses": {
                    "materialization": record.materialization_status,
                    "validation": record.validation_status,
                    "duplicate": record.duplicate_kind,
                    "accepted": bool(record.accepted),
                },
                "metrics": {
                    "baseline_f1": float(record.baseline_f1),
                    "baseline_em": float(record.baseline_em),
                    "baseline_utility": float(record.baseline_utility),
                    "baseline_estimated_full_f1": float(record.baseline_estimated_full_f1),
                    "candidate_f1": float(record.candidate_f1),
                    "candidate_em": float(record.candidate_em),
                    "candidate_utility": float(record.candidate_utility),
                    "candidate_estimated_full_f1": float(record.candidate_estimated_full_f1),
                    "estimated_full_f1_delta": float(record.estimated_full_f1_delta),
                    "utility_delta": float(record.utility_delta),
                },
                "reasoning": {
                    "rationale": record.rationale,
                    "history_reference": record.history_reference,
                    "expected_effect": record.expected_effect,
                },
            }

        recent_records = self.recent_records(recent_limit)
        best_records = self.best_accepted_records(best_limit)
        recent_failures = self.recent_failures(failure_limit)
        failure_patterns = self.summarize_failure_patterns(max_patterns=failure_limit)
        best_patterns = self.summarize_pattern_performance(accepted=True, max_patterns=max(best_limit, 3))
        risky_patterns = self.summarize_pattern_performance(accepted=False, max_patterns=max(failure_limit, 5))
        node_edit_outcomes = self.summarize_node_edit_outcomes()

        narrative_lines: List[str] = []
        total = len(self.records)
        accepted_count = sum(1 for record in self.records if record.accepted)
        if total > 0:
            narrative_lines.append(
                f"Total {total} modifications attempted, {accepted_count} accepted ({accepted_count / max(1, total) * 100:.0f}% acceptance rate)."
            )
        if best_records:
            best = best_records[0]
            target_name = best.target_node_name or "__STRUCTURE__"
            narrative_lines.append(
                f"Most effective edit so far: {best.edit_kind}/{best.style} on {target_name} "
                f"(estimated_full_f1 {best.baseline_estimated_full_f1:.4f} -> {best.candidate_estimated_full_f1:.4f}, "
                f"delta {best.estimated_full_f1_delta:+.4f})."
            )
        if failure_patterns:
            narrative_lines.append(
                f"Most common failure pattern: {failure_patterns[0]}. Avoid repeating it without new evidence."
            )
        if len(recent_records) >= 3 and all(not record.accepted for record in recent_records[-3:]):
            narrative_lines.append(
                "Recent 3 attempts all failed. Consider changing strategy instead of repeating the same edit family."
            )

        # Summarize per-node impact from metric deltas
        node_impact: Dict[str, Dict[str, float]] = {}
        for record in self.records:
            if not record.node_metric_deltas:
                continue
            for nd_name, deltas in record.node_metric_deltas.items():
                entry = node_impact.setdefault(nd_name, {"sum_task_ok_delta": 0.0, "count": 0})
                entry["sum_task_ok_delta"] += float(deltas.get("task_ok_delta", 0.0))
                entry["count"] += 1
        node_impact_summary = {
            nd: {"avg_task_ok_delta": round(v["sum_task_ok_delta"] / max(1, v["count"]), 4), "modification_count": v["count"]}
            for nd, v in node_impact.items()
        }

        # Build AFlow-style direct prohibition list: explicitly name what NOT to try
        # Format mirrors AFlow "Absolutely prohibit X (Score: Y)" for clear LLM guidance
        plain_prohibitions: List[str] = []
        for record in recent_failures:
            if record.rationale or record.style:
                desc = record.rationale or f"{record.edit_kind}/{record.style} on {record.target_node_name}"
                plain_prohibitions.append(
                    f"AVOID: {desc} (estimated_full_f1 {record.baseline_estimated_full_f1:.3f} -> "
                    f"{record.candidate_estimated_full_f1:.3f})"
                )
        for rec in self.records:
            if rec.accepted and (rec.rationale or rec.style):
                desc = rec.rationale or f"{rec.edit_kind}/{rec.style} on {rec.target_node_name}"
                plain_prohibitions.append(
                    f"ALREADY_DONE: {desc} (estimated_full_f1_delta={rec.estimated_full_f1_delta:+.4f})"
                )

        return {
            "recent_records": [_record_to_dict(record) for record in recent_records],
            "best_accepted_records": [_record_to_dict(record) for record in best_records],
            "recent_failures": [_record_to_dict(record) for record in recent_failures],
            "recent_attempts_by_node": self.summarize_recent_attempts_by_node(),
            "failure_patterns": failure_patterns,
            "best_accepted_patterns": best_patterns,
            "risky_patterns": risky_patterns,
            "node_edit_outcomes": node_edit_outcomes,
            "node_impact_summary": node_impact_summary,
            "narrative_summary": " ".join(narrative_lines).strip(),
            "plain_prohibitions": plain_prohibitions[:12],
            "primary_selection_metric": "estimated_full_f1",
        }

    def style_summary(self) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
        buckets: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
        for record in self.records:
            key = (
                record.edit_kind or "",
                record.style or "",
                record.op_family or "",
                record.structure_variant or "",
            )
            bucket = buckets.setdefault(
                key,
                {
                    "attempts": 0.0,
                    "accepts": 0.0,
                    "mean_utility_delta": 0.0,
                    "mean_estimated_full_f1_delta": 0.0,
                },
            )
            bucket["attempts"] += 1.0
            bucket["accepts"] += 1.0 if record.accepted else 0.0
            attempts = bucket["attempts"]
            prev_mean = bucket["mean_utility_delta"]
            bucket["mean_utility_delta"] = prev_mean + (float(record.utility_delta) - prev_mean) / max(1.0, attempts)
            prev_score_mean = bucket["mean_estimated_full_f1_delta"]
            bucket["mean_estimated_full_f1_delta"] = (
                prev_score_mean
                + (float(record.estimated_full_f1_delta) - prev_score_mean) / max(1.0, attempts)
            )
        return buckets

    def to_dict(self) -> dict:
        return {"records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, d: dict) -> "ModificationHistory":
        obj = cls()
        obj.records = [ModificationRecord.from_dict(r) for r in d.get("records", [])]
        return obj

