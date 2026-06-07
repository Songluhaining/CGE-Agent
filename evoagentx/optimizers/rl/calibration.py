from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


def _safe_rate(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _beta_mean(successes: float, total: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    total = max(0.0, float(total))
    successes = max(0.0, min(float(successes), total))
    return (alpha + successes) / (alpha + beta + total)


@dataclass
class FactorCalibrationProfile:
    prompt_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    params_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    return_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    edge_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    structure_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    final_obs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    step_slip: Dict[str, float] = field(default_factory=dict)
    health_prior: float = 0.85
    sample_count: int = 0
    action_adjustments: Dict[str, float] = field(default_factory=dict)


def _obs_source(section: str, ev: Any, node_name: Optional[str], dim_name: str) -> str:
    if section == "return":
        if dim_name in {"exact_ok", "overlap_ok"}:
            return "ground_truth"
        judged_dims = set((getattr(ev, "llm_judged_dims", {}) or {}).get(node_name or "", set()) or [])
        if dim_name in judged_dims:
            return "llm_judged"
        if dim_name in {"type_ok", "answer_normalized"}:
            return "deterministic"
        return "heuristic"
    if section == "params":
        return "deterministic"
    return "heuristic"


def _iter_dim_values(packages: Iterable[Any], section: str):
    for pkg in packages:
        evidences = getattr(pkg, "evidences", {}) or {}
        for ev in evidences.values():
            metrics = getattr(ev, "metrics", {}) or {}
            sample_success = 1 if (_safe_rate(metrics.get("em", 0.0)) >= 1.0 or _safe_rate(metrics.get("f1", 0.0)) >= 0.6) else 0
            obs_map = getattr(ev, f"{section}_obs", {}) or {}
            if section == "structure":
                for dim_name, dim_value in obs_map.items():
                    yield sample_success, str(dim_name), _safe_rate(dim_value, 0.0), "heuristic"
            else:
                for node_name, dim_map in obs_map.items():
                    for dim_name, dim_value in (dim_map or {}).items():
                        yield sample_success, str(dim_name), _safe_rate(dim_value, 0.0), _obs_source(section, ev, str(node_name), str(dim_name))


def _build_obs_profile(
    packages: Iterable[Any],
    section: str,
    default_good: float,
    default_bad: float,
    source_defaults: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    pos: Dict[str, float] = {}
    pos_total: Dict[str, float] = {}
    neg: Dict[str, float] = {}
    neg_total: Dict[str, float] = {}
    pos_by_source: Dict[Tuple[str, str], float] = {}
    pos_total_by_source: Dict[Tuple[str, str], float] = {}
    neg_by_source: Dict[Tuple[str, str], float] = {}
    neg_total_by_source: Dict[Tuple[str, str], float] = {}
    for sample_success, dim_name, dim_value, source in _iter_dim_values(packages, section):
        if sample_success:
            pos_total[dim_name] = pos_total.get(dim_name, 0.0) + 1.0
            pos[dim_name] = pos.get(dim_name, 0.0) + dim_value
            pos_total_by_source[(dim_name, source)] = pos_total_by_source.get((dim_name, source), 0.0) + 1.0
            pos_by_source[(dim_name, source)] = pos_by_source.get((dim_name, source), 0.0) + dim_value
        else:
            neg_total[dim_name] = neg_total.get(dim_name, 0.0) + 1.0
            neg[dim_name] = neg.get(dim_name, 0.0) + dim_value
            neg_total_by_source[(dim_name, source)] = neg_total_by_source.get((dim_name, source), 0.0) + 1.0
            neg_by_source[(dim_name, source)] = neg_by_source.get((dim_name, source), 0.0) + dim_value

    dims = set(pos_total) | set(neg_total)
    profile: Dict[str, Dict[str, float]] = {}
    for dim in dims:
        good = _beta_mean(pos.get(dim, 0.0), pos_total.get(dim, 0.0))
        bad = _beta_mean(neg.get(dim, 0.0), neg_total.get(dim, 0.0))
        if pos_total.get(dim, 0.0) <= 0:
            good = default_good
        if neg_total.get(dim, 0.0) <= 0:
            bad = default_bad
        good = max(0.55, min(0.98, good))
        bad = max(0.02, min(0.45, bad))
        if bad >= good:
            bad = max(0.02, min(good - 0.05, 0.45))
        entry = {
            "good_match_prob": good,
            "bad_match_prob": bad,
            "slip_prob": max(0.01, min(0.25, 1.0 - good)),
            "source": "heuristic",
        }
        if source_defaults:
            source_profiles: Dict[str, Dict[str, float]] = {}
            for source_name, defaults in source_defaults.items():
                src_good_default, src_bad_default = defaults
                alpha = 1.0 if source_name == "llm_judged" else 2.0
                beta = 1.0 if source_name == "llm_judged" else 2.0
                src_good = _beta_mean(
                    pos_by_source.get((dim, source_name), 0.0),
                    pos_total_by_source.get((dim, source_name), 0.0),
                    alpha=alpha,
                    beta=beta,
                )
                src_bad = _beta_mean(
                    neg_by_source.get((dim, source_name), 0.0),
                    neg_total_by_source.get((dim, source_name), 0.0),
                    alpha=alpha,
                    beta=beta,
                )
                if pos_total_by_source.get((dim, source_name), 0.0) <= 0:
                    src_good = src_good_default
                if neg_total_by_source.get((dim, source_name), 0.0) <= 0:
                    src_bad = src_bad_default
                src_good = max(0.55, min(0.98, src_good))
                src_bad = max(0.02, min(0.45, src_bad))
                if src_bad >= src_good:
                    src_bad = max(0.02, min(src_good - 0.05, 0.45))
                source_profiles[source_name] = {
                    "good_match_prob": src_good,
                    "bad_match_prob": src_bad,
                    "slip_prob": max(0.01, min(0.25, 1.0 - src_good)),
                    "source": source_name,
                }
            if source_profiles:
                dominant_source = max(
                    source_profiles.keys(),
                    key=lambda src: pos_total_by_source.get((dim, src), 0.0) + neg_total_by_source.get((dim, src), 0.0),
                )
                entry["source_profiles"] = source_profiles
                entry["source"] = dominant_source
        profile[dim] = entry
    return profile


def _apply_family_adjustment(profile: Dict[str, Dict[str, float]], adjustment: float) -> Dict[str, Dict[str, float]]:
    if not profile:
        return profile
    scale = max(-0.15, min(0.15, float(adjustment) - 0.5))
    adjusted: Dict[str, Dict[str, float]] = {}
    for dim, entry in profile.items():
        good = max(0.55, min(0.98, float(entry["good_match_prob"]) + 0.10 * scale))
        bad = max(0.02, min(0.45, float(entry["bad_match_prob"]) - 0.08 * scale))
        if bad >= good:
            bad = max(0.02, min(good - 0.05, 0.45))
        adjusted[dim] = {
            "good_match_prob": good,
            "bad_match_prob": bad,
            "slip_prob": max(0.01, min(0.25, 1.0 - good)),
        }
        if isinstance(entry.get("source_profiles"), dict):
            adjusted[dim]["source_profiles"] = {
                src: _apply_dim_adjustment(src_entry, adjustment)
                for src, src_entry in entry["source_profiles"].items()
            }
            adjusted[dim]["source"] = entry.get("source", "heuristic")
    return adjusted


def _apply_dim_adjustment(entry: Dict[str, float], adjustment: float) -> Dict[str, float]:
    scale = max(-0.15, min(0.15, float(adjustment) - 0.5))
    good = max(0.55, min(0.98, float(entry["good_match_prob"]) + 0.10 * scale))
    bad = max(0.02, min(0.45, float(entry["bad_match_prob"]) - 0.08 * scale))
    if bad >= good:
        bad = max(0.02, min(good - 0.05, 0.45))
    adjusted = {
        "good_match_prob": good,
        "bad_match_prob": bad,
        "slip_prob": max(0.01, min(0.25, 1.0 - good)),
    }
    if "source" in entry:
        adjusted["source"] = entry["source"]
    if isinstance(entry.get("source_profiles"), dict):
        adjusted["source_profiles"] = {
            src: _apply_dim_adjustment(src_entry, adjustment)
            for src, src_entry in entry["source_profiles"].items()
        }
    return adjusted


def build_factor_calibration_profile(
    *,
    evaluation_packages: Iterable[Any],
    action_history: Optional[Any] = None,
    min_packages: int = 3,
) -> Optional[FactorCalibrationProfile]:
    packages = [pkg for pkg in (evaluation_packages or []) if getattr(pkg, "evidences", None)]
    if len(packages) < max(1, int(min_packages)):
        return None

    prompt_obs = _build_obs_profile(packages, "prompt", default_good=0.58, default_bad=0.46)
    params_obs = _build_obs_profile(packages, "params", default_good=0.90, default_bad=0.16)
    return_obs = _build_obs_profile(
        packages,
        "return",
        default_good=0.62,
        default_bad=0.44,
        source_defaults={
            "ground_truth": (0.95, 0.08),
            "llm_judged": (0.82, 0.22),
            "deterministic": (0.88, 0.18),
            "heuristic": (0.58, 0.46),
        },
    )
    edge_obs = _build_obs_profile(packages, "edge", default_good=0.74, default_bad=0.34)
    structure_obs = _build_obs_profile(packages, "structure", default_good=0.84, default_bad=0.20)

    total_samples = 0.0
    success_samples = 0.0
    for pkg in packages:
        for ev in (getattr(pkg, "evidences", {}) or {}).values():
            total_samples += 1.0
            metrics = getattr(ev, "metrics", {}) or {}
            if _safe_rate(metrics.get("em", 0.0)) >= 1.0 or _safe_rate(metrics.get("f1", 0.0)) >= 0.6:
                success_samples += 1.0

    health_prior = max(0.60, min(0.95, _beta_mean(success_samples, total_samples, alpha=8.0, beta=2.0)))
    action_adjustments: Dict[str, float] = {}
    if action_history is not None:
        for label in ("prompt_explore", "params_explore", "structure_explore"):
            if action_history.label_attempts(label) > 0:
                action_adjustments[label] = float(action_history.label_success_rate(label))
        if "prompt_explore" in action_adjustments:
            prompt_obs = _apply_family_adjustment(prompt_obs, action_adjustments["prompt_explore"])
            edge_obs = _apply_family_adjustment(edge_obs, action_adjustments["prompt_explore"])
        if "params_explore" in action_adjustments:
            params_obs = _apply_family_adjustment(params_obs, action_adjustments["params_explore"])
        if "structure_explore" in action_adjustments:
            structure_obs = _apply_family_adjustment(structure_obs, action_adjustments["structure_explore"])

        dim_mapping = {
            ("prompt_explore", "Prompt", "Binding"): ("prompt_obs", ["input_binding"]),
            ("prompt_explore", "Prompt", "Contract"): ("prompt_obs", ["output_contract"]),
            ("prompt_explore", "Prompt", "Grounding"): ("prompt_obs", ["grounded", "executable"]),
            ("params_explore", "Params", "Length"): ("params_obs", ["not_truncated"]),
            ("params_explore", "Params", "Parse"): ("params_obs", ["format_parseable"]),
            ("prompt_explore", "Edge", "Binding"): ("edge_obs", ["edge_consumed", "dependency_preserved"]),
            ("prompt_explore", "Edge", "Semantic"): ("edge_obs", ["entity_overlap", "semantic_transfer"]),
            ("structure_explore", "Structure", "Coverage"): ("structure_obs", ["role_coverage"]),
            ("structure_explore", "Structure", "Ordering"): ("structure_obs", ["topology_ordering"]),
        }
        family_refs = {
            "prompt_obs": prompt_obs,
            "params_obs": params_obs,
            "edge_obs": edge_obs,
            "structure_obs": structure_obs,
        }
        for (label, component, subtype), (family_name, dims) in dim_mapping.items():
            if action_history.label_component_subtype_attempts(label, component, subtype) <= 0:
                continue
            adj = action_history.label_component_subtype_success_rate(label, component, subtype)
            family_profile = family_refs[family_name]
            for dim in dims:
                if dim in family_profile:
                    family_profile[dim] = _apply_dim_adjustment(family_profile[dim], adj)

    final_obs = {
        "em": {
            "good_match_prob": 0.92,
            "bad_match_prob": 0.08,
        },
        "f1": {
            "good_match_prob": 0.86,
            "bad_match_prob": 0.14,
        },
    }
    step_slip = {
        "prompt": max(0.02, min(0.20, 1.0 - max([v["good_match_prob"] for v in prompt_obs.values()], default=0.58))),
        "params": max(0.02, min(0.20, 1.0 - max([v["good_match_prob"] for v in params_obs.values()], default=0.90))),
        "return": max(0.05, min(0.22, 1.0 - max([v["good_match_prob"] for v in return_obs.values()], default=0.62))),
        "edge": max(0.04, min(0.20, 1.0 - max([v["good_match_prob"] for v in edge_obs.values()], default=0.74))),
    }
    return FactorCalibrationProfile(
        prompt_obs=prompt_obs,
        params_obs=params_obs,
        return_obs=return_obs,
        edge_obs=edge_obs,
        structure_obs=structure_obs,
        final_obs=final_obs,
        step_slip=step_slip,
        health_prior=health_prior,
        sample_count=int(total_samples),
        action_adjustments=action_adjustments,
    )
