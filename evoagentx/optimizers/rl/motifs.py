"""Workflow structural motifs library.

Each motif is a dataset-agnostic topology pattern described by **symptoms**
(when it should be considered) and **rationale** (why it usually helps), so
the planner LLM can decide whether to adopt it for the current benchmark
instead of the framework hard-coding any per-dataset choice.

The library has two roles:

  1. As a *vocabulary* shown unconditionally in the planner prompt so the
     LLM always knows what topological alternatives exist beyond a single
     monolithic solver. This is documentation, not a directive.

  2. As a *trigger-driven recommendation* attached to ``planner_context``
     under ``recommended_motifs`` only when the optimizer's run-time state
     matches the motif's symptoms. The trigger logic looks at structural
     state (node count, edit-kind distribution, no-improve count) - not at
     the dataset name - so the same rule fires for any benchmark whose
     trajectory shows the same pattern.

Adding a motif: drop a new entry into ``MOTIFS`` and (optionally) one
clause into ``recommend_motifs``. Both are deliberately small so the
planner JSON stays well under the prompt budget.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------------------
# Motif vocabulary. Keep ``what`` short enough to render inline in the
# planner prompt; ``when`` is the symptom that should make the planner
# consider the motif; ``why`` is the mechanism justifying the symptom.
# ---------------------------------------------------------------------------
MOTIFS: Dict[str, Dict[str, str]] = {
    "single_solver": {
        "what": "1 node: goal -> answer.",
        "when": "Task is simple or baseline F1 is already high; minimal-latency default.",
        "why": "Removes orchestration overhead; relies entirely on the model's single-pass quality.",
    },
    "framing_solver_finalize": {
        "what": "3 linear nodes: framing -> solver -> finalize.",
        "when": "Output format is strict and parsing/format errors dominate the failures.",
        "why": "Separates reasoning from formatting so each stage can be prompt-tuned independently.",
    },
    "parallel_voting": {
        "what": "N parallel solvers (>=3) -> 1 aggregator that picks/votes the final answer.",
        "when": "Single-node prompt edits have saturated AND outputs vary across reruns of the same prompt.",
        "why": "Majority voting / aggregation reduces single-call sampling noise; effective when the model is right on average but not consistently.",
    },
    "solver_critic_revise": {
        "what": "3 nodes: solver -> critic (audits the draft) -> revise (rewrites using critique).",
        "when": "Errors are detectable from the output alone (format violations, missing pieces, contradicting evidence).",
        "why": "A self-correction loop catches mistakes the solver can recognize but does not avoid in a single pass.",
    },
    "decompose_synthesize": {
        "what": "decomposer -> N parallel sub-solvers (one per sub-question) -> synthesizer.",
        "when": "Goal contains multiple sub-questions or requires multi-hop / multi-step composition.",
        "why": "Localizes errors per sub-question so one bad sub-answer does not poison the rest.",
    },
}


# ---------------------------------------------------------------------------
# Generic schema skeletons. Used as illustrative ``edit.new_workflow``
# examples; placeholders are deliberately abstract (solver_1/aggregator),
# never benchmark-specific (no entry_point, no test_list, no boxed-answer
# affordances).
# ---------------------------------------------------------------------------
PARALLEL_VOTING_SKELETON: Dict[str, Any] = {
    "nodes": [
        {
            "name": "solver_1",
            "description": "Independent solver attempt #1; reads the goal directly.",
            "inputs": [{"name": "goal", "type": "string", "description": "Task statement.", "required": True}],
            "outputs": [{"name": "candidate_1", "type": "string", "description": "Candidate answer #1.", "required": True}],
        },
        {
            "name": "solver_2",
            "description": "Independent solver attempt #2; reads the goal directly.",
            "inputs": [{"name": "goal", "type": "string", "description": "Task statement.", "required": True}],
            "outputs": [{"name": "candidate_2", "type": "string", "description": "Candidate answer #2.", "required": True}],
        },
        {
            "name": "solver_3",
            "description": "Independent solver attempt #3; reads the goal directly.",
            "inputs": [{"name": "goal", "type": "string", "description": "Task statement.", "required": True}],
            "outputs": [{"name": "candidate_3", "type": "string", "description": "Candidate answer #3.", "required": True}],
        },
        {
            "name": "aggregator",
            "description": "Compares the candidates and emits the final answer (e.g. majority vote, or best by an explicit criterion).",
            "inputs": [
                {"name": "candidate_1", "type": "string", "description": "From solver_1.", "required": True},
                {"name": "candidate_2", "type": "string", "description": "From solver_2.", "required": True},
                {"name": "candidate_3", "type": "string", "description": "From solver_3.", "required": True},
            ],
            "outputs": [{"name": "answer", "type": "string", "description": "Final answer.", "required": True}],
        },
    ],
    "edges": [
        {"source": "solver_1", "target": "aggregator"},
        {"source": "solver_2", "target": "aggregator"},
        {"source": "solver_3", "target": "aggregator"},
    ],
}


# ---------------------------------------------------------------------------
# Trigger-driven recommendation. Inputs are run-time signals, NOT dataset
# names: a recommendation only fires when the trajectory shows symptoms
# matching the motif's ``when`` condition. Returns at most a few names so
# the planner does not get overwhelmed.
# ---------------------------------------------------------------------------
def recommend_motifs(
    *,
    no_improve_count: int,
    node_count: int,
    edit_kind_attempts: Dict[str, int],
    edit_kind_accepts: Dict[str, int],
    failure_mode_distribution: Dict[str, int],
) -> List[str]:
    """Return motif names worth surfacing this iteration.

    Args:
        no_improve_count: consecutive non-improving iterations on the
            current incumbent.
        node_count: number of nodes in the incumbent workflow.
        edit_kind_attempts: how many candidates of each kind have been
            attempted across history (``prompt_edit`` / ``structure_edit``
            / ``params_edit``).
        edit_kind_accepts: how many of each kind were accepted.
        failure_mode_distribution: counts of failure_mode labels (e.g.
            ``empty_answer`` / ``wrong_value`` / ``wrong_logic``) over the
            most recent baseline failure examples.

    Order of returned names is significance-first; callers may slice.
    """
    out: List[str] = []
    prompt_attempts = int(edit_kind_attempts.get("prompt_edit", 0))
    prompt_accepts = int(edit_kind_accepts.get("prompt_edit", 0))
    struct_attempts = int(edit_kind_attempts.get("structure_edit", 0))
    struct_accepts = int(edit_kind_accepts.get("structure_edit", 0))

    # Symptom: prompt-only saturation. Accepts are rare relative to attempts
    # AND we have stalled - typical of a single-node workflow plateauing
    # at the model's per-call ceiling, where averaging across reruns is
    # the cheapest way to break through.
    if no_improve_count >= 2 and prompt_attempts >= 4 and prompt_accepts <= 1:
        out.append("parallel_voting")
        out.append("solver_critic_revise")

    # Symptom: still a monolithic solver and stuck. Even if structure_edit
    # has not been tried yet, splitting reasoning from formatting is the
    # smallest topological step a planner can take.
    if node_count <= 1 and no_improve_count >= 1:
        if "framing_solver_finalize" not in out:
            out.append("framing_solver_finalize")

    # Symptom: high "wrong_logic" or "wrong_value" share - errors that look
    # like reasoning gaps rather than format issues. Decomposition helps
    # when the goal can be split.
    total_failures = sum(int(v) for v in (failure_mode_distribution or {}).values())
    if total_failures >= 4:
        reasoning_failures = sum(
            int(failure_mode_distribution.get(label, 0))
            for label in ("wrong_logic", "wrong_value", "wrong_entity", "partial_overlap")
        )
        if reasoning_failures / max(1, total_failures) >= 0.6 and "decompose_synthesize" not in out:
            out.append("decompose_synthesize")

    # Symptom: structure_edit has been tried multiple times with no accept.
    # Don't push more structural motifs - the planner should focus on
    # prompt/params instead.
    if struct_attempts >= 3 and struct_accepts == 0:
        out = [m for m in out if m not in ("parallel_voting", "solver_critic_revise", "decompose_synthesize")]

    return out


def render_motifs_for_prompt(motif_names: Sequence[str]) -> List[Dict[str, str]]:
    """Compact projection of selected motifs for the planner JSON. Returns
    only the names provided, in input order, with their what/when/why."""
    rendered: List[Dict[str, str]] = []
    for name in motif_names:
        spec = MOTIFS.get(str(name))
        if not spec:
            continue
        rendered.append({"name": name, "what": spec["what"], "when": spec["when"], "why": spec["why"]})
    return rendered
