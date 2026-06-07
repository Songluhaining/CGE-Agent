"""
Agentic prompt optimizer for EvoAgentX workflow nodes.

The optimizer runs a ReAct-style tool-calling loop: the LLM receives a concise
task description, dynamically queries tools for the information it needs, and
finally emits a structured list of ADD/DELETE/MODIFY/REWRITE operations.

All prompt templates are declared as module-level constants so they can be
audited, versioned, and overridden independently of the agent logic.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Standardized prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a workflow optimization agent that improves the prompts of individual
    nodes inside a multi-step AI workflow.

    You operate in a tool-assisted analysis loop:
    1. Use tools to gather evidence about the node you are optimizing.
    2. Reason about root causes from the evidence.
    3. Propose a targeted list of prompt-editing operations.

    Always call at least one tool before emitting FINAL_OPERATIONS.
    Focus on evidence; avoid speculation without data.
""").strip()

# The opening task message sent to the agent.  Placeholders filled at runtime.
TASK_TEMPLATE = textwrap.dedent("""\
    # Optimization Task

    ## Target
    - Node: `{node_name}`
    - Iteration: {iteration}
    - Failure Probability: {failure_prob:.3f}
    - RCA Diagnosis: component=`{component}`, subtype=`{subtype}`
    - Preferred Edit Style: `{style}`
    - Preferred Operation Family: `{preferred_op_family}`

    ## Hard Constraints (MUST preserve)
    - Input placeholders: {placeholder_warning}
    - Output variable names: {output_names}
    - Do NOT remove the `### Output Format` section.
    - Prefer targeted ADD / MODIFY / DELETE over REWRITE.

    ## Available Tools
    {tool_listing}

    ## Operation Schema
    {op_schema}

    ## Instructions
    1. Call one or more tools to understand the node's current state and history.
    2. After gathering sufficient evidence, emit FINAL_OPERATIONS followed by a
       JSON array of operations.  No other text after the JSON array.

    Start by calling the tools you need.
""").strip()

# Compact listing injected into TASK_TEMPLATE.
TOOL_LISTING_TEMPLATE = textwrap.dedent("""\
    Call a tool by writing exactly:
      TOOL: <tool_name>({{<json-args>}})

    Available tools:
    {tool_entries}
""").strip()

# Single tool entry line.
TOOL_ENTRY_TEMPLATE = "  - {name}({params_summary}): {description}"

# Operation schema block injected into TASK_TEMPLATE.
OP_SCHEMA_TEMPLATE = textwrap.dedent("""\
    Return a JSON array.  Each element is one of:

    ADD    - Insert new content
      {"op":"ADD","position":"before|after|beginning|end","anchor":"<text near insertion, required for before/after>","content":"<new text>","reason":"<why>"}

    DELETE - Remove content
      {"op":"DELETE","target":"<exact text to remove>","reason":"<why>"}

    MODIFY - Replace content
      {"op":"MODIFY","target":"<exact text to find>","replacement":"<new text>","reason":"<why>"}

    REWRITE - Full rewrite (only when major restructuring is necessary)
      {"op":"REWRITE","content":"<complete new prompt>","reason":"<why full rewrite>"}

    Style guidance:
      BINDING_REPAIR   -> strengthen variable binding and upstream/downstream linking
      SCHEMA_HARDEN    -> make output contract and field naming stricter
      GROUNDING_HARDEN -> reduce unsupported claims; require evidence-grounded reasoning
      CHAIN_SYNTHESIS  -> clarify multi-hop combination and bridge reasoning
      DEDUP_SIMPLIFY   -> remove repeated or noisy instructions without losing constraints
      ANSWER_NORMALIZE -> force concise, canonical answer formatting

    Emit your operations like this (no markdown fences, no extra text after):
    FINAL_OPERATIONS: [{...}, {...}]
""").strip()

# Injected after each tool result to continue the loop.
TOOL_RESULT_TEMPLATE = textwrap.dedent("""\
    ## Tool Result: {tool_name}
    {result_text}

    Continue your analysis or emit FINAL_OPERATIONS.
""").strip()

# Final nudge when the agent has used up its analysis turns.
FINAL_INSTRUCTION = textwrap.dedent("""\
    You have gathered sufficient information.
    Now emit your FINAL_OPERATIONS JSON array.
    Remember: preserve input placeholders and output variable names.
    Output only the JSON array after "FINAL_OPERATIONS:".
""").strip()

# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    params_summary: str          # e.g. 'node_name: str, max_records: int = 5'
    handler: Callable[..., str]  # returns a human-readable string


def _build_tool_listing(tools: List[ToolSpec]) -> str:
    entries = "\n".join(
        TOOL_ENTRY_TEMPLATE.format(
            name=t.name,
            params_summary=t.params_summary,
            description=t.description,
        )
        for t in tools
    )
    return TOOL_LISTING_TEMPLATE.format(tool_entries=entries)


# ---------------------------------------------------------------------------
# OptimizationContext
# ---------------------------------------------------------------------------

class OptimizationContext:
    """
    Holds all runtime data and exposes it as callable tool methods.

    Parameters are typed generically so this class works for any workflow,
    not just HotPotQA.
    """

    def __init__(
        self,
        workflow_graph,                         # WorkFlowGraph
        node_stats: Dict[str, Any],             # per-node diagnostic scores
        prompt_history,                         # PromptHistory or None
        action_history,                         # ActionOutcomeHistory or None
        modification_history,                   # ModificationHistory or None
        get_node_prompt: Callable[[str], str],  # callable to avoid circular import
        failure_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        self._wg = workflow_graph
        self._node_stats = node_stats
        self._ph = prompt_history
        self._ah = action_history
        self._mh = modification_history
        self._get_prompt = get_node_prompt
        self._failure_examples = failure_examples or []

    # ------------------------------------------------------------------
    # Tool 1 - node info
    # ------------------------------------------------------------------
    def get_node_info(self, node_name: str) -> str:
        """Return the node's current prompt, inputs, outputs, and description."""
        try:
            node = self._wg.get_node(node_name)
        except Exception:
            return f"ERROR: node '{node_name}' not found in workflow."

        prompt = self._get_prompt(node_name) or "(no prompt)"
        inputs = "\n".join(f"  - {p.name} ({p.type}): {p.description}" for p in node.inputs) or "  (none)"
        outputs = "\n".join(f"  - {p.name} ({p.type}): {p.description}" for p in node.outputs) or "  (none)"

        return textwrap.dedent(f"""\
            Node: {node.name}
            Description: {getattr(node, 'description', '') or '(no description)'}
            Inputs:
            {inputs}
            Outputs:
            {outputs}
            Current Prompt:
            ---
            {prompt}
            ---
        """).strip()

    # ------------------------------------------------------------------
    # Tool 2 - diagnostic scores
    # ------------------------------------------------------------------
    def get_diagnostic_scores(self, node_name: str) -> str:
        """Return all diagnostic metric scores for the node with explanations."""
        stats = self._node_stats.get(node_name, {})
        if not stats:
            return f"No diagnostic scores available for '{node_name}'."

        lines: List[str] = [f"Diagnostic scores for node '{node_name}' (0-1, higher is better):"]
        score_explanations = {
            "input_binding":    "Are input placeholders correctly used?",
            "output_contract":  "Are output variable names mentioned?",
            "grounded":         "Does the prompt avoid hallucination?",
            "executable":       "Can this node run without impossible assumptions?",
            "type_ok":          "Does the output match the expected type?",
            "content_ok":       "Is the output non-empty and meaningful?",
            "task_ok":          "Does the output correctly satisfy the task?",
            "not_truncated":    "Was the output NOT cut off by a token limit?",
            "format_parseable": "Can the output be parsed correctly?",
        }
        for section_name, section_scores in stats.items():
            if not isinstance(section_scores, dict):
                continue
            lines.append(f"\n  [{section_name}]")
            for metric, value in section_scores.items():
                expl = score_explanations.get(metric, "")
                try:
                    fval = float(value)
                    flag = " <- LOW" if fval < 0.5 else ""
                    lines.append(f"    {metric}: {fval:.4f}{flag}  {expl}")
                except (TypeError, ValueError):
                    lines.append(f"    {metric}: {value}  {expl}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 3 - prompt history trend
    # ------------------------------------------------------------------
    def get_node_prompt_history(self, node_name: str, max_records: int = 5) -> str:
        """Return the per-iteration prompt performance trend for a node."""
        if self._ph is None:
            return "No prompt history available."
        text = self._ph.format_history_for_llm(node_name, max_records=max_records)
        return f"Prompt history for '{node_name}':\n{text}"

    # ------------------------------------------------------------------
    # Tool 4 - style stats
    # ------------------------------------------------------------------
    def get_node_style_stats(self, node_name: str) -> str:
        """Return per-style success rates and mean rewards for a node."""
        if self._ah is None:
            return "No action history available."

        from evoagentx.optimizers.rl_workflow_optimizer import _PROMPT_STYLES  # lazy import
        lines: List[str] = [f"Style statistics for node '{node_name}':"]
        for style in _PROMPT_STYLES:
            sr = self._ah.node_style_success_rate("prompt_explore", node_name, style)
            mr = self._ah.node_style_mean_reward("prompt_explore", node_name, style)
            at = self._ah.node_style_attempts("prompt_explore", node_name, style)
            lines.append(f"  {style}: attempts={at:.0f}, success_rate={sr:.4f}, mean_reward={mr:.4f}")

        last_suc = self._ah.last_successful_style("prompt_explore", node_name) or "None"
        failed = self._ah.failed_styles("prompt_explore", node_name) or ["None"]
        lines.append(f"  Last successful style: {last_suc}")
        lines.append(f"  Recently failed styles: {failed}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 5 - workflow structure
    # ------------------------------------------------------------------
    def get_workflow_structure(self) -> str:
        """Return a high-level overview of all nodes and edges in the workflow."""
        nodes = self._wg.nodes
        edges = self._wg.edges
        node_lines = [f"  - {n.name}: {getattr(n, 'description', '')[:80]}" for n in nodes]
        edge_lines = [f"  - {e.source} -> {e.target}" for e in edges]
        return (
            f"Nodes ({len(nodes)}):\n"
            + "\n".join(node_lines)
            + f"\n\nEdges ({len(edges)}):\n"
            + ("\n".join(edge_lines) if edge_lines else "  (none)")
        )

    # ------------------------------------------------------------------
    # Tool 6 - node neighbors (upstream / downstream)
    # ------------------------------------------------------------------
    def get_node_neighbors(self, node_name: str) -> str:
        """Return the immediate upstream and downstream nodes and their I/O."""
        upstream = self._wg.get_node_predecessors(node_name)
        downstream = self._wg.get_node_children(node_name)

        def _fmt_neighbor(name: str) -> str:
            try:
                n = self._wg.get_node(name)
                outs = ", ".join(f"{p.name}({p.type})" for p in n.outputs) or "(none)"
                ins = ", ".join(f"{p.name}({p.type})" for p in n.inputs) or "(none)"
                return f"  {name}: outputs=[{outs}], inputs=[{ins}]"
            except Exception:
                return f"  {name}: (not found)"

        lines = [f"Neighbors of '{node_name}':"]
        lines.append("Upstream (providers):")
        if upstream:
            lines.extend(_fmt_neighbor(n) for n in upstream)
        else:
            lines.append("  (none)")
        lines.append("Downstream (consumers):")
        if downstream:
            lines.extend(_fmt_neighbor(n) for n in downstream)
        else:
            lines.append("  (none)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 7 - global best patterns
    # ------------------------------------------------------------------
    def get_global_best_patterns(self) -> str:
        """Return the globally most effective edit patterns across all nodes."""
        if self._mh is None:
            return "No modification history available."
        rows = self._mh.summarize_pattern_performance(accepted=True, max_patterns=5)
        if not rows:
            return "No accepted patterns recorded yet."
        lines = ["Top accepted edit patterns (highest estimated_full_f1 gain):"]
        for i, row in enumerate(rows, 1):
            edit = row.get("edit", {})
            tgt = row.get("target", {})
            lines.append(
                f"  {i}. kind={edit.get('kind','?')} style={edit.get('style','?')} "
                f"op={edit.get('op_family','?')} | "
                f"component={tgt.get('component','?')} subtype={tgt.get('subtype','?')} | "
                f"accept_rate={row.get('accept_rate',0):.2f} "
                f"mean_f1_delta={row.get('mean_estimated_full_f1_delta',0):+.4f} "
                f"attempts={row.get('attempts',0):.0f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 8 - global failure patterns
    # ------------------------------------------------------------------
    def get_global_failure_patterns(self) -> str:
        """Return the most common failure patterns to help avoid repeating mistakes."""
        if self._mh is None:
            return "No modification history available."
        patterns = self._mh.summarize_failure_patterns(max_patterns=8)
        if not patterns:
            return "No failure patterns recorded yet."
        lines = ["Common failure patterns (avoid these):"]
        lines.extend(f"  - {p}" for p in patterns)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 9 - node-specific edit history
    # ------------------------------------------------------------------
    def get_node_edit_history(self, node_name: str, max_records: int = 5) -> str:
        """Return recent ModificationRecords for a specific node."""
        if self._mh is None:
            return "No modification history available."
        records = [r for r in reversed(self._mh.records) if r.target_node_name == node_name]
        records = records[:max_records]
        if not records:
            return f"No edit history for node '{node_name}'."

        lines = [f"Recent edits for node '{node_name}' (most recent first):"]
        for r in records:
            lines.append(
                f"  iter={r.iteration} | kind={r.edit_kind} style={r.style} op={r.op_family} | "
                f"status={r.materialization_status}/{r.validation_status} accepted={r.accepted} | "
                f"f1_delta={r.estimated_full_f1_delta:+.4f} utility_delta={r.utility_delta:+.4f}"
            )
            if r.rationale:
                lines.append(f"    rationale: {r.rationale[:120]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool 10 - failure examples with fault localization
    # ------------------------------------------------------------------
    def get_failure_examples(self, node_name: str = "", max_examples: int = 5) -> str:
        """Return recent failure examples with per-node fault localization."""
        if not self._failure_examples:
            return "No failure examples available for this iteration."
        examples = list(self._failure_examples)
        if node_name:
            node_first = [e for e in examples if e.get("first_fault_node") == node_name]
            node_other = [e for e in examples if e.get("first_fault_node") != node_name]
            examples = (node_first + node_other)[:max_examples]
        else:
            examples = examples[:max_examples]

        lines: List[str] = [f"Failure examples (sorted by relevance to \'{node_name or 'all'}\'):"]
        for i, ex in enumerate(examples, 1):
            question = str(ex.get("question", ""))[:300]
            gold = str(ex.get("gold_answer", ""))
            predicted = str(ex.get("predicted_answer", ""))
            f1 = float(ex.get("f1", 0.0))
            mode = ex.get("failure_mode", "")
            first_fault = ex.get("first_fault_node", "")
            lines.append(f"  [{i}] Q: {question}")
            lines.append(f"      Gold: {gold!r}  |  Predicted: {predicted!r}")
            lines.append(f"      F1={f1:.3f}  failure_mode={mode}  first_fault_node={first_fault}")
            gold_in = ex.get("gold_in_node_output", {})
            if gold_in:
                presence = ", ".join(("YES" if v else "no") + "=" + n for n, v in gold_in.items())
                lines.append(f"      Gold in node outputs: {presence}")
            node_outputs = ex.get("node_outputs", {})
            if node_outputs and node_name and node_name in node_outputs:
                lines.append(f"      This node\'s output: {node_outputs[node_name][:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build tool registry
    # ------------------------------------------------------------------
    def build_tool_registry(self) -> Dict[str, ToolSpec]:
        return {
            "get_node_info": ToolSpec(
                name="get_node_info",
                description="Get the node's current prompt, inputs, outputs, and description.",
                params_summary="node_name: str",
                handler=lambda args: self.get_node_info(args.get("node_name", "")),
            ),
            "get_diagnostic_scores": ToolSpec(
                name="get_diagnostic_scores",
                description="Get all diagnostic metric scores with explanations.",
                params_summary="node_name: str",
                handler=lambda args: self.get_diagnostic_scores(args.get("node_name", "")),
            ),
            "get_node_prompt_history": ToolSpec(
                name="get_node_prompt_history",
                description="Get the per-iteration prompt performance trend.",
                params_summary="node_name: str, max_records: int = 5",
                handler=lambda args: self.get_node_prompt_history(
                    args.get("node_name", ""), int(args.get("max_records", 5))
                ),
            ),
            "get_node_style_stats": ToolSpec(
                name="get_node_style_stats",
                description="Get per-style success rates and mean rewards for the node.",
                params_summary="node_name: str",
                handler=lambda args: self.get_node_style_stats(args.get("node_name", "")),
            ),
            "get_workflow_structure": ToolSpec(
                name="get_workflow_structure",
                description="Get a high-level overview of all nodes and edges.",
                params_summary="(no args)",
                handler=lambda args: self.get_workflow_structure(),
            ),
            "get_node_neighbors": ToolSpec(
                name="get_node_neighbors",
                description="Get upstream/downstream nodes and their I/O for context.",
                params_summary="node_name: str",
                handler=lambda args: self.get_node_neighbors(args.get("node_name", "")),
            ),
            "get_global_best_patterns": ToolSpec(
                name="get_global_best_patterns",
                description="Get the globally most effective edit patterns.",
                params_summary="(no args)",
                handler=lambda args: self.get_global_best_patterns(),
            ),
            "get_global_failure_patterns": ToolSpec(
                name="get_global_failure_patterns",
                description="Get common failure patterns to avoid.",
                params_summary="(no args)",
                handler=lambda args: self.get_global_failure_patterns(),
            ),
            "get_node_edit_history": ToolSpec(
                name="get_node_edit_history",
                description="Get recent edit records for a specific node.",
                params_summary="node_name: str, max_records: int = 5",
                handler=lambda args: self.get_node_edit_history(
                    args.get("node_name", ""), int(args.get("max_records", 5))
                ),
            ),
            "get_failure_examples": ToolSpec(
                name="get_failure_examples",
                description="Get recent failure examples with fault localization showing which node lost the gold answer.",
                params_summary="node_name: str, max_examples: int = 5",
                handler=lambda args: self.get_failure_examples(
                    args.get("node_name", ""), int(args.get("max_examples", 5))
                ),
            ),
        }


# ---------------------------------------------------------------------------
# Tool-call parsing helpers
# ---------------------------------------------------------------------------

# Matches:  TOOL: tool_name({"key": "val"})
_TOOL_CALL_RE = re.compile(
    r"TOOL:\s*(\w+)\s*\((\{.*?\}|\[\])\)",
    re.DOTALL,
)

# Matches:  FINAL_OPERATIONS: [...]
_FINAL_OPS_RE = re.compile(
    r"FINAL_OPERATIONS\s*:\s*(\[.*\])",
    re.DOTALL,
)


def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (tool_name, args_dict) from the first TOOL: line found, or None."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    tool_name = m.group(1).strip()
    raw_args = m.group(2).strip()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return tool_name, args


def _parse_final_operations(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return the operations list from FINAL_OPERATIONS: [...], or None."""
    m = _FINAL_OPS_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        ops = json.loads(raw)
        if isinstance(ops, list):
            return ops
    except json.JSONDecodeError:
        pass
    return None


def _execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    registry: Dict[str, ToolSpec],
) -> str:
    spec = registry.get(tool_name)
    if spec is None:
        available = ", ".join(registry.keys())
        return f"ERROR: unknown tool '{tool_name}'. Available: {available}"
    try:
        return spec.handler(args)
    except Exception as exc:
        return f"ERROR calling {tool_name}: {exc}"


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_optimizer_agent(
    llm,
    context: OptimizationContext,
    node_name: str,
    failure_prob: float,
    iteration: int,
    component: str = "",
    subtype: str = "",
    style: str = "",
    preferred_op_family: str = "",
    attempt_idx: int = 0,
    previous_failed_ops: Optional[List[str]] = None,
    max_turns: int = 5,
) -> Optional[List[Dict[str, Any]]]:
    """
    Run the agentic optimizer loop for a single node.

    Returns a list of operation dicts (may be empty) or None on failure.
    The caller is responsible for applying the operations.

    The loop:
      1. Build the opening task prompt.
      2. Send to LLM; check response for TOOL: or FINAL_OPERATIONS:.
      3. If TOOL:, execute it and append the result; continue.
      4. If FINAL_OPERATIONS:, return parsed ops.
      5. After max_turns without FINAL_OPERATIONS, send a nudge and parse one last time.
    """
    registry = context.build_tool_registry()
    tool_listing = _build_tool_listing(list(registry.values()))

    # Resolve node I/O constraints
    try:
        node = context._wg.get_node(node_name)
        input_names = [p.name for p in node.inputs]
        output_names = [p.name for p in node.outputs]
    except Exception:
        input_names = []
        output_names = []

    placeholder_warning = ", ".join(f"{{{n}}}" for n in input_names) or "(none)"
    output_names_str = ", ".join(output_names) or "(none)"

    # Add retry hint if this is a retry attempt
    retry_hint = ""
    if attempt_idx > 0 and previous_failed_ops:
        failed_str = ", ".join(str(op) for op in previous_failed_ops[-6:])
        retry_hint = (
            f"\n\n## Retry Context\n"
            f"This is retry attempt #{attempt_idx}. Previously failed operations: {failed_str}\n"
            f"Propose a meaningfully different strategy."
        )

    task_message = TASK_TEMPLATE.format(
        node_name=node_name,
        iteration=iteration,
        failure_prob=failure_prob,
        component=component or "Prompt",
        subtype=subtype or "Prompt",
        style=style or "UNSPECIFIED",
        preferred_op_family=preferred_op_family or "MODIFY",
        placeholder_warning=placeholder_warning,
        output_names=output_names_str,
        tool_listing=tool_listing,
        op_schema=OP_SCHEMA_TEMPLATE,
    ) + retry_hint

    # Accumulated conversation (single string, pseudo-multi-turn)
    conversation = task_message
    turns_used = 0

    while turns_used < max_turns:
        # Call LLM
        try:
            llm_out = llm.generate(prompt=conversation, system_message=SYSTEM_PROMPT)
            response = str(getattr(llm_out, "content", llm_out)).strip()
        except Exception as e:
            print(f">>> [OptimizerAgent] node={node_name} LLM call failed: {e}")
            return None

        print(f">>> [OptimizerAgent] node={node_name} turn={turns_used + 1} response preview: {response[:120]!r}")

        # Check for FINAL_OPERATIONS first (agent may combine tool + final in one turn)
        ops = _parse_final_operations(response)
        if ops is not None:
            print(f">>> [OptimizerAgent] node={node_name} got {len(ops)} operations after {turns_used + 1} turns")
            return ops

        # Check for tool call
        tool_call = _parse_tool_call(response)
        if tool_call is not None:
            tool_name, tool_args = tool_call
            result_text = _execute_tool(tool_name, tool_args, registry)
            print(f">>> [OptimizerAgent] node={node_name} called {tool_name}({tool_args})")
            # Append assistant response + tool result
            conversation += (
                f"\n\n## Assistant (Turn {turns_used + 1})\n{response}"
                + f"\n\n"
                + TOOL_RESULT_TEMPLATE.format(tool_name=tool_name, result_text=result_text)
            )
            turns_used += 1
            continue

        # No tool call and no final ops - LLM may have emitted free text
        conversation += f"\n\n## Assistant (Turn {turns_used + 1})\n{response}"
        turns_used += 1

    # Max turns reached - send final nudge
    conversation += f"\n\n## System\n{FINAL_INSTRUCTION}"
    try:
        llm_out = llm.generate(prompt=conversation, system_message=SYSTEM_PROMPT)
        final_response = str(getattr(llm_out, "content", llm_out)).strip()
    except Exception as e:
        print(f">>> [OptimizerAgent] node={node_name} final LLM call failed: {e}")
        return None

    ops = _parse_final_operations(final_response)
    if ops is not None:
        print(f">>> [OptimizerAgent] node={node_name} got {len(ops)} operations after final nudge")
        return ops

    print(f">>> [OptimizerAgent] node={node_name} failed to parse FINAL_OPERATIONS after nudge")
    return None
