import json
import re
from typing import Optional, List, Dict, Any, Set
from pydantic import Field, PositiveInt 

import time
from ..core.logging import logger
from ..core.module import BaseModule
# from ..core.base_config import Parameter
from ..core.message import Message, MessageType
from ..models.base_model import BaseLLM
from ..agents.agent import Agent
from ..agents.task_planner import TaskPlanner
from ..agents.agent_generator import AgentGenerator
from ..agents.workflow_reviewer import WorkFlowReviewer
from ..actions.task_planning import TaskPlanningOutput
from ..actions.agent_generation import AgentGenerationOutput
from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode, WorkFlowEdge
from ..tools.tool import Toolkit

def _param_to_dict(p):
    if isinstance(p, dict):
        d = dict(p)
    else:
        d = {}
        for k in ("name", "type", "description", "required"):
            if hasattr(p, k):
                d[k] = getattr(p, k)
    d.setdefault("required", True)
    d.setdefault("description", "")
    d.setdefault("type", "string")
    return d


def _parse_fallback_goal_contract(goal: str):
    """Lightweight extractor for parse_mode + verbatim rules from goal text.

    Mirrors `_parse_goal_contract` in rl_workflow_optimizer but kept local
    so the workflow generator does not import from optimizers (avoids a
    cycle). Returns dict with keys: parse_mode, verbatim_rules, terminal_output_name.
    """
    contract = {"parse_mode": None, "verbatim_rules": [], "terminal_output_name": None}
    if not goal:
        return contract
    pm = re.search(r'parse_mode["\'`]?\s*:\s*["\'`]?(title|str|json)\b', goal, re.IGNORECASE)
    if pm:
        contract["parse_mode"] = pm.group(1).lower()
    for m in re.finditer(r'MUST contain this sentence literally:?\s*`([^`]+)`', goal, re.IGNORECASE):
        s = m.group(1).strip()
        if s:
            contract["verbatim_rules"].append(s)
    for m in re.finditer(r'MUST include this exact one-line anchor:?\s*`([^`]+)`', goal, re.IGNORECASE):
        s = m.group(1).strip()
        if s and s not in contract["verbatim_rules"]:
            contract["verbatim_rules"].append(s)
    out = re.search(r'name`?\s+is\s+exactly\s+`?(\w+)`?\s*\(?lowercase', goal, re.IGNORECASE)
    if out:
        contract["terminal_output_name"] = out.group(1).lower()
    return contract


def _build_fallback_agent_for_node(node, goal: str = "", dataset_name: str = ""):
    inputs = [_param_to_dict(x) for x in (getattr(node, "inputs", []) or [])]
    outputs = [_param_to_dict(x) for x in (getattr(node, "outputs", []) or [])]

    node_name = getattr(node, "name", "node")
    description = getattr(node, "description", "Complete the task.")

    # Passthrough hint for simple identity nodes
    passthrough_hint = ""
    if len(inputs) == 1 and len(outputs) == 1:
        if inputs[0]["name"] in ("goal", "query") and outputs[0]["name"] in ("query", "goal"):
            passthrough_hint = "\n5. IMPORTANT: Do NOT rewrite. Preserve the original meaning and wording as much as possible.\n"

    contract = _parse_fallback_goal_contract(goal)
    declared_mode = contract.get("parse_mode")

    def _needs_structured_output() -> bool:
        # Goal-declared parse_mode takes precedence: if the contract says
        # title or str, the terminal node MUST emit non-JSON, regardless of
        # the output field name. Only fall back to the legacy heuristic when
        # the goal text is silent on parse_mode.
        if declared_mode == "json":
            return True
        if declared_mode in ("title", "str"):
            return False
        if len(outputs) > 1:
            return True
        for out in outputs:
            out_type = str(out.get("type", "") or "").lower().strip()
            if out_type.startswith("list") or out_type.startswith("dict"):
                return True
        if len(outputs) == 1:
            out_name = str(outputs[0].get("name", "") or "").lower().strip()
            if out_name and out_name not in {"answer", "result", "output", "content", "text", "final_answer"}:
                return True
        return False

    def _json_value_hint(param_type: str) -> str:
        normalized = str(param_type or "").lower().strip()
        if normalized.startswith("list"):
            return '["..."]'
        if normalized.startswith("dict"):
            return '{{"key": "..."}}'
        if normalized in {"bool", "boolean"}:
            return 'true'
        return '"..."'

    structured_output = _needs_structured_output()

    input_lines = "\n".join([f"- {{{i['name']}}} ({i['type']}): {i.get('description', '')}" for i in inputs])
    output_keys = "\n".join([f'- "{o["name"]}"' for o in outputs]) or '- "answer"'
    json_skeleton = "{{\n" + ",\n".join(
        [f'  "{o["name"]}": {_json_value_hint(o.get("type", ""))}' for o in outputs]
    ) + "\n}}"
    plain_output_name = outputs[0]["name"] if outputs else "answer"

    if structured_output:
        output_format_block = (
            "### Output Format\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            f"{output_keys}\n"
            "Use this JSON shape:\n"
            f"{json_skeleton}\n"
            "Do not output any other text, markdown, or explanation outside this JSON object."
        )
    elif declared_mode == "title":
        # title parser extracts the body under `## <output_name>`. Anchor the
        # response with that single heading and append any verbatim contract
        # sentences (rule 5 / few-shot anchor in MATH-style goals).
        verbatim_block = ""
        if contract.get("verbatim_rules"):
            verbatim_block = "\n\n# Contract Rules (verbatim from goal)\n" + "\n".join(
                f"- {r}" for r in contract["verbatim_rules"]
            )
        output_format_block = (
            "### Output Format\n"
            "Your final output MUST follow this exact two-section template and contain no other markdown headings:\n\n"
            f"## {plain_output_name}\n"
            f"<the value for `{plain_output_name}`>\n"
            + verbatim_block
        )
    else:
        output_format_block = (
            "### Output Format\n"
            f"Return ONLY the value for `{plain_output_name}` as plain text.\n"
            "Do not wrap the output in JSON, markdown, quotes, labels, or explanation."
        )

    # Inject benchmark-specific imperative instructions. When the workflow goal
    # matches a supported benchmark (DROP, HotpotQA, MATH, GSM8K, HumanEval, MBPP)
    # and the node matches a known role (task_analysis, evidence_extract, reasoning,
    # answer_extract), _derive_domain_guidance returns imperative bullets that are
    # dropped into the prompt as an explicit guidance section. This ensures the
    # deterministic fallback path carries the same domain expertise as the LLM path,
    # so a validator-rejected LLM generation does not cause a silent quality collapse.
    domain_bullets = _derive_domain_guidance(goal, node, dataset_name=dataset_name) if goal else []
    if domain_bullets:
        domain_lines = "\n".join(f"- {b}" for b in domain_bullets)
        domain_section = f"\n### Domain-Specific Guidance\n{domain_lines}\n"
    else:
        domain_section = ""

    # When domain-specific guidance is available, the Instructions section anchors
    # on it explicitly ("follow the Domain-Specific Guidance below step-by-step")
    # rather than relying on generic boilerplate that the model can skim. When no
    # domain guidance applies (unknown benchmark or unknown node role), the
    # instructions fall back to the original generic phrasing so existing non-
    # benchmark workflows are unaffected.
    if domain_bullets:
        instructions_body = (
            "1. Read the inputs listed in the '### Inputs' section below.\n"
            "2. Follow EVERY rule in the '### Domain-Specific Guidance' section below step-by-step. "
            "Those rules are the authoritative specification for this node and override any generic habits.\n"
            "3. Ground every decision ONLY in the provided inputs; never invent facts or use external knowledge.\n"
            "4. Keep the output minimal and directly usable by downstream nodes — no preamble, no explanation, "
            "no markdown, no JSON wrapping unless the output format explicitly requires it."
        )
    else:
        instructions_body = (
            "1. Read all inputs provided below.\n"
            "2. Analyze the inputs carefully to produce the required outputs.\n"
            "3. Use ONLY the information from the provided inputs; do not use external knowledge.\n"
            "4. Keep outputs minimal and directly usable by downstream steps."
        )
    prompt = f"""### Objective
{description}

### Instructions
{instructions_body}{passthrough_hint}
{domain_section}
### Inputs
{input_lines}

{output_format_block}"""

    agent = {
        "name": f"{node_name}_fallback_agent",
        "description": f"Fallback agent for node {node_name} (auto-filled because no agent was generated).",
        "inputs": inputs,
        "outputs": outputs,
        "prompt": prompt,
    }
    if structured_output:
        agent["parse_mode"] = "json"
    elif declared_mode == "title":
        agent["parse_mode"] = "title"
    elif declared_mode == "str":
        agent["parse_mode"] = "str"
    return agent


def _derive_domain_guidance(goal: str, node, dataset_name: str = "") -> List[str]:
    """Return imperative-voice, benchmark-specific instruction bullets for a node.

    This is the single source of truth for per-benchmark, per-role domain knowledge.
    Both the LLM agent-generator suggestion (_derive_node_agent_suggestion) and the
    deterministic fallback agent (_build_fallback_agent_for_node) consume these
    bullets so the two paths deliver the same domain expertise and a validator-
    rejected LLM generation does not cause a silent quality collapse.

    Benchmark detection is driven by tokens in the workflow goal; role detection is
    driven by output names, node name, and node description. Bullets are written in
    imperative voice so they can be dropped directly into an agent prompt as part
    of the Instructions section.
    """
    outputs = [_param_to_dict(x) for x in (getattr(node, "outputs", []) or [])]
    inputs  = [_param_to_dict(x) for x in (getattr(node, "inputs",  []) or [])]
    node_name = str(getattr(node, "name", "") or "").lower()
    node_desc = str(getattr(node, "description", "") or "").lower()
    out_names_lower = {o["name"].lower() for o in outputs}
    in_names_lower  = {i["name"].lower() for i in inputs}

    # ---- Node role detection ----
    _is_question_parse = (
        any(kw in out_names_lower for kw in {"question_intent", "question_focus", "question_schema", "question_type"})
        or any(kw in node_name for kw in ("question_pars", "question_und", "parse_question", "intent"))
        or any(kw in node_desc for kw in ("entity type", "answer type", "bridge relation", "multi-hop logic"))
    )
    # NEW: covers DROP task_analysis which classifies (reasoning_type, answer_type)
    # and similar classification-style nodes for other benchmarks.
    _is_task_analysis = (
        any(kw in out_names_lower for kw in {"reasoning_type", "answer_type", "operation", "op_type", "task_type"})
        or any(kw in node_name for kw in ("task_analysis", "task_classif", "classify_task", "identify_reasoning", "identify_operation"))
    )
    _is_evidence_extract = (
        any(kw in out_names_lower for kw in {"evidence_quotes", "supporting_quotes", "evidence", "quotes", "supporting_facts", "grounded_facts", "facts", "extracted_spans", "spans"})
        or any(kw in node_name for kw in ("evidence", "supporting_fact", "extract", "retriev", "fact_extract"))
        or any(kw in node_desc for kw in ("extract", "verbatim", "sentences from", "context passages"))
    )
    _is_reasoning = (
        any(kw in out_names_lower for kw in {"reasoning_chain", "reasoning_summary", "reasoning", "chain", "derivation", "raw_result", "computed_value", "intermediate_result"})
        or any(kw in node_name for kw in ("reasoning", "chain", "multi_hop", "infer", "compute", "calculate", "discrete_reasoning"))
        or any(kw in node_desc for kw in ("chain the evidence", "step-by-step", "multi-hop", "derive the answer", "apply the identified reasoning", "compute the"))
    )
    _is_answer_extract = (
        out_names_lower == {"answer"}
        or (len(out_names_lower) == 1 and "answer" in out_names_lower)
        or any(kw in node_name for kw in ("answer_extract", "answer_norm", "answer_gen", "final_answer", "answer_finaliz", "finaliz"))
        or any(kw in node_desc for kw in ("minimal answer", "shortest", "noun phrase", "normalize the", "format the final answer", "final answer"))
    )

    # ---- Benchmark detection ----
    _ds = (dataset_name or "").strip().lower()
    if _ds:
        _is_hotpotqa = _ds == "hotpotqa"
        _is_math = _ds == "math"
        _is_gsm8k = _ds == "gsm8k"
        _is_humaneval = _ds == "humaneval"
        _is_mbpp = _ds == "mbpp"
        _is_drop = _ds == "drop"
    else:
        goal_upper = (goal or "").upper()
        _is_hotpotqa = (
            "HOTPOTQA" in goal_upper
            or "HOT POT" in goal_upper
            or "MULTI-HOP" in goal_upper.replace("_", "-")
        )
        _is_math = (
            "MATH500" in goal_upper
            or "\\BOXED" in goal_upper
            or "BOXED{" in goal_upper
            or ("MATH" in goal_upper and ("BOXED" in goal_upper or "LATEX" in goal_upper))
        )
        _is_gsm8k = "GSM8K" in goal_upper or "GRADE SCHOOL MATH" in goal_upper
        _is_humaneval = "HUMANEVAL" in goal_upper or "HUMAN EVAL" in goal_upper
        _is_mbpp = "MBPP" in goal_upper or "MOSTLY BASIC PYTHON" in goal_upper
        _is_drop = "DROP" in goal_upper and (
            "DISCRETE REASONING" in goal_upper
            or "PASSAGE" in goal_upper
            or "REF_TEXT" in goal_upper
        )

    bullets: List[str] = []

    if _is_hotpotqa:
        if _is_question_parse:
            bullets.append(
                "HotPotQA question parsing: "
                "(1) Identify the expected answer type (person, place, date, occupation, organization, work). "
                "(2) Identify the multi-hop bridge relation (e.g., 'person who wrote X', 'city where Y lived'). "
                "(3) Output a concise one-sentence description combining both. "
                "Example: 'person who directed the film mentioned in passage 1 and was born in the city in passage 2'."
            )
        elif _is_answer_extract:
            bullets.append(
                "HotPotQA answer extraction: "
                "(1) Identify the LAST clause or conclusion in the reasoning chain. "
                "(2) Extract ONLY the minimal noun phrase (1-5 words) - the named entity, date, or key term. "
                "(3) Strip all articles (a, an, the), adjectives, verbs, and punctuation from the answer. "
                "(4) Return ONLY the bare phrase with no explanation, no JSON, no markdown. "
                "Examples: 'Anne Perry', 'novelist', '1985', 'University of Toronto'. "
                "NEVER return a full sentence."
            )
        elif _is_reasoning:
            bullets.append(
                "HotPotQA multi-hop reasoning: "
                "(1) Number each reasoning step. "
                "(2) Explicitly cite which evidence sentence supports each step. "
                "(3) Connect the chain: 'Passage A states X. Passage B says Y is X. Therefore Z is the answer.' "
                "(4) Conclude with a clear statement of the answer concept. "
                "Ground every step ONLY in the provided evidence quotes; do not use external knowledge."
            )
        elif _is_evidence_extract:
            bullets.append(
                "HotPotQA evidence extraction: "
                "(1) Read all context passages. "
                "(2) Select ONLY 2-5 verbatim sentences that directly support the multi-hop chain implied by the question intent. "
                "(3) Exclude distractor sentences, summaries, and paraphrases. "
                "(4) Prefer sentences that together form a logical bridge to the answer. "
                "Output a list of exact quoted sentences, NOT summaries."
            )
    elif _is_math:
        if _is_reasoning:
            bullets.append(
                "MATH reasoning: "
                "(1) Parse the problem into given quantities and the target expression. "
                "(2) Perform explicit step-by-step algebraic/arithmetic derivation, one step per line. "
                "(3) Simplify the final expression to canonical form (integer, reduced fraction, exact radical). "
                "(4) Prepare the final answer ready to be wrapped in \\boxed{...}. "
                "Use ONLY the provided problem text; do not assume external facts."
            )
        elif _is_answer_extract:
            bullets.append(
                "MATH answer extraction: "
                "(1) Extract the final simplified expression from the reasoning. "
                "(2) Place it inside \\boxed{...} exactly. "
                "(3) The VERY LAST LINE of the output MUST be only `\\boxed{<answer>}` and nothing else. "
                "(4) Do NOT add any prose, commentary, or repeat of the question after the boxed line. "
                "(5) Use integers, reduced fractions (a/b), or exact radicals - never decimals unless the problem requires them. "
                "Examples: \\boxed{42}, \\boxed{\\frac{3}{7}}, \\boxed{\\sqrt{2}}."
            )
    elif _is_gsm8k:
        if _is_reasoning:
            bullets.append(
                "GSM8K reasoning: "
                "(1) Identify all numerical quantities mentioned in the problem. "
                "(2) Write explicit arithmetic operations one step per line. "
                "(3) Chain the operations to derive the final numeric answer. "
                "Use ONLY the problem text; do not rely on external knowledge."
            )
        elif _is_answer_extract:
            bullets.append(
                "GSM8K answer extraction: "
                "(1) Extract ONLY the final numeric result. "
                "(2) Strip all units ($, dollars, %, kg, etc.), thousand separators (commas), and currency symbols. "
                "(3) The VERY LAST LINE of the output MUST be a single bare number (integer or decimal), NOTHING else. "
                "(4) Do NOT include equations, units, trailing punctuation, JSON, markdown, or explanation on the last line. "
                "Examples: 42, 3.5, 1200."
            )
    elif _is_humaneval or _is_mbpp:
        bench_name = "HumanEval" if _is_humaneval else "MBPP"
        if _is_reasoning:
            bullets.append(
                f"{bench_name} code generation: "
                "(1) Read the problem description and any docstring examples carefully. "
                "(2) Identify edge cases (empty input, single element, type boundaries, negative numbers, None). "
                "(3) Implement a correct Python function whose name EXACTLY matches the declared entry_point. "
                "(4) Include any necessary `import` statements at the top. "
                "(5) Use clear variable names and concise logic. "
                "Do NOT invent library behaviour - use only standard Python and documented libraries."
            )
        elif _is_answer_extract:
            bullets.append(
                f"{bench_name} final code: "
                "(1) Return raw, executable Python source code only. "
                "(2) The function name MUST exactly match the declared entry_point. "
                "(3) Include all necessary `import` statements at the top. "
                "(4) Do NOT wrap the code in markdown fences (```python ... ```), JSON, natural-language explanation, or commentary. "
                "(5) Ensure the code is syntactically valid and would pass unit tests asserting the examples from the docstring. "
                "Output ONLY the raw code."
            )
    elif _is_drop:
        if _is_task_analysis:
            bullets.append(
                "DROP task analysis (classify the discrete-reasoning operation and expected answer type): "
                "(1) Read the question text in the Passage/Question prompt carefully. "
                "(2) Identify the reasoning operation required - pick EXACTLY ONE from: "
                "'count' (counting occurrences - 'how many X'), "
                "'numeric_arithmetic' (add/subtract/multiply/divide numbers - 'how many more X than Y'), "
                "'date_subtraction' (compute duration between two dates - 'how many years did X last'), "
                "'span_extraction' (copy a single passage span verbatim - 'who did X', 'which X'), "
                "'span_comparison' (pick the larger/longer/first span among candidates - 'which was longer'), "
                "'set_union' (combine multiple spans into a list - 'who and which'). "
                "(3) Identify the expected answer type - pick EXACTLY ONE from: "
                "'number' (integer/decimal, e.g. 20, 3.5), "
                "'date' (year or full date, e.g. 1482, 1980-05-23), "
                "'single_span' (one noun phrase copied from the passage), "
                "'multiple_spans' (two or more noun phrases joined by ' and '). "
                "(4) Use the question wording as primary evidence: 'how many' usually implies number; 'how many years/months/days' implies number (via date_subtraction or numeric_arithmetic); "
                "'who/which' single implies single_span; 'who and which' or 'both' implies multiple_spans; explicit date references imply date type. "
                "Output minimal labels - just the reasoning_type and answer_type values, nothing else. "
                "Use ONLY the provided Passage/Question text; do not invoke external knowledge."
            )
        elif _is_answer_extract:
            bullets.append(
                "DROP answer finalization (normalize to the strict DROP answer format): "
                "(1) If the expected answer_type is 'number' or 'date': extract the bare numeric/date substring, strip all "
                "units, articles, explanations, and surrounding punctuation; preserve hyphens/slashes only if standard for "
                "the number or date format ('20', not '20 years'; '1980', not 'in 1980'). "
                "(2) If the expected answer_type is 'single_span': extract the minimal noun phrase verbatim; strip leading "
                "articles ('the', 'a', 'an') unless they are integral to a proper noun ('United States' keeps 'United'); "
                "preserve original casing of proper nouns. "
                "(3) If the expected answer_type is 'multiple_spans': normalize each span individually then join them with "
                "' and ' (space-and-space), NOT commas and NOT 'or'. "
                "(4) The VERY LAST LINE of the output MUST be ONLY the bare answer with no 'Answer:' prefix, no JSON, "
                "no quotes, no trailing punctuation, no markdown. "
                "Examples: '42', '1980', 'John Smith', 'Germany', 'apples and oranges'. "
                "NEVER return a full sentence or include explanatory text on the final line."
            )
        elif _is_reasoning:
            bullets.append(
                "DROP discrete reasoning (execute the operation on the extracted facts): "
                "(1) For 'count': count the distinct matching items in the fact list; output the count as a bare integer. "
                "(2) For 'numeric_arithmetic': perform the required arithmetic on the extracted numbers, showing each step; "
                "output the bare final number (no units, no thousand separators, no words). "
                "(3) For 'date_subtraction': compute the difference between the two endpoint dates in the requested unit "
                "(years/months/days); output the bare integer. "
                "(4) For 'span_extraction': select the passage span that exactly answers the question, preserving original "
                "casing and wording; output the bare span (no quotes, no 'the answer is' prefix). "
                "(5) For 'span_comparison': select the winning span (longer/larger/first/etc.) from the candidates; output verbatim. "
                "(6) For 'set_union': combine the relevant spans into a single result; output them verbatim in passage order. "
                "General rules: output ONLY the bare result - a single number, a single span, or multiple spans. "
                "For numeric results, output ONLY the number, not '20 years' or 'about 20'. "
                "For span results, do NOT add articles ('a', 'an', 'the') unless they are part of a proper noun in the passage. "
                "Use ONLY the provided facts; do not invent values or use external knowledge."
            )
        elif _is_evidence_extract:
            bullets.append(
                "DROP evidence / fact extraction: "
                "(1) Read the passage carefully and locate sentences that contain the EXACT numerical quantities, "
                "entity names, dates, or events relevant to the question. "
                "(2) Copy verbatim spans from the passage - exact wording, casing, and punctuation - NEVER paraphrase or summarize. "
                "(3) For numeric reasoning (count / arithmetic), extract EVERY number mentioned in the relevant context. "
                "(4) For date arithmetic, extract BOTH endpoint dates (start date and end date). "
                "(5) For span extraction, extract the candidate entity phrase(s) verbatim from the passage. "
                "(6) Prefer completeness over brevity - the next step computes over this list, so missing a fact "
                "causes a wrong answer. "
                "Use ONLY the passage text; do not add outside knowledge. Output the list of extracted spans."
            )

    # Note: an earlier version pasted 'Workflow goal constraint: <goal line>'
    # bullets extracted from the goal text whenever any input/output name
    # appeared in a goal line. That matched far too aggressively: template
    # placeholders like 'Question: <question>' and the literal 'Answer:' prompt
    # marker were being injected as guidance, polluting every fallback prompt.
    # The domain-specific bullets above already encode everything the node needs;
    # goal-line echoes add no signal, so they are intentionally omitted.
    return bullets


def _derive_node_agent_suggestion(goal: str, node, workflow, dataset_name: str = "") -> str:
    """Build a natural-language suggestion for the LLM agent generator.

    The suggestion is consumed by the AgentGeneration action as its `suggestion`
    field and steers the LLM toward generating a concrete, high-quality agent
    prompt. It covers three concerns:

      1. Output format: MUST match the agent prompt template's `## <output_name>`
         subsection convention so the strict validator (GeneratedAgent.validate_prompt)
         accepts the generated prompt. Structured outputs use one subsection per
         output key; plain-text outputs instruct the model to emit a bare value.
      2. Grounding + instruction quality: generic constraints that apply to all nodes.
      3. Domain-specific guidance: delegated to _derive_domain_guidance so the LLM
         path and the deterministic fallback path draw from the same knowledge base.
    """
    outputs = [_param_to_dict(x) for x in (getattr(node, "outputs", []) or [])]

    def _needs_structured_output() -> bool:
        if len(outputs) > 1:
            return True
        for out in outputs:
            if str(out.get("type", "") or "").lower().startswith(("list", "dict")):
                return True
        if len(outputs) == 1:
            out_name = str(outputs[0].get("name", "") or "").lower().strip()
            if out_name and out_name not in {"answer", "result", "output", "content", "text"}:
                return True
        return False

    structured = _needs_structured_output()
    lines: List[str] = []

    # --- Output format: aligned with agent template and validator ---
    if structured:
        out_names = [o["name"] for o in outputs]
        subsection_example = "\n\n".join(
            f"## {n}\n<value for {n}>" for n in out_names
        )
        lines.append(
            f"Output format (CRITICAL - validator enforces this exactly): this node has output "
            f"key(s) {out_names}. The generated agent prompt's '### Output Format' section MUST "
            f"contain one '## <output_name>' subsection per output, in this EXACT shape:\n"
            f"```\n## Thought\nBrief reasoning for achieving the objective.\n\n"
            f"{subsection_example}\n```\n"
            f"Do NOT wrap outputs in a single JSON object. Do NOT collapse multiple outputs into "
            f"one '## Output' section. Subsection names must be EXACTLY {out_names} (case-sensitive). "
            f"The validator extracts `## <name>` subsections with a strict regex and rejects the "
            f"agent if the subsection names or count do not match the declared outputs."
        )
    else:
        plain_name = outputs[0]["name"] if outputs else "answer"
        lines.append(
            f"IMPORTANT - Output format: the agent prompt's '### Output Format' section MUST "
            f"contain exactly one '## {plain_name}' subsection (in addition to '## Thought'). "
            f"The '## {plain_name}' subsection must instruct the model to return ONLY the bare "
            f"plain-text value for '{plain_name}'. No JSON wrapping, no labels, no explanation. "
            "If the output is a final answer, specify it must be a minimal phrase (1-5 words) - "
            "never a full sentence."
        )

    # --- Grounding constraint ---
    lines.append(
        "Grounding: the prompt MUST explicitly tell the model to use ONLY the provided inputs "
        "and not rely on external knowledge or parametric memory."
    )

    # --- Instructions quality ---
    lines.append(
        "Instructions quality: generate detailed, step-by-step instructions specific to this "
        "sub-task. Avoid generic phrases like 'Analyze the inputs.' - instead describe exactly "
        "what evidence to look for, how to chain facts, and what to output."
    )

    # --- Domain-specific guidance (single source of truth shared with fallback) ---
    domain_bullets = _derive_domain_guidance(goal, node)
    if domain_bullets:
        lines.append(
            "Domain-specific guidance - the generated agent prompt's '### Instructions' section "
            "MUST incorporate these rules verbatim or closely paraphrased, preserving all "
            "numeric steps and examples:"
        )
        for b in domain_bullets:
            lines.append(f"  - {b}")

    return "\n".join(lines)

_IDENTIFIER_STOPWORDS = {
    "a", "an", "the", "goal", "answer", "input", "inputs", "output", "outputs",
    "data", "info", "information", "result", "results", "context", "content",
    "task", "tasks", "step", "steps", "item", "items", "value", "values", "list", "final",
}


def _normalize_identifier(text: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text or ""))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower()


def _identifier_tokens(text: Any) -> Set[str]:
    normalized = _normalize_identifier(text)
    return {token for token in normalized.split("_") if token and token not in _IDENTIFIER_STOPWORDS}


def _param_name(param: Any) -> str:
    return str(param.get("name", "")) if isinstance(param, dict) else str(getattr(param, "name", "") or "")


def _param_description(param: Any) -> str:
    return str(param.get("description", "")) if isinstance(param, dict) else str(getattr(param, "description", "") or "")


def _param_required(param: Any) -> bool:
    if isinstance(param, dict):
        return bool(param.get("required", True))
    return bool(getattr(param, "required", True))


def _non_global_inputs(node: WorkFlowNode) -> List[Any]:
    return [param for param in (node.inputs or []) if _normalize_identifier(_param_name(param)) not in {"goal", "answer"}]


def _is_feedback_input(param: Any) -> bool:
    return not _param_required(param)


def _param_similarity_score(source_param: Any, target_param: Any) -> float:
    source_name = _normalize_identifier(_param_name(source_param))
    target_name = _normalize_identifier(_param_name(target_param))
    if not source_name or not target_name or source_name == "answer" or target_name == "goal":
        return 0.0
    score = 0.0
    source_tokens = _identifier_tokens(source_name) | _identifier_tokens(_param_description(source_param))
    target_tokens = _identifier_tokens(target_name) | _identifier_tokens(_param_description(target_param))
    if source_name == target_name:
        score += 10.0
    elif source_name in target_name or target_name in source_name:
        score += 5.0
    overlap = source_tokens & target_tokens
    if len(overlap) >= 2:
        score += 2.0 * len(overlap)
        union = source_tokens | target_tokens
        if union:
            score += len(overlap) / len(union)
    if _is_feedback_input(target_param):
        score *= 0.7
    return score


def _edge_support_score(source_node: WorkFlowNode, target_node: WorkFlowNode) -> float:
    source_outputs = [param for param in (source_node.outputs or []) if _normalize_identifier(_param_name(param)) != "answer"]
    target_inputs = _non_global_inputs(target_node)
    if not source_outputs or not target_inputs:
        return 0.0
    score = 0.0
    exact_matches = 0
    for target_input in target_inputs:
        best_match = 0.0
        for source_output in source_outputs:
            match_score = _param_similarity_score(source_output, target_input)
            if match_score > best_match:
                best_match = match_score
            if _normalize_identifier(_param_name(source_output)) == _normalize_identifier(_param_name(target_input)):
                exact_matches += 1
        score += best_match
    if exact_matches:
        score += 2.0 * exact_matches
    return score


def _format_existing_agents_for_prompt(existing_agents: Optional[List[Agent]]) -> str:
    catalog: List[Dict[str, Any]] = []
    for agent in existing_agents or []:
        if isinstance(agent, Agent):
            catalog.append({"name": agent.name, "description": agent.description})
        elif isinstance(agent, dict):
            catalog.append({"name": agent.get("name", ""), "description": agent.get("description", "")})
        elif isinstance(agent, str):
            catalog.append({"name": agent, "description": "A reusable predefined agent available for selection."})
    return json.dumps(catalog, ensure_ascii=False, indent=2) if catalog else ""


def _existing_agent_runtime_map(existing_agents: Optional[List[Agent]]) -> Dict[str, Dict[str, Any]]:
    runtime_map: Dict[str, Dict[str, Any]] = {}
    for agent in existing_agents or []:
        if isinstance(agent, Agent):
            runtime_map[agent.name] = agent.get_config()
        elif isinstance(agent, dict) and agent.get("name"):
            runtime_map[str(agent["name"])] = dict(agent)
    return runtime_map


class WorkFlowGenerator(BaseModule):
    """
    Automated workflow generation system based on high-level goals.
    
    The WorkFlowGenerator is responsible for creating complete workflow graphs
    from high-level goals or task descriptions. It breaks down the goal into
    subtasks, creates the necessary dependency connections between tasks,
    and assigns or generates appropriate agents for each task.
    
    Attributes:
        llm: Language model used for generation and planning
        task_planner: Component responsible for breaking down goals into subtasks
        agent_generator: Component responsible for agent assignment or creation
        workflow_reviewer: Component for reviewing and improving workflows
        num_turns: Number of refinement iterations for the workflow
    """
    llm: Optional[BaseLLM] = None
    task_planner: Optional[TaskPlanner] = Field(default=None, description="Responsible for breaking down the high-level task into manageable sub-tasks.")
    agent_generator: Optional[AgentGenerator] = Field(default=None, description="Assigns or generates the appropriate agent(s) to handle each sub-task.")
    workflow_reviewer: Optional[WorkFlowReviewer] = Field(default=None, description="Provides feedback and reflections to improve the generated workflow.")
    num_turns: Optional[PositiveInt] = Field(default=0, description="Specifies the number of refinement iterations for the generated workflow.")
    tools: Optional[List[Toolkit]] = Field(default=None, description="A list of tools that can be used in the workflow.")
    
    def init_module(self):
        if self.task_planner is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `task_planner` is None")
            self.task_planner = TaskPlanner(llm=self.llm)
        
        if self.agent_generator is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `agent_generator` is None")
            self.agent_generator = AgentGenerator(llm=self.llm, tools=self.tools)
        
        # TODO add WorkFlowReviewer
        # if self.workflow_reviewer is None:
        #     if self.llm is None:
        #         raise ValueError(f"Must provide `llm` when `workflow_reviewer` is None")
        #     self.workflow_reviewer = WorkFlowReviewer(llm=self.llm)

    def get_tool_info(self):
        self.tool_info =[
            {
                tool.name: [
                    s["function"]["description"] for s in tool.get_tool_schemas()
                ],
            }
            for tool in self.tools
        ]

    def _execute_with_retry(self, operation_name: str, operation, retries_left: int = 1, **kwargs):
        """Helper method to execute operations with retry logic.
        
        Args:
            operation_name: Name of the operation for logging
            operation: Callable that performs the operation
            retries_left: Number of retry attempts remaining
            **kwargs: Additional arguments to pass to the operation
            
        Returns:
            Tuple of (operation_result, number_of_retries_used)
            
        Raises:
            ValueError: If operation fails after all retries are exhausted
        """
        cur_retries = 0

        while cur_retries <= retries_left:  # Changed < to <= to include the initial try
            try:
                logger.info(f"{operation_name} (attempt {cur_retries + 1}/{retries_left + 1}) ...")
                result = operation(**kwargs)
                return result, cur_retries
            except Exception as e:
                if cur_retries == retries_left:
                    raise ValueError(f"Failed to {operation_name} after {cur_retries + 1} attempts.\nError: {e}")
                sleep_time = 2 ** cur_retries
                logger.error(f"Failed to {operation_name} in {cur_retries + 1} attempts. Retry after {sleep_time} seconds.\nError: {e}")
                time.sleep(sleep_time)
                cur_retries += 1

    def generate_workflow(self, goal: str, existing_agents: Optional[List[Agent]] = None, retry: int = 1, suggestion: str = "", **kwargs) -> WorkFlowGraph:
        # Validate input
        if not goal or len(goal.strip()) < 10:
            raise ValueError("Goal must be at least 10 characters and descriptive")

        plan_history = ""

        # Generate the initial workflow plan
        cur_retries = 0
        plan, added_retries = self._execute_with_retry(
            operation_name="Generating a workflow plan",
            operation=self.generate_plan,
            retries_left=retry,
            goal=goal,
            history=plan_history,
            suggestion=suggestion,
        )
        cur_retries += added_retries
        print("plan: ", plan)
        # Build workflow from plan
        workflow, added_retries = self._execute_with_retry(
            operation_name="Building workflow from plan",
            operation=self.build_workflow_from_plan,
            retries_left=retry - cur_retries,
            goal=goal,
            plan=plan
        )
        cur_retries += added_retries

        # Validate initial workflow structure
        logger.info("Validating initial workflow structure...")
        workflow._validate_workflow_structure()
        logger.info(f"Successfully generate the following workflow:\n{workflow.get_workflow_description()}")

        # generate / assigns the initial agents
        logger.info("Generating agents for the workflow ...")
        workflow, added_retries = self._execute_with_retry(
            operation_name="Generating agents for the workflow",
            operation=self.generate_agents,
            retries_left=retry - cur_retries,
            goal=goal,
            workflow=workflow,
            existing_agents=existing_agents
        )
        print("workflow: ", workflow)
        # Validate workflow after agent generation
        logger.info("Validating workflow after agent generation...")
        workflow._validate_workflow_structure()
        # Validate that all nodes have agents
        for node in workflow.nodes:
            if not node.agents:
                if not getattr(node, "agents", None):
                    node.agents = [_build_fallback_agent_for_node(node, goal)]

        return workflow

    def generate_plan(self, goal: str, history: Optional[str] = None, suggestion: Optional[str] = None) -> TaskPlanningOutput:
        history = "" if history is None else history
        suggestion = "" if suggestion is None else suggestion
        task_planner: TaskPlanner = self.task_planner
        task_planning_action_data = {"goal": goal, "history": history, "suggestion": suggestion}
        task_planning_action_name = task_planner.task_planning_action_name
        message: Message = task_planner.execute(
            action_name=task_planning_action_name,
            action_input_data=task_planning_action_data,
            return_msg_type=MessageType.REQUEST
        )
        return message.content
    
    @staticmethod
    def _topological_node_order(workflow: WorkFlowGraph, selected_names: set) -> List[str]:
        """Return selected node names in topological (dependency) order."""
        import networkx as nx
        G = nx.DiGraph()
        for node in workflow.nodes:
            G.add_node(node.name)
        for edge in (workflow.edges or []):
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            if src and tgt:
                G.add_edge(src, tgt)
        try:
            topo = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            topo = [n.name for n in workflow.nodes]
        return [name for name in topo if name in selected_names]

    @staticmethod
    def _upstream_output_schema(workflow: WorkFlowGraph, node_name: str) -> str:
        """Build a description of outputs from upstream nodes for context injection."""
        upstream_names = set()
        for edge in (workflow.edges or []):
            if getattr(edge, "target", None) == node_name:
                src = getattr(edge, "source", None)
                if src:
                    upstream_names.add(src)
        if not upstream_names:
            return ""
        lines = []
        for node in workflow.nodes:
            if node.name in upstream_names:
                for out in (getattr(node, "outputs", None) or []):
                    lines.append(f"  - {node.name}.{out.name} ({out.type}): {out.description}")
        return "\n".join(lines) if lines else ""

    def generate_agents(
        self, 
        goal: str, 
        workflow: WorkFlowGraph,
        existing_agents: Optional[List[Agent]] = None,
        target_node_names: Optional[List[str]] = None,
        dataset_name: str = "",
        # history: Optional[str] = None, 
        # suggestion: Optional[str] = None
    ) -> WorkFlowGraph:
        
        agent_generator: AgentGenerator = self.agent_generator
        workflow_desc = workflow.get_workflow_description()
        agent_generation_action_name = agent_generator.agent_generation_action_name
        existing_agents_prompt = _format_existing_agents_for_prompt(existing_agents)
        existing_agent_map = _existing_agent_runtime_map(existing_agents)
        target_names = {str(name) for name in (target_node_names or []) if str(name)}
        all_selected_names = {subtask.name for subtask in workflow.nodes if not target_names or subtask.name in target_names}
        ordered_names = self._topological_node_order(workflow, all_selected_names)
        node_map = {subtask.name: subtask for subtask in workflow.nodes}
        selected_nodes = [node_map[name] for name in ordered_names if name in node_map]
        generated_output_schemas: dict = {}  # node_name -> list of output descriptions
        for subtask in selected_nodes:
            subtask_fields = ["name", "description", "reason", "inputs", "outputs"]
            subtask_data = {key: value for key, value in subtask.to_dict(ignore=["class_name"]).items() if key in subtask_fields}
            # Inject upstream output schema for cross-node coordination
            upstream_schema = self._upstream_output_schema(workflow, subtask.name)
            if generated_output_schemas:
                # Append already-generated upstream agent output info
                dynamic_upstream_lines = []
                for up_name, up_outputs in generated_output_schemas.items():
                    if up_name in {getattr(e, "source", None) for e in (workflow.edges or []) if getattr(e, "target", None) == subtask.name}:
                        dynamic_upstream_lines.extend(up_outputs)
                if dynamic_upstream_lines:
                    upstream_schema = (upstream_schema + "\n" if upstream_schema else "") + "\n".join(dynamic_upstream_lines)
            if upstream_schema:
                subtask_data["upstream_output_context"] = upstream_schema
            subtask_desc = json.dumps(subtask_data, indent=4)
            node_suggestion = _derive_node_agent_suggestion(goal, subtask, workflow, dataset_name=dataset_name)
            agent_generation_action_data = {
                "goal": goal,
                "workflow": workflow_desc,
                "task": subtask_desc,
                "history": "",
                "suggestion": node_suggestion,
                "existing_agents": existing_agents_prompt,
            }
            logger.info(f"Generating agents for subtask: {subtask_data['name']}")
            try:
                agents: AgentGenerationOutput = agent_generator.execute(
                    action_name=agent_generation_action_name, 
                    action_input_data=agent_generation_action_data,
                    return_msg_type=MessageType.RESPONSE
                ).content
                selected_agents = []
                unresolved_selected_agents = []
                for agent_name in agents.selected_agents or []:
                    agent_config = existing_agent_map.get(str(agent_name))
                    if agent_config is None:
                        unresolved_selected_agents.append(str(agent_name))
                        continue
                    selected_agents.append(agent_config)
                generated_agents = []
                for agent in agents.generated_agents:
                    generated_agents.append(agent.to_dict(ignore=["class_name"]))
                if unresolved_selected_agents:
                    logger.warning(
                        f"Selected predefined agents for subtask '{subtask_data['name']}' could not be resolved: "
                        f"{unresolved_selected_agents}"
                    )
                assigned_agents = selected_agents + generated_agents
                subtask.set_agents(agents=assigned_agents or [_build_fallback_agent_for_node(subtask, goal, dataset_name=dataset_name)])
                # Record output schema for downstream coordination
                out_lines = []
                for out in (getattr(subtask, "outputs", None) or []):
                    out_lines.append(f"  - {subtask.name}.{out.name} ({out.type}): {out.description}")
                if out_lines:
                    generated_output_schemas[subtask.name] = out_lines
            except Exception as e:
                logger.warning(
                    f"Generating agents for subtask '{subtask_data['name']}' failed: {e}. "
                    "Use fallback agent instead."
                )
                subtask.set_agents(agents=[_build_fallback_agent_for_node(subtask, goal, dataset_name=dataset_name)])
        return workflow
    
    # def review_plan(self, goal: str, )
    #将plan构造为得到的.json格式工作流
    def build_workflow_from_plan(self, goal: str, plan: TaskPlanningOutput) -> WorkFlowGraph:
        """Build a WorkFlowGraph from the planner output.

        Edge construction uses exact normalized name matching: the planner is
        instructed to name each sub-task input after the output of the upstream
        node that produces it, so deterministic name matching is both correct
        and sufficient.  Token-similarity heuristics are intentionally avoided
        because they produced false edges and missed true ones when variable
        names were generic or inconsistently cased.

        Algorithm:
        1. Build an index: normalized_output_name -> source_node_name for all
           nodes seen so far (in plan order).
        2. For each input of the current node (excluding 'goal' / 'answer'),
           look up its normalized name in the index.  Every match adds an edge.
        3. If no input matched any earlier output, add a backbone edge from the
           immediate predecessor to preserve a connected DAG.
        """
        nodes: List[WorkFlowNode] = plan.sub_tasks
        edges: List[WorkFlowEdge] = []
        seen_edges: set = set()

        def _register_edge(source_name: str, target_name: str) -> None:
            if source_name == target_name:
                return
            key = (source_name, target_name)
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges.append(WorkFlowEdge(edge_tuple=key))

        # Global inputs that come from the user / workflow context, not from
        # any node output, so they must never trigger edge creation.
        _GLOBAL_INPUTS = {"goal", "answer"}

        for target_idx, target_node in enumerate(nodes):
            if target_idx == 0:
                continue  # first node has no predecessors

            # Build output-name → producer index for all preceding nodes.
            # Later nodes override earlier ones when names collide (most recent
            # definition wins, which matches how the planner chains outputs).
            output_producer: Dict[str, str] = {}
            for source_node in nodes[:target_idx]:
                for out_param in (source_node.outputs or []):
                    out_name = _normalize_identifier(_param_name(out_param))
                    if out_name and out_name not in _GLOBAL_INPUTS:
                        output_producer[out_name] = source_node.name

            # Match each input of the target node against the output index.
            found_any = False
            for inp_param in (target_node.inputs or []):
                inp_name = _normalize_identifier(_param_name(inp_param))
                if not inp_name or inp_name in _GLOBAL_INPUTS:
                    continue  # skip global / anonymous inputs
                producer = output_producer.get(inp_name)
                if producer is not None:
                    _register_edge(producer, target_node.name)
                    found_any = True

            # Backbone fallback: if the target has no matched upstream, link it
            # to the immediate predecessor to keep the graph connected.
            if not found_any:
                _register_edge(nodes[target_idx - 1].name, target_node.name)

        workflow = WorkFlowGraph(goal=goal, nodes=nodes, edges=edges)
        return workflow
    
