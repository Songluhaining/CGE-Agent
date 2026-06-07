import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from evoagentx.benchmark import AFlowHotPotQA, AFlowMATH, AFlowHumanEval, AFlowMBPP, AFlowGSM8K, AFlowDROP
from evoagentx.core.base_config import Parameter
from evoagentx.models import AliyunLLM, AliyunLLMConfig, OpenAILLM, OpenRouterConfig, OpenRouterLLM, OpenAILLMConfig
from evoagentx.optimizers import LLMWorkflowOptimizer
from evoagentx.optimizers import rl_workflow_optimizer as _legacy
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow.workflow_graph import WorkFlowEdge, WorkFlowGraph, WorkFlowNode


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "hotpotqa_initial_workflow_gemini-2.5-flash-lite.json"
BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" /"workflows"/ "hotpotqa_best_workflow_gemini-2.5-flash-lite_woMem.json"
# Set USE_BEST_WORKFLOW=1 to skip optimization and run directly on the saved best workflow.
USE_BEST_WORKFLOW = os.getenv("USE_BEST_WORKFLOW", "0").strip() == "1"
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""

# OpenAI-compatible endpoint for both the optimizer and executor LLMs.
# Set these in your shell or in a local .env file (see .env.example). Never hard-code keys.
# To reproduce the paper, point OPENAI_BASE_URL at any OpenAI-compatible gateway that
# serves the configured model (the paper used gemini-2.5-flash-lite via such a gateway).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Dataset selector: "hotpotqa" or "math" or mbpp or gsm8k or humaneval drop
DATASET = "drop"

# MATH-specific cache paths
MATH_INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "math_initial_workflow_gemini-2.5-flash-lite.json"
MATH_BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "math_best_workflow_gemini-2.5-flash-lite_woMem.json"

# HumanEval-specific cache paths
HUMANEVAL_INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "humaneval_initial_workflow_gemini-2.5-flash-lite.json"
HUMANEVAL_BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "humaneval_best_workflow_gemini-2.5-flash-lite_woMem.json"

# MBPP-specific cache paths
MBPP_INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "mbpp_initial_workflow_gemini-2.5-flash-lite.json"
MBPP_BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "mbpp_best_workflow_gemini-2.5-flash-lite_woMem.json"

# GSM8K-specific cache paths
GSM8K_INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "gsm8k_initial_workflow_gemini-2.5-flash-lite.json"
GSM8K_BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "gsm8k_best_workflow_gemini-2.5-flash-lite_woMem.json"

# drop-specific cache paths
DROP_INITIAL_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "drop_initial_workflow_gemini-2.5-flash-lite.json"
DROP_BEST_WORKFLOW_CACHE_PATH = REPO_ROOT / "data" / "workflows" / "drop_best_workflow_gemini-2.5-flash-lite_woMem.json"


# HotPotQA-specific workflow generation goal.
# Overrides the generic WORKFLOW_GOAL to give the LLM domain context during initial workflow generation.
HOTPOTQA_WORKFLOW_GOAL = """
# Task: Design a workflow for Multi-Hop Question Answering (HotPotQA) problem

## Key Properties of HotPotQA
- Questions require multi-hop reasoning: the answer depends on connecting facts from MULTIPLE context passages.
- The answer should be a direct response to the question, without including explanations or reasoning.

## Example
- {"_id": "5a7c49dc55429935c91b514f", "answer": "Engineering", "question": "The head of the Foreign Relations Department of the Rastriya Janashakti Party holds a degree that can be abbreviated MS, M.S., or ScM, in what field?", "supporting_facts": [["Hari Bahadur Basnet", 1], ["Hari Bahadur Basnet", 3], ["Master of Science", 0]], "context": [["Sikkim Janashakti Party", ["Sikkim Janashakti Party (translation: Sikkim People's Power Party), was a political party in the Indian state of Sikkim.", " SJP was founded in 1997, when Tara Man Rai broke away from Sikkim Ekta Manch.", " Rai was the president of SJP.", " In January 1999 SJP merged with Indian National Congress."]], ["Master of Science", ["A Master of Science (Latin: \"Magister Scientiae\" ; abbreviated MS, M.S., MSc, M.Sc., MSci, M.Sci., ScM, Sc.M., SciM or Sci.M.)", " is a master's degree in the field of science awarded by universities in many countries, or a person holding such a degree.", " In contrast to the Master of Arts degree, the Master of Science degree is typically granted for studies in sciences, engineering, and medicine, and is usually for programs that are more focused on scientific and mathematical subjects; however, different universities have different conventions and may also offer the degree for fields typically considered within the humanities and social sciences.", " While it ultimately depends upon the specific program, earning a Master of Science degree typically includes writing a thesis."]], ["Hari Bahadur Basnet", ["Hari Bahadur Basnet is a Nepalese politician.", " He is the head of the Foreign Relations Department of the Rastriya Janashakti Party.", " Basnet holds a M.Sc.", " in Engineering."]], ["Gregory Weeks", ["Gregory Weeks (born 1970) is a lecturer at the International Relations Department at Webster University in Vienna, Austria.", " He was the Head of the International Relations Department from 2005 until 2011.", " Weeks teaches and researches civil-military relations, genocide prevention, and twentieth century Austrian and German diplomatic and military history."]], ["Rastriya Janashakti Mahila Sangh", ["Rastriya Janashakti Mahila Sangh (Nepali: \u0930\u093e\u0937\u094d\u091f\u094d\u0930\u093f\u092f \u091c\u0928\u0936\u0915\u094d\u0924\u093f \u092e\u0939\u093f\u0932\u093e \u0938\u0902\u0918 ) is a women's organisation in Nepal, politically aligned with the Rastriya Janashakti Party."]], ["Rastriya Janashakti Student Union", ["Rastriya Janashakti Student Union is a students organisation in Nepal.", " It is the students wing of the Royalist Rashtriya Janashakti Party."]], ["Rastriya Janashakti Party", ["Rastriya Janashakti Party is a liberal political party in Nepal, led by former Prime Minister Surya Bahadur Thapa.", " Thapa had split away from the Rastriya Prajatantra Party in November 2004.", " The party is registered with the Election Commission of Nepal in March 2005."]], ["Politics in the San Francisco Bay Area", ["Politics in the San Francisco Bay Area is widely regarded as one of the most liberal in the country.", " According to the California Secretary of State, the Democratic Party holds a voter registration advantage in every congressional district, state senate district, state assembly district, State Board of Equalization districts, all nine counties, and all but three of the 101 incorporated municipalities in the Bay Area.", " The Republican Party holds a voter registration advantage in one state assembly subdistrict (the portion of California's 4th State Assembly district in Solano county) and three cities, Atherton, Hillsborough, and Danville."]], ["Foreign relations of Finland", ["The foreign relations of Finland are the responsibility of the President of Finland, who leads foreign policy in cooperation with the government.", " Implicitly the government is responsible for internal policy and decision making in the European Union.", " Within the government, preparative discussions are conducted in the government committee of foreign and security policy (\"ulko- ja turvallisuuspoliittinen ministerivaliokunta\"), which includes the Prime Minister and at least the Minister of Foreign Affairs and the Minister of Defence, and at most four other ministers as necessary.", " The committee meets with the President as necessary.", " Laws concerning foreign relations are discussed in the parliamentary committee of foreign relations (\"ulkoasiainvaliokunta, utrikesutskottet\").", " The Ministry of Foreign Affairs implements the foreign policy."]], ["ITV (Thailand)", ["iTV was a television station in Thailand owned by ITV Public Company Limited, a unit of Shin Corporation.", " Thailand's first UHF channel, the station was started in 1995 when the company was granted a 30-year concession by the Office of the Permanent Secretary to the Prime Minister's Office to operate a free-to-air television station in the Ultra High Frequency (UHF) spectrum at 510-790 MHz (from Channel 26 to 60).", " After a lengthy dispute over unpaid concession fees to the Prime Minister's Office, iTV was taken in 2007 over by the government's Public Relations Department and its name was changed to Thai Independent Television (TITV).", " Following a previously unannounced order of Thailand's Public Relations Department delivered the same day, the station closed down operations at the crack of dawn on January 15th, 2008.", " In accordance with the Public Broadcasting Service Act, the channel's frequency was assigned to the Thai Public Broadcasting Service, or Thai PBS."]]], "type": "bridge", "level": "hard"}

## Output Format Contract (the ONLY hard requirements; topology, node count, node names, and intermediate design are entirely up to the task planner)

HotPotQA F1 is computed as token-overlap against a short gold entity (e.g. "Engineering", "1994"). The rules below concern ONLY how the FINAL answer string is produced and parsed — they do NOT restrict how you decompose the task. You may use any number of nodes and any intermediate structure you think is best, as long as the workflow's TERMINAL (sink) node satisfies every rule below.

1. Terminal output parameter. The terminal node's `outputs` MUST contain EXACTLY ONE parameter whose `name` is exactly `answer` (lowercase), `type` is `string`, `required` is true. Whatever the upstream topology looks like, the sink MUST emit a field called `answer`.

2. Parse mode. The terminal task spec in the planner JSON MUST set `"parse_mode": "title"`. The agent dict inside that task's `agents` list MUST ALSO set `"parse_mode": "title"`. This is how the runtime extracts the answer string from the LLM response.

3. Single heading in the terminal prompt. The terminal node's `prompt` MUST use `## answer` as the ONLY `## <name>` markdown heading anywhere in that prompt. Do NOT include `## Thought`, `## reasoning`, `## explanation`, `## analysis`, or any other `## <name>` heading — extra headings break the `title` parser and produce empty/wrong `answer` values.

4. No placeholder text under `## answer`. Inside the terminal prompt, nothing descriptive should appear under the `## answer` heading. Do NOT write things like "The final answer extracted from the context." or "<your answer here>" on the line after `## answer`, because the LLM will copy such placeholder sentences verbatim into its response and destroy F1. Put all explanation and instruction ABOVE the Output Format block; after `## answer` leave either an empty line or only the short directive `<the 1-5 word verbatim answer, or yes/no>`.

5. Answer content instruction (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `Extract the answer to the question as a minimal 1-5 word noun phrase copied VERBATIM from the Context paragraphs inside the goal. Do NOT paraphrase, explain, translate, add articles, add JSON, add markdown code fences, or prefix with 'The answer is'. If the answer is yes/no, output exactly yes or no.`

6. Few-shot anchor in the terminal prompt. Immediately before the Output Format section, the terminal prompt MUST include this exact one-line anchor: `Example: if the gold answer is "Engineering", your entire response MUST be exactly two lines - "## answer" then "Engineering" - with no other text.`

7. Intermediate-node freedom (explicitly allowed). You MAY add any intermediate nodes you think improve the workflow (evidence selection, reasoning, verification, ensemble, etc.). Their names, inputs, outputs, and prompts are entirely your call, and rules 3-6 apply ONLY to the terminal node (intermediate nodes may use any `## <name>` structure and any parse_mode that fits their job). If you use multiple nodes, ensure each downstream node's `inputs` list explicitly names every upstream output it actually consumes — otherwise the edge carries no data and the downstream node will see only the workflow's global inputs (e.g. `goal`).



"""

MATH_WORKFLOW_GOAL = """
# Task: Design a workflow for Mathematical Problem Solving (MATH competition dataset)

## Key Properties of MATH
- Problems are competition-level mathematics spanning 7 subjects: Algebra, Geometry, Intermediate Algebra, Counting & Probability, Precalculus, Number Theory, and Prealgebra.
- Problems range from Level 1 (easiest) to Level 5 (hardest).
- The final answer MUST be enclosed in \\boxed{} format (LaTeX). This is the canonical answer format — the evaluation extracts the content inside \\boxed{} for scoring.
- Solutions require multi-step mathematical reasoning, symbolic manipulation, and sometimes numerical computation.
- The answer should be a precise mathematical expression (number, fraction, expression), not a natural language explanation.

## Example
- {"problem": "A particular convex pentagon has two congruent, acute angles. The measure of each of the other interior angles is equal to the sum of the measures of the two acute angles. What is the common measure of the large angles, in degrees?", "level": "Level 5", "type": "Prealgebra", "solution": "If $x$ is the measure in degrees of each of the acute angles, then each of the larger angles measures $2x$ degrees.  Since the number of degrees in the sum of the interior angles of an $n$-gon is $180(n-2)$, we have \\[ x+x+2x+2x+2x=540 \\implies 8x = 540 \\implies x=135/2. \\] The large angles each measure $2x=\\boxed{135}$ degrees."}

## Output Format Contract (the ONLY hard requirements; topology is up to the task planner)

1. Terminal node output parameter. The terminal (sink) node's `outputs` MUST contain EXACTLY ONE parameter whose `name` is exactly `final_answer` (lowercase), `type` is `string`, `required` is true.

2. Terminal parse mode. The terminal task spec in the planner JSON MUST set `"parse_mode": "title"`. The agent dict inside that task's `agents` list MUST ALSO set `"parse_mode": "title"`. Do NOT use `"json"` on the terminal node - JSON parsing chokes on bare LaTeX backslashes.

3. Single heading in the terminal prompt. The terminal node's `prompt` MUST use `## final_answer` as the ONLY `## <name>` markdown heading anywhere in that prompt. Do NOT add `## reasoning`, `## solution`, `## steps`, `## explanation`, or any other `## <name>` heading - extra headings break the `title` parser.

4. No placeholder text under `## final_answer`. Inside the terminal prompt, nothing descriptive should appear under the `## final_answer` heading. Do NOT write things like "The boxed answer goes here." or "<your answer here>" on the line after `## final_answer`, because the LLM will copy such placeholder sentences verbatim into its response. Put all explanation and instruction ABOVE the Output Format block; after `## final_answer` leave either an empty line or only the short directive `<the final answer in \\boxed{...} form, on its own line>`.

5. Answer content instruction (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `Return ONLY the final answer in the exact form \\boxed{...}. Do NOT output JSON, do NOT wrap in code fences, do NOT add explanation, do NOT repeat the question. The \\boxed{...} expression must be on its own line immediately under the "## final_answer" heading, and must be the ONLY text after that heading.`

6. Few-shot anchor in the terminal prompt. Immediately before the Output Format section, the terminal prompt MUST include this exact one-line anchor: `Example: for the pentagon problem, your entire response MUST be exactly two lines - "## final_answer" then "\\boxed{135}" - with no JSON, no code fence, no extra text.`

7. Intermediate-node parse mode (CRITICAL). Any intermediate node whose outputs MAY contain LaTeX - step-by-step derivation, symbolic manipulation, case analysis, algebraic simplification, verification, etc. - MUST set `"parse_mode": "title"` (preferred) or `"parse_mode": "str"`. Intermediate nodes MUST NOT use `"parse_mode": "json"` if any of their output fields can legitimately contain `\\(`, `\\)`, `\\sqrt`, `\\frac`, `\\boxed`, `\\sum`, `\\int`, `\\binom`, `|`, or other LaTeX/math characters. This restriction applies even if the field is named `reasoning`, `solution_steps`, `derivation`, `computation`, or similar.

## Topology requirements (CRITICAL — affects accuracy)

- The raw `problem` text MUST be carried as a direct input to EVERY node that performs reasoning, computation, or final-answer generation. Specifically: any node whose role is to compute a numeric/symbolic value (e.g. `solve`, `execute_solution`, `final_answer`, `verify`) MUST list `problem` in its `inputs`. Do NOT rely on a chain like `problem_analysis → solution_strategy → execute_solution(solution_steps_only)` where downstream nodes only receive intermediate descriptive text — that loses the original numbers, equations, and constraints and forces the LLM to guess.
- The terminal node's `inputs` MUST include `problem` so it can sanity-check the boxed answer against the original problem statement.
- Prefer FEWER nodes (1-3 total) over many small nodes; mid-tier executor models lose context across long chains. A single-node `solve(problem) → final_answer` workflow is a perfectly valid baseline.
- Each intermediate node, if any, MUST also include `problem` (or the original `goal`) in its inputs in addition to its upstream outputs.

8. Intermediate-node freedom (explicitly allowed). You MAY add any intermediate nodes you think improve the workflow (problem analysis, sub-problem decomposition, case split, numerical check, symbolic verification, ensemble, etc.), as long as the topology requirements above and rule 7 hold.

"""

HUMANEVAL_WORKFLOW_GOAL = """
# Task: Design a workflow for Python Code Generation (HumanEval dataset)

## Key Properties of HumanEval
- HumanEval is a collection of hand-written Python programming problems. Each problem provides a function signature and a docstring describing the expected behavior, usually illustrated with a few input/output examples inside the docstring.
- Problems cover string manipulation, list / tuple processing, arithmetic, sorting, recursion, and basic algorithms. Most canonical solutions are 1-15 lines of Python.
- The `entry_point` field gives the exact function name; the generated function MUST be defined with that name so the hidden unit tests can import it.
- Solutions are scored with pass@1: the generated code is passed to `exec()` by the evaluator and then run against hidden unit tests. A single failing assertion counts the problem as wrong.
- The final `answer` string MUST be raw, directly-executable Python defining the requested function (signature + body). No natural-language explanation, no markdown code fences, no rationale, no example invocations, no "__main__" blocks - anything that makes `exec()` fail counts as a wrong answer.

## Example
- {"task_id": "HumanEval/135", "prompt": "\ndef can_arrange(arr):\n    \"\"\"Create a function which returns the largest index of an element which\n    is not greater than or equal to the element immediately preceding it. If\n    no such element exists then return -1. The given array will not contain\n    duplicate values.\n\n    Examples:\n    can_arrange([1,2,4,3,5]) = 3\n    can_arrange([1,2,3]) = -1\n    \"\"\"\n", "entry_point": "can_arrange", "canonical_solution": "    ind=-1\n    i=1\n    while i<len(arr):\n      if arr[i]<arr[i-1]:\n        ind=i\n      i+=1\n    return ind\n"}


## Output Format Contract (the ONLY hard requirements; topology is up to the task planner)

1. Terminal output parameter. The terminal (sink) node's `outputs` MUST contain EXACTLY ONE parameter whose `name` is exactly `answer` (lowercase), `type` is `string`, `required` is true.

2. Terminal parse mode. The terminal task spec in the planner JSON MUST set `"parse_mode": "str"`. The agent dict inside that task's `agents` list MUST ALSO set `"parse_mode": "str"`. Do NOT use `"json"` or `"title"` on the terminal node - JSON parsing chokes on bare Python backslashes/quotes, and `title` mode misaligns when code contains markdown-like sequences inside comments or docstrings.

3. Terminal prompt content. The terminal node's `prompt` MUST instruct the LLM to emit ONLY raw Python source code (the requested function definition, plus any required imports or helper definitions above it) and nothing else. The terminal prompt MUST contain this sentence literally: `Output ONLY raw Python source code. Do NOT wrap the code in markdown fences, do NOT prefix with "Here is" or any explanation, do NOT include example invocations, do NOT include assert statements or test harnesses, do NOT include an __main__ guard block, and do NOT append any trailing natural-language text. The first non-blank line of your response MUST begin with "import", "from", "def", or "class".`

4. Entry-point enforcement (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `The generated function MUST be defined with the EXACT name that appears after "def" in the function signature inside the goal (this is the task entry_point). A name mismatch is counted as a wrong answer, because the hidden unit tests import the function by that exact name.`

5. Few-shot anchor in the terminal prompt. Immediately before the Output Format section, the terminal prompt MUST include this exact one-line anchor: `Example: for the can_arrange task, your entire response MUST be the raw definition starting with "def can_arrange(arr):" followed by the function body - no fences, no explanation, no asserts.`

## Topology requirements (CRITICAL - affects pass@1 accuracy)

- The terminal node's `inputs` MUST include `goal` so the code-generating LLM sees the ORIGINAL function signature and docstring (which contains the exact entry_point name and the I/O examples). Paraphrasing the prompt through an upstream "problem understanding" node and feeding only the paraphrase downstream loses the exact function name and collapses pass@1 - do NOT do that.
- Every reasoning or code-generating intermediate node (if any) MUST also list `goal` in its `inputs` in addition to any upstream outputs it consumes.
- Prefer FEWER nodes (1-2 total) over many small nodes; mid-tier executor models lose context and the exact function signature across long chains. A single-node `solve(goal) -> answer` workflow is a perfectly valid and recommended baseline for HumanEval.
- FORBIDDEN node kinds (these categorically hurt pass@1):
  - Any node whose role is "execute the code", "run the unit tests", "validate by running", or similar. The LLM CANNOT actually execute Python; it hallucinates a True/False verdict, and any downstream node that gates on that hallucinated verdict (e.g. "return an error message when validation_result is false") will overwrite the correct code with error text and destroy pass@1.
  - Any node that produces a boolean `validation_result`, `passed`, `tests_passed` or similar verdict field based on "reading" the code. Such a verdict cannot be trusted and must not be used as a gate.
  If you believe validation is needed, omit it - a single `solve(goal) -> answer` node is strictly better than a chain with fake validation.

6. Intermediate-node freedom (explicitly allowed). You MAY add ONE intermediate analysis or planning node if you believe it helps (e.g. "identify algorithmic approach", "enumerate edge cases", "outline solution strategy"), as long as the topology requirements above and the Output Format Contract hold. The intermediate node MUST NOT produce code itself (that is the terminal node's job) and MUST NOT gate the terminal node's execution on a boolean verdict. Intermediate nodes may use any `## <name>` structure and any parse_mode that fits their job; only the rules under Output Format Contract apply to the terminal node.

"""


MBPP_WORKFLOW_GOAL = """
# Task: Design a workflow for Python Code Generation (MBPP dataset)

## Key Properties of MBPP
- MBPP (Mostly Basic Python Programming) is a collection of short Python programming problems. Each item is a one- or two-sentence natural-language specification followed by the target function header (sometimes with a trailing URL reference). There is usually NO docstring and NO input/output examples inside the prompt.
- Problems cover list / tuple operations, string processing, arithmetic, sorting and searching, simple numerical algorithms, dictionary manipulation, and basic regex. The target function usually does ONE well-defined thing; most canonical solutions are 3-15 lines of Python.
- The `entry_point` field gives the exact function name the hidden unit tests will call; the generated function MUST be defined with that name. The evaluator does attempt to auto-rename a mis-named last top-level function to `entry_point`, but that heuristic fails when a helper is defined after the main function — so emitting the correct name up front is strictly safer.
- The test cases (`test_list`) are HIDDEN at runtime. The workflow CANNOT see them, CANNOT run them, and CANNOT verify behavior against them. Any node that claims to "validate / test / run the code" is hallucinating — the LLM has no sandbox, no interpreter, and no oracle.
- Solutions are scored with pass@1: the evaluator concatenates `prompt + "\\n" + solution`, passes the result to `exec()`, then runs the hidden unit tests. A single failing assertion (including a NameError / SyntaxError / ImportError at exec time) counts the problem wrong.
- The final `answer` string MUST be raw, directly-executable Python that (a) optionally starts with required `import` statements, (b) defines the requested function with the EXACT `entry_point` name and the signature shown in the prompt, and (c) contains NO natural-language explanation, NO markdown code fences, NO rationale, NO `assert` / `check(...)` / test-harness wrappers, NO example invocations, and NO `if __name__ == "__main__":` blocks. Helper functions and imports MUST live in the SAME output string BEFORE the main function definition.

## Example
- {"task_id": 802, "prompt": "Write a python function to count the number of rotations required to generate a sorted array. https://www.geeksforgeeks.org/count-of-rotations-required-to-generate-a-sorted-array/\\n\\ndef count_rotation(arr):   ", "code": "def count_rotation(arr):   \\n    for i in range (1,len(arr)): \\n        if (arr[i] < arr[i - 1]): \\n            return i  \\n    return 0", "test_list": ["assert count_rotation([3,2,1]) == 1", "assert count_rotation([4,5,1,2,3]) == 2", "assert count_rotation([7,8,9,1,2,3]) == 3", "assert count_rotation([1,2,3]) == 0", "assert count_rotation([1,3,2]) == 2"], "entry_point": "count_rotation"}

## Output Format Contract (the ONLY hard requirements; topology is up to the task planner)

1. Terminal output parameter. The terminal (sink) node's `outputs` MUST contain EXACTLY ONE parameter whose `name` is exactly `answer` (lowercase), `type` is `string`, `required` is true.

2. Terminal parse mode. The terminal task spec in the planner JSON MUST set `"parse_mode": "str"`. The agent dict inside that task's `agents` list MUST ALSO set `"parse_mode": "str"`. Do NOT use `"json"` or `"title"` on the terminal node - JSON parsing chokes on bare Python backslashes/quotes inside strings and regexes, and `title` mode misaligns when code contains markdown-like sequences inside comments or docstrings.

3. Terminal prompt content. The terminal node's `prompt` MUST instruct the LLM to emit ONLY raw Python source code and nothing else. The terminal prompt MUST contain this sentence literally: `Output ONLY raw Python source code. Do NOT wrap the code in markdown fences, do NOT prefix with "Here is" or any explanation, do NOT include example invocations, do NOT include assert statements or test harnesses, do NOT include an __main__ guard block, and do NOT append any trailing natural-language text. The first non-blank line of your response MUST begin with "import", "from", "def", or "class".`

4. Entry-point enforcement (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `The generated function MUST be defined with the EXACT name that appears after "def" in the function header inside the goal (this is the task entry_point). A name mismatch is counted as a wrong answer, because the hidden unit tests import the function by that exact name.`

5. Completeness of output (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `Your response MUST contain the COMPLETE function definition (def line + indented body), not just the body — the evaluator concatenates the prompt and your answer and passes the result to exec(), so a bare body produces an IndentationError. If helper functions or imports are needed, place them BEFORE the main function in the SAME output string.`

6. Few-shot anchor in the terminal prompt. Immediately before the Output Format section, the terminal prompt MUST include this exact one-line anchor: `Example: for the count_rotation task whose prompt ends with "def count_rotation(arr):   ", your entire response MUST begin with "def count_rotation(arr):" followed by the function body — no fences, no explanation, no asserts, no test calls.`

## Topology Requirements (CRITICAL - affects pass@1 accuracy)

- The terminal node's `inputs` MUST include `goal` so the code-generating LLM sees the ORIGINAL specification and function header (which contain the exact `entry_point` name, the signature, and whatever constraints the natural-language description imposes). Paraphrasing the prompt through an upstream "contract extraction" / "problem understanding" node and feeding only the paraphrase downstream loses the exact function name and the precise signature, and collapses pass@1 — do NOT do that. If any intermediate node extracts a "function signature" or "behavioral intent", the terminal node MUST STILL list `goal` directly in its `inputs` alongside (not in place of) those intermediate outputs.
- Every reasoning or code-generating intermediate node (if any) MUST also list `goal` in its `inputs` in addition to any upstream outputs it consumes.
- Prefer FEWER nodes (1-2 total) over many small nodes; mid-tier executor models lose context and the exact function header across long chains. A single-node `solve(goal) -> answer` workflow is a perfectly valid and RECOMMENDED baseline for MBPP.
- FORBIDDEN node kinds (these categorically hurt pass@1 — the hidden-test setting makes them strictly worse than doing nothing):
  - Any node whose role is "execute the code", "run the unit tests", "validate by running", "check against test cases", or similar. The LLM CANNOT actually execute Python, has NO access to the hidden `test_list`, and will hallucinate a True/False verdict. Any downstream node that gates on that hallucinated verdict (e.g. "return an error message when validation_result is false", "rewrite the code until tests pass") will overwrite correct code with error text and destroy pass@1.
  - Any node whose role is "extract / derive / guess test assertions" from the prompt. MBPP prompts do NOT contain reference assertions; the LLM will fabricate plausible-looking but wrong asserts, and any downstream "validation" built on those fabricated asserts will flag correct solutions as failing and trigger counter-productive rewrites.
  - Any node that produces a boolean `validation_result`, `passed`, `tests_passed`, `is_correct`, or similar verdict field based on "reading" the code or running invented tests. Such verdicts cannot be trusted and MUST NOT be used as a gate on the terminal node.
  - Any "candidate_generation without goal" pattern — generating code from a paraphrased `signature` + `intent` without seeing the original `goal` string loses the exact function name and constraints.
  If you believe validation is needed, omit it — a single `solve(goal) -> answer` node is strictly better than any chain with fake validation.

7. Intermediate-node freedom (explicitly allowed). You MAY add ONE intermediate analysis or planning node if you believe it helps (e.g. "identify algorithmic approach", "enumerate edge cases", "outline solution strategy"), as long as the Topology Requirements above and the Output Format Contract hold. The intermediate node MUST NOT produce code itself (that is the terminal node's job), MUST NOT produce test assertions, and MUST NOT gate the terminal node's execution on a boolean verdict. Intermediate nodes may use any `## <name>` structure and any parse_mode that fits their job; only the rules under Output Format Contract apply to the terminal node.
"""


GSM8K_WORKFLOW_GOAL = """
# Task: Design a workflow for Grade School Math Problem Solving (GSM8K dataset)

## Key Properties of GSM8K
- GSM8K problems are grade-school level math WORD problems that describe a short real-world scenario and ask for a specific numeric answer. Solving them typically requires 2-8 steps of arithmetic reasoning over integer or simple decimal quantities.
- Each `question` is self-contained natural language; the reference `cot` field shows the step-by-step computation with intermediate values annotated in `<<expr=result>>` form. The `answer` is always a single number.
- Correctness is judged by exact-match against the reference number (solve rate). The scoring function extracts the LAST number found anywhere in the prediction string via regex, so ANY number that appears after the true final answer - inside trailing explanation, a "verification" sentence, a units-clarification clause - OVERRIDES the correct value and makes the example wrong.
- The VERY LAST LINE of the final `answer` string MUST be the numeric answer ALONE - no units (no "signatures", "hours", "dollars"), no currency symbols ($ / GBP / EUR), no thousands separators (write "1234" not "1,234"), no trailing period, no words like "The answer is".
- Good final lines: `36`, `1234.5`, `100`. Bad final lines: `36 signatures`, `$42`, `The answer is 36`, `36.`

## Example
- {"question": "Carol and Jennifer are sisters ... They have 20 + 44 = 64 signatures. Goal is 100. How many more do they need?", "answer": "36"}

## Output Format Contract (MANDATORY - applies to the TERMINAL node only)

1. Terminal output parameter. The terminal (sink) node's `outputs` MUST contain EXACTLY ONE parameter whose `name` is exactly `answer` (lowercase), `type` is `string`, `required` is true.

2. Terminal parse mode. The terminal task spec in the planner JSON MUST set `"parse_mode": "str"`. The agent dict inside that task's `agents` list MUST ALSO set `"parse_mode": "str"`. Do NOT use `"json"` or `"title"` on the terminal node - JSON parsing chokes on arithmetic expressions, and `title` mode misaligns whenever the LLM emits extra `## <name>` headings.

3. Terminal prompt content (must appear verbatim in the terminal prompt). The terminal prompt MUST contain this sentence literally: `Output ONLY the minimal bare-number answer. Do NOT include reasoning, do NOT include units (such as "dollars", "signatures", "hours", "years"), do NOT include currency symbols, do NOT include thousands separators (write "1234" not "1,234"), do NOT wrap the number in JSON or markdown or code fences, do NOT prefix with "The answer is" or "Answer:". The entire response MUST be a single line containing a bare integer or decimal.`

4. No reasoning preamble under the answer. The terminal prompt MUST instruct the LLM NOT to produce a `## Thought`, `## reasoning`, `## steps`, or `## solution` section before the answer. If any `## <name>` heading appears in the response at all, it MUST be only `## answer`, and the bare number MUST be on the line IMMEDIATELY under it (no other text, no trailing clarification line - any number appearing after the bare-number line will be picked up by the scoring regex and destroy solve_rate).

5. Few-shot anchor in the terminal prompt. Immediately before the Output Format block, the terminal prompt MUST include this exact one-line anchor: `Example: for a problem whose correct answer is 36, your entire response MUST be exactly the single line "36" - no units, no currency, no extra words, no trailing sentence.`

## Topology Requirements (CRITICAL - affects solve_rate)

- Prefer a SINGLE node: `solve(goal) -> answer`. This is the AFlow-style baseline and the RECOMMENDED default for GSM8K. GSM8K problems are short grade-school arithmetic; mid-tier executor models solve them in one shot ~90% of the time when the prompt is clean, and multi-node chains typically REDUCE accuracy because they lose the exact original numbers across paraphrased intermediate representations and reintroduce reasoning-prose numbers that confuse the scoring regex.
- Maximum 2 nodes. If a second node is used, it MUST consume `goal` directly and be a pure reasoning step (e.g. `reason(goal) -> working_out` followed by `answer_formatting(goal, working_out) -> answer`). Every node that touches the final answer MUST list `goal` (the original question) in its `inputs`, NOT just an upstream paraphrase.
- FORBIDDEN node kinds - DO NOT generate any of these:
  - Any "verify / validate / cross-check / re-solve / double-check / audit / compare-to-reference" node. There is no oracle at runtime, the LLM cannot reliably self-verify arithmetic, and any boolean gate will overwrite correct answers with retry noise.
  - Any node that outputs a boolean `is_correct`, `verified`, `passed`, `validation_result` field and gates the final answer on it.
  - Any "code execution" / "run calculator" / "execute Python" node - the runtime has no sandbox, the LLM only hallucinates a result, and gating on the hallucinated value destroys solve_rate.
  - Any `answer_formatting` / `format_result` node whose job is to "add units back" or "format nicely" - the benchmark wants a BARE number; any formatting layer risks reintroducing units/prefixes or trailing sentences whose numbers the regex then mis-extracts.
- All intermediate outputs MUST be derivable purely from `goal`; no exogenous data, no reference answer, no gold label, no ground-truth lookup.
"""


DROP_WORKFLOW_GOAL = """
# Task: Design a workflow for Reading Comprehension with Discrete Reasoning (DROP dataset)

## Key Properties of DROP
- DROP (Discrete Reasoning Over Paragraphs) pairs a passage with a question that requires discrete reasoning over the passage text: counting, sorting, date arithmetic, number arithmetic, span extraction, or set operations.
- The answer is a NUMBER, a DATE, a single SPAN copied verbatim from the passage, or MULTIPLE SPANS. Output the minimal surface form (e.g. bare integer "20", not "20 years").
- The workflow receives ONLY the concatenated passage+question as the input named `goal`. There is NO reference answer, NO gold label, and NO verifier available at runtime. Any attempt to reference "the correct answer", "reference answer", or "expected output" is impossible and must not appear in the workflow.

## Example (runtime input/output only — the workflow sees `goal`, must emit `answer`)
- Input `goal`: "Passage: ...The Yamethin rebellion went on until 1500... With a serious rebellion so close to Ava, vassal states broke away one by one. The rebellion started in 1480...\nQuestion: How many years did the Yamethin rebellion last?\nAnswer:"
- Expected `answer`: "20"

## Output Format Contract (MANDATORY)
1. The TERMINAL node MUST have exactly ONE output named `answer`, type `string`, required=true.
2. The terminal node's parse_mode MUST be `"str"` (NOT json / title / xml). No wrapping object, no extra fields.
3. The terminal node's prompt MUST instruct the LLM: "Output ONLY the minimal-surface final answer (a number, a date, or a verbatim span from the passage). Do NOT include explanations, units beyond what the question asks, reasoning, prefixes like 'Answer:' or wrapping quotes."
4. For numeric answers, the final string MUST be the bare number (e.g. `20`, not `20 years` or `twenty`). For span answers, copy verbatim from the passage.
5. The FIRST node MUST accept an input named `goal` of type `string`.

## Topology Requirements (CRITICAL)
- Prefer a SINGLE node: `solve(goal) -> answer`. This is the AFlow-style baseline and is recommended unless a clear intermediate-reasoning subtask justifies more.
- Maximum 2 nodes. If a second node is used, it MUST be a pure reasoning/extraction step that consumes `goal` (or a `goal`-derived string) and whose output feeds the terminal node.
- The terminal node's `inputs` MUST include (directly or indirectly via predecessor output) the original passage+question content.
- FORBIDDEN node kinds — DO NOT generate any of these:
  - Any node whose inputs include `ref_text`, `reference`, `gold`, `expected_answer`, `label`, `ground_truth`, or any synonym of "reference answer".
  - Any "verify / validate / cross-check / compare-to-reference / correctness-check" node — there is no oracle at runtime.
  - Any node named `answer_verification`, `verification`, `verify_*`, `compare_*`, `audit_*`, `finalize_against_*`.
  - Any node that outputs a boolean verdict and gates the final answer on it.
- All intermediate outputs MUST be derivable purely from `goal`; no exogenous data is injected at runtime.
"""



PromptSuggestion = """You are an expert workflow planner specializing in decomposing complex objectives into structured, executable steps.

Given a target task, your job is to design one or more sub-steps that ensure the goal is achieved correctly and efficiently.

# Steps

1. **Understand the goal** — Carefully interpret the task objective and clarify any ambiguities.
2. **Identify workflow nodes** — Break the goal into progressive, discrete sub-steps required to accomplish it.
3. **Define workflow edges** — Determine the sequential dependencies and ordering relationships between steps.
4. **Detect parallelism** — Identify any steps that can be executed concurrently to improve efficiency.
5. **Output the workflow** — Present a clear, valid workflow based on your analysis.

## Critical Constraints
- Input variable MUST be named `goal`.
- Output variable MUST be named `answer`.
"""

# Model routing.
OPTIMIZER_MODEL = "gemini-2.5-flash-lite"
EXECUTOR_MODEL = "gemini-2.5-flash-lite"

# Optimization-time validation policy.
MAX_OPT_STEPS = 10
ADAPTIVE_EVAL_ENABLED = False  #True
ADAPTIVE_SUCCESS_KEEP_RATIO = 0.2
ADAPTIVE_SUCCESS_F1_THRESHOLD = max(0.0, min(1.0, float(os.getenv("ADAPTIVE_SUCCESS_F1_THRESHOLD", "0.3"))))
INITIAL_WORKFLOW_MAX_REGENERATIONS = 3

# AFlow parity targets for fair speed comparison.
AFLOW_PARITY_MAX_CONCURRENT_TASKS = 50
# Test-set evaluation runs MUCH longer per problem (full 486 MATH items, multi-node
# workflow, no early stop) so the 50-way concurrency we use for short validation
# rounds OOM-killed the worker (52GB RSS observed). Cap test-time concurrency
# separately. Tune via env TEST_EVAL_NUM_WORKERS if needed.
TEST_EVAL_NUM_WORKERS = max(1, int(os.getenv("TEST_EVAL_NUM_WORKERS", "8")))


def _normalize_aliyun_api_key(raw_value: str) -> str:
    api_key = str(raw_value or "").strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key.split(None, 1)[1].strip()
    if not api_key:
        raise ValueError(
            "ALIYUN_API_KEY / DASHSCOPE_API_KEY is empty. Please set your Aliyun key before running."
        )
    return api_key


# def build_optimizer_llm() -> AliyunLLM:
#     # qwen-plus
#     api_key = _normalize_aliyun_api_key(ALIYUN_API_KEY)
#     llm_config = AliyunLLMConfig(
#         model=OPTIMIZER_MODEL,
#         aliyun_api_key=api_key,
#         temperature=0.2,
#         max_tokens=4096,
#         timeout=600,
#     )
#     return AliyunLLM(config=llm_config)

    # deepseek

def build_optimizer_llm() -> OpenAILLM:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it or add it to a local .env file "
            "(see .env.example) before running."
        )
    llm_config = OpenAILLMConfig(
        model=OPTIMIZER_MODEL,
        openai_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.5,
        max_tokens=4096,
        timeout=3000,
    )
    return OpenAILLM(config=llm_config)

# def build_executor_llm() -> AliyunLLM:
#     api_key = _normalize_aliyun_api_key(ALIYUN_API_KEY)
#     llm_config = AliyunLLMConfig(
#         model=EXECUTOR_MODEL,
#         aliyun_api_key=api_key,
#         temperature=0.0,
#         max_tokens=4096,
#         timeout=600,
#     )
#     return AliyunLLM(config=llm_config)

def build_executor_llm() -> OpenAILLM:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it or add it to a local .env file "
            "(see .env.example) before running."
        )
    llm_config = OpenAILLMConfig(
        model=EXECUTOR_MODEL,
        openai_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.0,
        max_tokens=4096,
        timeout=3000,
    )
    return OpenAILLM(config=llm_config)

# def build_executor_llm() -> OpenRouterLLM:
#     api_key = os.getenv("OPENROUTER_API_KEY", "")
#     llm_config = OpenRouterConfig(
#         model=EXECUTOR_MODEL,
#         openrouter_key=api_key,
#         temperature=0.0,
#         max_tokens=4096,
#         timeout=600,
#     )
#     return OpenRouterLLM(config=llm_config)


def _all_indices(dataset, mode: str) -> List[int]:
    if mode == "dev":
        data = dataset.get_dev_data()
    elif mode == "test":
        data = dataset.get_test_data()
    else:
        raise ValueError(f"Unsupported mode for AFlowHotPotQA indices: {mode}")
    return list(range(len(data or [])))


def _make_param(name: str, param_type: str, description: str) -> Parameter:
    return Parameter(name=name, type=param_type, description=description, required=True)


def _agent_io_schema(params: List[Parameter]) -> List[Dict[str, Any]]:
    return [param.to_dict(ignore=["class_name"]) for param in params]


def _make_json_agent(name: str, description: str, inputs: List[Parameter], outputs: List[Parameter], prompt: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputs": _agent_io_schema(inputs),
        "outputs": _agent_io_schema(outputs),
        "prompt": prompt,
        "parse_mode": "json",
    }


def _make_text_agent(name: str, description: str, inputs: List[Parameter], outputs: List[Parameter], prompt: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputs": _agent_io_schema(inputs),
        "outputs": _agent_io_schema(outputs),
        "prompt": prompt,
    }


def _validate_cached_workflow(workflow_graph: WorkFlowGraph, source_label: str = "initial workflow") -> WorkFlowGraph:
    canonical_graph, canonical_prompt_repaired, canonical_parse_mode_changed = _legacy._canonicalize_workflow_graph(workflow_graph)
    prompt_contract_repaired, structured_parse_changed = _legacy._enforce_workflow_contracts(canonical_graph)
    valid, reasons, meta = _legacy._validate_workflow_structure_for_evolution(canonical_graph)
    if not valid:
        raise ValueError(
            f"{source_label} failed structural validation: "
            + "; ".join(reasons)
            + f". meta={meta}"
        )
    if canonical_prompt_repaired or canonical_parse_mode_changed or prompt_contract_repaired or structured_parse_changed:
        print(
            f">>> {source_label} safety pass: "
            f"canonical_prompt_repaired={canonical_prompt_repaired}, "
            f"canonical_parse_mode_changed={canonical_parse_mode_changed}, "
            f"prompt_contract_repaired={prompt_contract_repaired}, "
            f"structured_parse_changed={structured_parse_changed}"
        )
    return canonical_graph


def _load_workflow_candidate_from_path(path: Path, source_label: str) -> Optional[WorkFlowGraph]:
    if not path.exists():
        return None
    try:
        workflow_graph = WorkFlowGraph.from_file(str(path))
    except Exception as exc:
        print(f">>> Warning: could not load {source_label} from {path}: {exc}")
        return None
    # MBPP-specific cache invalidation: the new MBPP_WORKFLOW_GOAL prescribes a
    # SINGLE-NODE topology. Any cached MBPP workflow with more than one node is
    # stale (built from the old 4-step QA-style goal) and must be regenerated.
    # if DATASET == "mbpp":
    #     _forbidden_names = {
    #         "parse_problem_spec", "validate_implementation_logic",
    #         "format_final_answer", "analyze_problem", "extract_signature",
    #         "review_code", "finalize_answer",
    #     }
    #     try:
    #         _node_count = len(workflow_graph.nodes) if getattr(workflow_graph, "nodes", None) is not None else 0
    #         _node_names = {getattr(n, "name", "") for n in (workflow_graph.nodes or [])}
    #     except Exception:
    #         _node_count, _node_names = 0, set()
    #     if _node_count > 1 or (_node_names & _forbidden_names):
    #         print(
    #             f">>> MBPP cache at {path} is stale (nodes={_node_count}, "
    #             f"names={sorted(_node_names)}); ignoring and will regenerate from new single-node goal."
    #         )
    #         return None
    return workflow_graph  # _validate_cached_workflow(workflow_graph, source_label=source_label)


def _save_initial_workflow_cache(workflow_graph: WorkFlowGraph, cache_path: Path, source_label: str) -> None:
    try:
        workflow_graph.save_module(str(cache_path))
        print(f">>> Initial workflow cache saved: {cache_path} (source={source_label})")
    except Exception as exc:
        print(f">>> Warning: could not save workflow cache (source={source_label}): {exc}")


def _save_best_workflow(
    workflow_graph: WorkFlowGraph,
    validation_f1: float,
    test_f1: Optional[float] = None,
) -> None:
    """Persist the best optimized workflow so future runs can skip optimization."""
    cache_path = DROP_BEST_WORKFLOW_CACHE_PATH if DATASET == "drop" else (GSM8K_BEST_WORKFLOW_CACHE_PATH if DATASET == "gsm8k" else (MATH_BEST_WORKFLOW_CACHE_PATH if DATASET == "math" else (HUMANEVAL_BEST_WORKFLOW_CACHE_PATH if DATASET == "humaneval" else (MBPP_BEST_WORKFLOW_CACHE_PATH if DATASET == "mbpp" else BEST_WORKFLOW_CACHE_PATH))))
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_graph.save_module(str(cache_path))
        meta_path = cache_path.with_suffix(".meta.json")
        import json as _json
        with open(str(meta_path), "w", encoding="utf-8") as _f:
            _json.dump({
                "validation_f1": round(float(validation_f1), 6),
                "test_f1": round(float(test_f1), 6) if test_f1 is not None else None,
                "saved_at": __import__("datetime").datetime.now().isoformat(),
                "optimizer_model": OPTIMIZER_MODEL,
                "executor_model": EXECUTOR_MODEL,
                "max_opt_steps": MAX_OPT_STEPS,
            }, _f, indent=2)
        _test_f1_str = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
        print(
            f">>> Best workflow saved: {cache_path} "
            f"(validation_f1={validation_f1:.4f}, test_f1={_test_f1_str})"
        )
    except Exception as exc:
        print(f">>> Warning: could not save best workflow: {exc}")


def _workflow_has_expert_agents(workflow_graph: WorkFlowGraph) -> bool:
    """Return True if the workflow has at least one non-fallback custom agent.
    A cache populated only by generic fallback agents is low-quality and should
    be replaced by the curated seed workflow.
    """
    total, fallback = 0, 0
    for node in workflow_graph.nodes:
        for agent in (getattr(node, "agents", None) or []):
            name = (
                agent.get("name", "") if isinstance(agent, dict)
                else getattr(agent, "name", "")
            )
            total += 1
            if "fallback" in str(name).lower():
                fallback += 1
    if total == 0:
        return False
    # Treat as expert only if fewer than half of all agents are fallback agents.
    return fallback / total < 0.5



def _workflow_is_curated_seed(workflow_graph: WorkFlowGraph) -> bool:
    """Detect the curated fallback seed so it is not reused as a strong cache.

    The curated seed is a safe fallback, but reusing it forever blocks later runs
    from regenerating a stronger LLM-authored initial workflow.
    """
    total_agents = 0
    curated_agents = 0
    for node in workflow_graph.nodes:
        for agent in (getattr(node, "agents", None) or []):
            total_agents += 1
            if isinstance(agent, dict):
                parts = [agent.get("name", ""), agent.get("description", ""), agent.get("prompt", "")]
            else:
                parts = [getattr(agent, "name", ""), getattr(agent, "description", ""), getattr(agent, "prompt", "")]
            fingerprint = " ".join(str(part or "") for part in parts).lower()
            if "curated hotpotqa" in fingerprint:
                curated_agents += 1
    return total_agents > 0 and curated_agents == total_agents



def _load_or_generate_initial_workflow(
    optimizer_llm,
) -> Tuple[WorkFlowGraph, str, Dict[str, Any]]:
    is_math = DATASET == "math"
    is_humaneval = DATASET == "humaneval"
    is_mbpp = DATASET == "mbpp"
    is_gsm8k = DATASET == "gsm8k"
    is_drop = DATASET == "drop"
    if is_drop:
        _best_cache = DROP_BEST_WORKFLOW_CACHE_PATH
        _init_cache = DROP_INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = DROP_WORKFLOW_GOAL
    elif is_gsm8k:
        _best_cache = GSM8K_BEST_WORKFLOW_CACHE_PATH
        _init_cache = GSM8K_INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = GSM8K_WORKFLOW_GOAL
    elif is_math:
        _best_cache = MATH_BEST_WORKFLOW_CACHE_PATH
        _init_cache = MATH_INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = MATH_WORKFLOW_GOAL
    elif is_humaneval:
        _best_cache = HUMANEVAL_BEST_WORKFLOW_CACHE_PATH
        _init_cache = HUMANEVAL_INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = HUMANEVAL_WORKFLOW_GOAL
    elif is_mbpp:
        _best_cache = MBPP_BEST_WORKFLOW_CACHE_PATH
        _init_cache = MBPP_INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = MBPP_WORKFLOW_GOAL
    else:
        _best_cache = BEST_WORKFLOW_CACHE_PATH
        _init_cache = INITIAL_WORKFLOW_CACHE_PATH
        _workflow_goal = HOTPOTQA_WORKFLOW_GOAL

    # Priority 0: if USE_BEST_WORKFLOW=1, load the saved best workflow and skip optimization.
    if USE_BEST_WORKFLOW:
        best_wf = _load_workflow_candidate_from_path(_best_cache, "best_workflow_cache")
        if best_wf is not None:
            print(f">>> USE_BEST_WORKFLOW=1: loaded saved best workflow from {_best_cache}")
            return best_wf, "best_workflow_cache", {
                "selected_source": "best_workflow_cache",
                "cache_path": str(_best_cache),
                "mode": "best_workflow_direct",
            }
        print(">>> USE_BEST_WORKFLOW=1 but no saved best workflow found; falling back to normal flow.")

    cache_path = _init_cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cached_workflow = _load_workflow_candidate_from_path(cache_path, source_label="current_cache")
    if cached_workflow is not None:
        return cached_workflow, "cache", {
            "selected_source": "cache",
            "cache_path": str(cache_path),
            "mode": "single_cache_or_generate",
            "max_regenerations": INITIAL_WORKFLOW_MAX_REGENERATIONS,
        }
    
    print(f">>> Generating LLM initial workflow for {DATASET.upper()} ...")
    source_label = "generated"

    wf_generator = WorkFlowGenerator(llm=optimizer_llm, tools=None)
    workflow_graph = _legacy._generate_valid_initial_workflow(
        wf_generator=wf_generator,
        base_goal=_workflow_goal,
        max_regenerations=INITIAL_WORKFLOW_MAX_REGENERATIONS,
        suggestion=PromptSuggestion,
    )
    workflow_graph.save_module(_init_cache)
    return workflow_graph, source_label, {
        "selected_source": source_label,
        "cache_path": str(cache_path),
        "mode": "single_cache_or_generate",
        "max_regenerations": INITIAL_WORKFLOW_MAX_REGENERATIONS,
    }


def build_optimizer(
    optimizer_llm,
    executor_llm,
    *,
    eval_mode: str,
    fixed_eval_indices: List[int],
    eval_seed: int,
    initial_graph: WorkFlowGraph,
) -> LLMWorkflowOptimizer:
    return LLMWorkflowOptimizer(
        graph=initial_graph,
        evaluator=None,
        llm=optimizer_llm,
        executor_llm=executor_llm,
        workflow_goal=DROP_WORKFLOW_GOAL if DATASET == "drop" else (GSM8K_WORKFLOW_GOAL if DATASET == "gsm8k" else (MATH_WORKFLOW_GOAL if DATASET == "math" else (HUMANEVAL_WORKFLOW_GOAL if DATASET == "humaneval" else (MBPP_WORKFLOW_GOAL if DATASET == "mbpp" else HOTPOTQA_WORKFLOW_GOAL)))),
        max_steps=MAX_OPT_STEPS,
        sample_k=len(fixed_eval_indices),
        eval_seed=eval_seed,
        eval_mode=eval_mode,
        fixed_eval_indices=list(fixed_eval_indices),
        target_f1=1.00,
        no_improve_patience=6,
        max_targets=5,
        strong_rca_threshold=0.05,
        prompt_retry_per_node=3,
        num_workers=AFLOW_PARITY_MAX_CONCURRENT_TASKS,
        planner_candidate_count=2,
        planner_repair_rounds=2,
        adaptive_eval_enabled=ADAPTIVE_EVAL_ENABLED,
        adaptive_success_sample_ratio=ADAPTIVE_SUCCESS_KEEP_RATIO,
        adaptive_success_f1_threshold=ADAPTIVE_SUCCESS_F1_THRESHOLD,
    )


def run_paper_aligned_experiment():
    optimizer_llm = build_optimizer_llm()
    executor_llm = build_executor_llm()
    if DATASET == "math":
        validation_benchmark = AFlowMATH(path=str(REPO_ROOT / "data" / "datasets"), mode="dev")
        test_benchmark = AFlowMATH(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "humaneval":
        validation_benchmark = AFlowHumanEval(path=str(REPO_ROOT / "data" / "datasets"), mode="dev")
        test_benchmark = AFlowHumanEval(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "mbpp":
        validation_benchmark = AFlowMBPP(path=str(REPO_ROOT / "data" / "datasets"), mode="dev")
        test_benchmark = AFlowMBPP(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "drop":
        validation_benchmark = AFlowDROP(path=str(REPO_ROOT / "data" / "datasets"), mode="dev")
        test_benchmark = AFlowDROP(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "gsm8k":
        validation_benchmark = AFlowGSM8K(path=str(REPO_ROOT / "data" / "datasets"), mode="dev")
        test_benchmark = AFlowGSM8K(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    else:
        validation_benchmark = AFlowHotPotQA(mode="dev")
        test_benchmark = AFlowHotPotQA(mode="test")
    initial_workflow, initial_workflow_source, initial_workflow_selection_meta = _load_or_generate_initial_workflow(
        optimizer_llm,
    )
    validation_indices = _all_indices(validation_benchmark, mode="dev")
    test_indices = _all_indices(test_benchmark, mode="test")

    print(
        ">>> Experiment config: "
        f"optimizer_model={OPTIMIZER_MODEL}, executor_model={EXECUTOR_MODEL}, "
        f"optimizer_channel=aliyun, executor_channel=aliyun, "
        f"resolved_optimizer_model={optimizer_llm.config.model}, resolved_executor_model={executor_llm.config.model}, "
        f"validation_mode=dev, test_mode=test, "
        f"validation_size={len(validation_indices)}, heldout_test_size={len(test_indices)}, "
        f"benchmark_data_path={validation_benchmark.path}, adaptive_eval_enabled={ADAPTIVE_EVAL_ENABLED}, "
        f"aflow_parity_max_concurrent_tasks={AFLOW_PARITY_MAX_CONCURRENT_TASKS}, "
        f"adaptive_success_keep_ratio={ADAPTIVE_SUCCESS_KEEP_RATIO}, adaptive_success_f1_threshold={ADAPTIVE_SUCCESS_F1_THRESHOLD}, "
        f"initial_workflow_cache={initial_workflow_selection_meta.get('cache_path') or INITIAL_WORKFLOW_CACHE_PATH}, initial_workflow_source={initial_workflow_source}, "
        f"initial_workflow_max_regenerations={INITIAL_WORKFLOW_MAX_REGENERATIONS}, "
        f"initial_workflow_selection_mode=single_cache_or_generate"
    )
    print(f">>> Validation indices first10: {validation_indices[:10]}")
    print(f">>> Held-out test indices first10: {test_indices[:10]}")
    print(f">>> Initial workflow selection meta: {initial_workflow_selection_meta}")

    optimizer = build_optimizer(
        optimizer_llm,
        executor_llm,
        eval_mode="dev",
        fixed_eval_indices=validation_indices,
        eval_seed=42,
        initial_graph=initial_workflow,
    )

    if initial_workflow_source == "best_workflow_cache":
        
        print(">>> USE_BEST_WORKFLOW=1: skipping optimization, evaluating saved best workflow directly.")
        best_graph = initial_workflow
        payload: Dict[str, Any] = {
            "best_workflow": best_graph,
            "workflow_graph": best_graph,
            "best_metrics": {},
            "skipped_optimization": True,
        }
    else:
        payload = optimizer.optimize(validation_benchmark)
        best_graph = payload.get("best_workflow") or payload.get("workflow_graph")
        if best_graph is None:
            raise RuntimeError("Optimization finished without a usable workflow graph.")

        # Save best workflow immediately after validation optimization (before test eval).
        _validation_f1 = float(payload.get("estimated_full_f1", 0.0)
                               or (payload.get("best_results") or {}).get("estimated_full_f1", 0.0)
                               or (payload.get("best_results") or {}).get("f1", 0.0) or 0.0)
        _save_best_workflow(best_graph, validation_f1=_validation_f1, test_f1=None)

    # Throttle worker count for the (much longer) test-set evaluation to
    # avoid the OOM kill we hit when reusing optimizer.num_workers=50.
    optimizer.num_workers = TEST_EVAL_NUM_WORKERS
    print(f">>> Test-set evaluation throttled to num_workers={TEST_EVAL_NUM_WORKERS} to avoid OOM.")
    paper_test_metrics = optimizer.evaluate(
        test_benchmark,
        eval_mode="test",
        graph=best_graph,
        indices=test_indices,
    )
    print(f">>> Paper held-out test metrics: {paper_test_metrics}")

    payload["paper_validation_indices"] = validation_indices
    payload["paper_test_indices"] = test_indices
    payload["paper_validation_size"] = len(validation_indices)
    payload["paper_test_size"] = len(test_indices)
    payload["paper_test_metrics"] = paper_test_metrics
    payload["optimizer_model"] = OPTIMIZER_MODEL
    payload["executor_model"] = EXECUTOR_MODEL
    payload["resolved_optimizer_model"] = optimizer_llm.config.model
    payload["resolved_executor_model"] = executor_llm.config.model
    payload["benchmark_data_path"] = validation_benchmark.path
    payload["adaptive_eval_enabled"] = ADAPTIVE_EVAL_ENABLED
    payload["adaptive_success_keep_ratio"] = ADAPTIVE_SUCCESS_KEEP_RATIO
    payload["adaptive_success_f1_threshold"] = ADAPTIVE_SUCCESS_F1_THRESHOLD
    payload["initial_workflow_cache_path"] = str(initial_workflow_selection_meta.get("cache_path") or INITIAL_WORKFLOW_CACHE_PATH)
    payload["initial_workflow_source"] = initial_workflow_source
    payload["initial_workflow_max_regenerations"] = INITIAL_WORKFLOW_MAX_REGENERATIONS
    payload["initial_workflow_selection_mode"] = "single_cache_or_generate"
    payload["initial_workflow_selection_meta"] = initial_workflow_selection_meta
    payload["aflow_parity_max_concurrent_tasks"] = AFLOW_PARITY_MAX_CONCURRENT_TASKS



    return payload


def run_test_only():
    """Evaluate the saved best workflow on the full test set without re-running optimization."""
    import argparse
    executor_llm = build_executor_llm()
    optimizer_llm = build_optimizer_llm()

    _best_cache = DROP_BEST_WORKFLOW_CACHE_PATH if DATASET == "drop" else (GSM8K_BEST_WORKFLOW_CACHE_PATH if DATASET == "gsm8k" else (MATH_BEST_WORKFLOW_CACHE_PATH if DATASET == "math" else (HUMANEVAL_BEST_WORKFLOW_CACHE_PATH if DATASET == "humaneval" else (MBPP_BEST_WORKFLOW_CACHE_PATH if DATASET == "mbpp" else BEST_WORKFLOW_CACHE_PATH))))
    # Load best workflow
    best_wf = _load_workflow_candidate_from_path(_best_cache, "best_workflow_for_test")
    if best_wf is None:
        raise FileNotFoundError(
            f"No saved best workflow found at {_best_cache}. "
            "Run optimization first, then use --test-only."
        )
    print(f">>> Loaded best workflow from {_best_cache}")

    if DATASET == "drop":
        test_benchmark = AFlowDROP(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "gsm8k":
        test_benchmark = AFlowGSM8K(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "math":
        test_benchmark = AFlowMATH(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "humaneval":
        test_benchmark = AFlowHumanEval(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    elif DATASET == "mbpp":
        test_benchmark = AFlowMBPP(path=str(REPO_ROOT / "data" / "datasets"), mode="test")
    else:
        test_benchmark = AFlowHotPotQA(mode="test")
    test_indices = _all_indices(test_benchmark, mode="test")
    print(f">>> Test set size: {len(test_indices)}")

    # Build a minimal optimizer just for the evaluate() method
    optimizer = build_optimizer(
        optimizer_llm,
        executor_llm,
        eval_mode="test",
        fixed_eval_indices=test_indices,
        eval_seed=42,
        initial_graph=best_wf,
    )

    optimizer.num_workers = TEST_EVAL_NUM_WORKERS
    print(f">>> Test-set evaluation throttled to num_workers={TEST_EVAL_NUM_WORKERS} to avoid OOM.")
    test_metrics = optimizer.evaluate(
        test_benchmark,
        eval_mode="test",
        graph=best_wf,
        indices=test_indices,
    )
    print(f"\n{'='*60}")
    print(f">>> Test-only evaluation results:")
    print(f">>>   F1:  {test_metrics.get('f1', 0.0):.4f}")
    print(f">>>   EM:  {test_metrics.get('em', 0.0):.4f}")
    print(f">>>   Count: {test_metrics.get('count', 'N/A')}")
    print(f"{'='*60}")
    return test_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark & self-evolution (HotPotQA / MATH / HumanEval)")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip optimization; evaluate saved best workflow on the full test set")
    args = parser.parse_args()

    if args.test_only:
        run_test_only()
    else:
        run_paper_aligned_experiment()


if __name__ == "__main__":
    main()