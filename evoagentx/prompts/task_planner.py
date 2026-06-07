TASK_PLANNER_DESC = "TaskPlanner is an intelligent workflow planning agent that turns complex goals into clear, reusable executable workflows with meaningful intermediate outputs."

TASK_PLANNER_SYSTEM_PROMPT = "You are a highly skilled workflow planning expert. Your role is to analyze the user's goal, decompose it into clear and reusable sub-tasks, and organize those sub-tasks into a strong execution structure with minimal redundancy and explicit dependencies."

TASK_PLANNER = {
    "name": "TaskPlanner",
    "description": TASK_PLANNER_DESC,
    "system_prompt": TASK_PLANNER_SYSTEM_PROMPT,
}


TASK_PLANNING_ACTION_DESC = "This action analyzes a user goal, breaks it into high-quality sub-tasks, and organizes them into a reusable workflow with clear dependencies and informative intermediate outputs."

TASK_PLANNING_ACTION_PROMPT_OLD = ""

TASK_PLANNING_ACTION_INST = """
Your Task: Given a user's goal, design a reusable workflow made of clear sub-tasks that are easy to execute, reason about, and improve later.

### Instructions:
1. **Understand the Goal**: Identify the core objective, required deliverable, constraints, and any bundled context.
2. **Review the History**: If a previous plan exists, use it to identify missing structure, weak decomposition, or redundant steps.
3. **Consider Suggestions**: Incorporate the provided suggestions when they improve workflow quality.
4. **Define Sub-Tasks**: Break the task into logical, non-overlapping sub-tasks with explicit dependencies.

4.1 **Principles for Designing the Workflow**:
- **Start with the smallest non-trivial workflow**: Prefer 3 to 6 connected sub-tasks unless the goal is truly simple.
- **Prefer specialized nodes**: Each sub-task should have one primary responsibility instead of repeating the same behavior as neighboring steps.
- **Use meaningful intermediate representations**: Non-final sub-tasks should output high-information variables that make downstream work easier and more precise. Avoid vague names like `data`, `result`, or `info` unless they are clearly qualified.
- **Solve the runtime task, not workflow design**: The workflow must act on the user's `goal` and produce the task answer. The workflow object you return now is already the final planning artifact, so the nodes inside it must be future runtime steps, not steps for designing or serializing the workflow itself.
- **Explicitly avoid meta-workflow nodes**: Do not create nodes like `workflow_design`, `workflow_synthesis`, `answer_serialization`, `constraint_extraction`, `goal_analysis`, `task_classification`, or outputs like `workflow_skeleton`, `synthesized_workflow`, `required_inputs`, `expected_output`, `functional_roles_needed`, `structural_constraints`, or `execution_plan` unless the user's runtime task explicitly asks for those artifacts.
- **Do not force every sub-task to read the global `goal`**: The first sub-task must read `goal`, but later sub-tasks should primarily consume outputs from preceding sub-tasks. Include `goal` again only when the full original context is truly needed.
- **Cover multiple functional roles**: Aim to cover at least three of these roles when useful: understanding/scoping, gathering/extracting signals, transforming/organizing information, reasoning/deciding, verifying/refining, and final answer generation.
- **Keep dependencies explicit**: Inputs of each sub-task must be chosen only from the user's `goal` and outputs of earlier sub-tasks.
- **Prefer mostly acyclic workflows**: Only introduce optional feedback inputs when iterative refinement is genuinely useful.
- **Preserve the global interface**: The workflow must start from `goal` and the final node must output `answer`.

4.2 **Sub-Task Format**:
Each sub-task should follow the structure below:
```json
{{
    "name": "subtask_name",
    "description": "A clear and concise explanation of the goal of this sub-task.",
    "reason": "Why this sub-task is necessary and how it contributes to achieving the user's goal.",
    "inputs": [
        {{
            "name": "the input's name",
            "type": "string/int/float/other_type",
            "required": true/false (only set to `false` when this input is optional feedback or a refinement signal),
            "description": "Description of the input's purpose and usage."
        }},
        ...
    ],
    "outputs": [
        {{
            "name": "the output's name",
            "type": "string/int/float/other_type",
            "required": true,
            "description": "Description of the output produced by this sub-task."
        }},
        ...
    ]
}}
```

### Notes:
- Provide concise, meaningful names for sub-tasks, inputs, and outputs.
- The first sub-task must have only one input named `goal`.
- Later sub-tasks should include `goal` only when the full original context is necessary.
- The workflow should directly solve the user's runtime task; do not output a workflow spec, workflow skeleton, role checklist, task-schema analysis, or workflow-serialization node unless that is the actual target answer.
- Ensure that at least one non-final output is consumed by a later sub-task.
- If a sub-task uses feedback from a later sub-task, include that feedback input and set `required` to `false`.
"""

TASK_PLANNING_ACTION_DEMOS = """
### Example 1:
### User's goal:
Review a collection of customer interview notes and produce the three highest-priority product improvements.
### Generated Workflow:
{{
    "sub_tasks": [
        {{
            "name": "task_understanding",
            "description": "Clarify the outcome, scope, and decision criteria for the recommendation workflow.",
            "reason": "A structured task frame helps downstream steps focus on the right evidence and output format.",
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string",
                    "required": true,
                    "description": "The user's goal in textual format."
                }}
            ],
            "outputs": [
                {{
                    "name": "task_frame",
                    "type": "string",
                    "required": true,
                    "description": "A structured summary of the goal, constraints, and evaluation criteria."
                }}
            ]
        }},
        {{
            "name": "signal_extraction",
            "description": "Extract the most important customer pain points and supporting evidence from the interview notes.",
            "reason": "This converts raw notes into a reusable evidence layer instead of forcing later steps to reread the entire goal.",
            "inputs": [
                {{
                    "name": "goal",
                    "type": "string",
                    "required": true,
                    "description": "The original goal and bundled interview notes."
                }},
                {{
                    "name": "task_frame",
                    "type": "string",
                    "required": true,
                    "description": "The structured task frame that defines what to extract and prioritize."
                }}
            ],
            "outputs": [
                {{
                    "name": "customer_signals",
                    "type": "list[string]",
                    "required": true,
                    "description": "The extracted customer pain points and supporting evidence snippets."
                }}
            ]
        }},
        {{
            "name": "prioritization",
            "description": "Evaluate the extracted signals against the task frame and turn them into ranked recommendations.",
            "reason": "A dedicated prioritization step creates a strong intermediate decision artifact before the final answer is written.",
            "inputs": [
                {{
                    "name": "task_frame",
                    "type": "string",
                    "required": true,
                    "description": "The structured task frame that defines success criteria."
                }},
                {{
                    "name": "customer_signals",
                    "type": "list[string]",
                    "required": true,
                    "description": "The extracted customer pain points and supporting evidence."
                }}
            ],
            "outputs": [
                {{
                    "name": "ranked_recommendations",
                    "type": "string",
                    "required": true,
                    "description": "A ranked list of recommended improvements with concise justification."
                }}
            ]
        }},
        {{
            "name": "final_answer_generation",
            "description": "Produce the final concise answer using the ranked recommendations.",
            "reason": "This isolates answer formatting from evidence processing and prioritization logic.",
            "inputs": [
                {{
                    "name": "ranked_recommendations",
                    "type": "string",
                    "required": true,
                    "description": "The ranked recommendation artifact prepared for final delivery."
                }}
            ],
            "outputs": [
                {{
                    "name": "answer",
                    "type": "string",
                    "required": true,
                    "description": "The final prioritized recommendation answer for the user."
                }}
            ]
        }}
    ]
}}
"""

TASK_PLANNING_OUTPUT_FORMAT = """
### Output Format
Your final output should ALWAYS be in the following format:

## Thought
Provide a brief explanation of why the workflow structure is appropriate for the goal.

## Goal
Restate the user's goal clearly and concisely.

## Plan
You MUST provide the workflow plan in the following JSON format. The description of each sub-task MUST STRICTLY follow the JSON format described in the **Sub-Task Format** section. If a sub-task does not require inputs or outputs, still include `inputs` and `outputs` as empty lists.
```json
{{
    "sub_tasks": [
        {{
            "name": "subtask_name",
            ...
        }},
        {{
            "name": "another_subtask_name",
            ...
        }}
    ]
}}
```

-----
Let's begin.

### History (previously generated task plan):
{history}

### Suggestions (ideas for improving or refining the workflow):
{suggestion}

### User's Goal:
{goal}

Output:
"""

TASK_PLANNING_ACTION_PROMPT = TASK_PLANNING_ACTION_INST + TASK_PLANNING_ACTION_DEMOS + TASK_PLANNING_OUTPUT_FORMAT

TASK_PLANNING_ACTION = {
    "name": "TaskPlanning",
    "description": TASK_PLANNING_ACTION_DESC,
    "prompt": TASK_PLANNING_ACTION_PROMPT,
}
