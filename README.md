# CGE-Agent

Official code for our paper on **self-evolving LLM agentic workflows**.
Starting from a task goal, CGE-Agent builds an initial multi-agent workflow and
iteratively evolves it, guided by causal root-cause analysis over execution
traces. Built on top of [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX).

> Paper: *<TODO: title / authors / venue>*

## Installation

Requires **Python ≥ 3.10**.

```bash
conda create -n cge-agent python=3.11 -y
conda activate cge-agent
pip install -e .
```

## Configuration

LLM credentials are read from environment variables (no keys are hard-coded).
Copy the template and fill in your own values:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...                      # your key
OPENAI_BASE_URL=https://api.openai.com/v1  # or any OpenAI-compatible gateway
```

`.env` is git-ignored. The model is set via `OPTIMIZER_MODEL` / `EXECUTOR_MODEL`
at the top of `examples/benchmark_and_evaluation.py` (the paper uses
`gemini-2.5-flash-lite` served through an OpenAI-compatible gateway).

## Run

The entry point is `examples/benchmark_and_evaluation.py`:

```bash
# optimize on the validation split, then evaluate on the held-out test split
python examples/benchmark_and_evaluation.py

# skip optimization and evaluate the bundled best workflow on the test set
python examples/benchmark_and_evaluation.py --test-only
```

Pick the dataset by editing `DATASET` near the top of that file
(`"hotpotqa" | "math" | "gsm8k" | "humaneval" | "mbpp" | "drop"`).
Benchmark splits are bundled under `data/datasets/`; pre-evolved workflows are
under `data/workflows/`.
