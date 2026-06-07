import os
import re
import string
from pathlib import Path
from collections import Counter
from typing import Any, Callable, List, Tuple, Union

from ..core.logging import logger
from .benchmark import Benchmark
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


def _default_repo_drop_data_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "data" / "datasets")


class AFlowDROP(Benchmark):
    """Benchmark class for evaluating reading comprehension on DROP dataset.

    DROP (Discrete Reasoning Over Paragraphs) requires discrete reasoning
    over passages: counting, sorting, arithmetic, etc. Answers can be
    numbers, entity names, or date spans; multiple accepted answers are
    separated by "|".

    Each AFlow-formatted DROP example:
    {
        "context": "Passage: ... Question: ... Answer:",
        "completion": " 20",
        "ref_text": "20",
        "id": "4032"
    }

    Evaluation uses token-level F1 (matching AFlow DROPBenchmark).
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.abspath(os.path.expanduser(path or _default_repo_drop_data_dir()))
        os.makedirs(path, exist_ok=True)
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="drop", save_folder=self.path)
        logger.info(f"Loading DROP data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = None
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(
                file_name=AFLOW_DATASET_FILES_MAP["drop"]["dev"]
            )
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(
                file_name=AFLOW_DATASET_FILES_MAP["drop"]["test"]
            )

    def _get_label(self, example: Any) -> Any:
        return example["ref_text"]

    def _get_id(self, example: Any) -> Any:
        return example["id"]

    # ---- AFlow-identical normalize_answer & F1 scoring ----

    def normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def calculate_score(self, ground_truth: str, prediction: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return float(f1)

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate matching AFlow evaluate_problem: split by |, compute max F1."""
        prediction_text = str(prediction or "").strip()
        label_text = str(label or "").strip()
        answers = label_text.split("|")
        output_parts = prediction_text.split("|")

        f1_scores = []
        for answer in answers:
            if answer.strip():
                for output_part in output_parts:
                    f1 = self.calculate_score(answer.strip(), output_part.strip())
                    f1_scores.append(f1)

        score = max(f1_scores) if f1_scores else 0.0
        return {"f1": score}

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        prompt = example["context"]
        solution = await graph(prompt)
        label = self._get_label(example)
        metrics = self.evaluate(prediction=solution, label=label)
        return metrics["f1"]
