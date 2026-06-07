import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Tuple
from .benchmark import Benchmark
from .measures import exact_match_score, f1_score, acc_score
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


HOTPOTQA_FILES_MAP = {"train": "hotpot_train_v1.1.json", "dev": "hotpot_dev_distractor_v1.json", "test": None}
VALIDE_RAW_HOTPOTQA_FILES = [file for file in list(HOTPOTQA_FILES_MAP.values()) if file is not None]

def _default_repo_hotpotqa_data_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "data" / "datasets")

def download_raw_hotpotqa_data(name: str, save_folder: str):

    assert name in VALIDE_RAW_HOTPOTQA_FILES, f"'{name}' is an invalid hotpotqa file name. Available file names: {VALIDE_RAW_HOTPOTQA_FILES}"
    url = f"http://curtis.ml.cmu.edu/datasets/hotpot/{name}"
    typ = "train" if "train" in name else "dev"
    logger.info(f"Downloading HotPotQA {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


class HotPotQA(Benchmark):

    """Benchmark class for evaluating multi-hop question answering on HotPotQA dataset.
    
    Each HotPotQA example has the following structure:
    {
        "_id": str, 
        "question": str, 
        "answer": str, 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", supporting_sentence_index]],
        "type": str,
        "level": str
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.abspath(os.path.expanduser(path or _default_repo_hotpotqa_data_dir()))
        os.makedirs(path, exist_ok=True)
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_hotpotqa_data(name=file_name, save_folder=self.path)
        logger.info(f"loading HotPotQA data from {file_path} ...")
        return load_json(path=file_path, type="json")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["test"])
    
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        return example["_id"]
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        em = exact_match_score(prediction=prediction, ground_truth=label)
        f1 = f1_score(prediction=prediction, ground_truth=label)
        acc = acc_score(prediction=prediction, ground_truths=[label])
        return {"f1": f1, "em": em, "acc": acc}
    

class AFlowHotPotQA(HotPotQA):

    """
    AFlow-specific implementation of HotPotQA benchmark.
    """

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            try:
                download_aflow_benchmark_data(dataset="hotpotqa", save_folder=self.path)
            except Exception as exc:
                raise RuntimeError(
                    "AFlow HotPotQA data is missing locally and automatic download failed. "
                    f"Please place '{file_name}' under '{self.path}' manually. Original error: {exc}"
                ) from exc
        logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["test"])

    def normalize_answer(self, text: str) -> str:
        def remove_articles(value: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", value)

        def white_space_fix(value: str) -> str:
            return " ".join(value.split())

        def remove_punc(value: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in value if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(str(text or "").lower())))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        prediction_text = str(prediction or "")
        prediction_tokens = self.normalize_answer(prediction_text).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0, prediction_text
        precision = num_same / max(1, len(prediction_tokens))
        recall = num_same / max(1, len(ground_truth_tokens))
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        return float(f1), prediction_text

    def evaluate(self, prediction: Any, label: Any) -> dict:
        score, _ = self.calculate_score(ground_truth=str(label or ""), prediction=str(prediction or ""))
        return {"f1": float(score)}
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        # generate solution 
        prompt = example["question"]
        paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:"
        solution = await graph(inputs)
        label = self._get_label(example)
        metrics = self.evaluate(prediction=solution, label=label)
        return metrics["f1"]