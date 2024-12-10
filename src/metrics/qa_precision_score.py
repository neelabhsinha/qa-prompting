from typing import List
from collections import Counter
import re
import string


class QAPrecisionScore:
    def __init__(self):
        pass

    @staticmethod
    def _normalize_answer(s: str) -> List[str]:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
        return normalized.split()

    @staticmethod
    def _precision(prediction_tokens: List[str], ground_truth_tokens: List[str]) -> float:
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0

        prediction_count = Counter(prediction_tokens)
        ground_truth_count = Counter(ground_truth_tokens)
        common_count = sum((prediction_count & ground_truth_count).values())
        precision = common_count / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
        return precision

    def get_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self._normalize_answer(prediction)
        ground_truth_tokens = self._normalize_answer(ground_truth)
        return self._precision(prediction_tokens, ground_truth_tokens)
