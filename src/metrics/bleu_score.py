from typing import List

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import word_tokenize
import nltk


class BleuScore:
    def __init__(self, type='corpus'):
        nltk.download('punkt')
        self._type = type
        self.chencherry = SmoothingFunction()

    def get_score(self, candidates: List[str], references: List[str]):
        bleu_scores = []
        reference_lists: List[List[List[str]]] = [[word_tokenize(ref)] for ref in references]
        candidate_tokens: List[List[str]] = [word_tokenize(candidate) for candidate in candidates]
        for candidate, reference in zip(candidate_tokens, reference_lists):
            score = corpus_bleu([reference], [candidate], smoothing_function=self.chencherry.method1)
            bleu_scores.append(score)
        return bleu_scores
