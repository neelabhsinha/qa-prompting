from evaluate import load
from const import cache_dir


class BertScore:
    def __init__(self, model_type='roberta-large', verbose=False, batch_size=256):
        self._bert_score = load('bertscore', cache_dir=cache_dir)
        self._model_type = model_type
        self._verbose = verbose
        self.batch_size = batch_size

    def get_score(self, predictions, references):
        scores = self._bert_score.compute(predictions=predictions, references=references, verbose=self._verbose,
                                          model_type=self._model_type, batch_size=self.batch_size)
        return scores
