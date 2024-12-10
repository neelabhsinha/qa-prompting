from rouge_score import rouge_scorer


class RougeScore:
    def __init__(self, use_stemmer=True):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)

    def get_score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        scores = []
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(pred, ref)
            scores.append({
                "rouge1": score["rouge1"].fmeasure,
                "rouge2": score["rouge2"].fmeasure,
                "rougeL": score["rougeL"].fmeasure
            })
        return scores
