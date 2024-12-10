from evaluate import load
from const import cache_dir


class Meteor:
    def __init__(self):
        self.meteor = load('meteor', cache_dir=cache_dir)

    def get_score(self, candidates, references):
        scores = [self.meteor.compute(predictions=[cand], references=[ref])["meteor"] for cand, ref in
                  zip(candidates, references)]
        return scores
