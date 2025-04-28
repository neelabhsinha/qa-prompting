from const import summary_outputs_dir
from src.handler.exit_handler import EXIT
from src.utils.results_io_util import write_results
from src.metrics.bert_score import BertScore
from src.metrics.rouge import RougeScore
from src.metrics.mover_score import MoverScore

import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def compute_metrics(force_recompute=False):
    skip_existing = False if force_recompute else True
    files = os.listdir(summary_outputs_dir)
    bert_score_calculator = BertScore()
    mover_score_calculator = MoverScore()
    rouge_calculator = RougeScore()
    for file in tqdm(files, desc=f'Calculating metrics for results'):
        path = os.path.join(summary_outputs_dir, file, 'predictions.csv')
        try:
            df = pd.read_csv(path)
            predictions = df['candidate'].fillna('').tolist()
            references = df['reference'].fillna('').tolist()
            # if not skip_existing or ('mover_score' not in df.columns):
            #     scores = mover_score_calculator.get_score(predictions, references)
            #     df['mover_score'] = np.array(scores) * 100
            if not skip_existing or ('bert_score_recall' not in df.columns or 'bert_score_f1' not in df.columns
                                          or 'bert_score_precision' not in df.columns):
                scores = bert_score_calculator.get_score(predictions, references)
                f1 = np.array(scores['f1']) * 100
                recall = np.array(scores['recall']) * 100
                precision = np.array(scores['precision']) * 100
                df['bert_score_precision'] = precision
                df['bert_score_recall'] = recall
                df['bert_score_f1'] = f1
            if not skip_existing or (
                    'rouge1' not in df.columns or 'rouge2' not in df.columns or 'rougeL' not in df.columns):
                scores = rouge_calculator.get_score(predictions, references)
                df['rouge1'] = np.array(scores['rouge1']) * 100
                df['rouge2'] = np.array(scores['rouge2']) * 100
                df['rougeL'] = np.array(scores['rougeL']) * 100
            write_results(df, os.path.join(summary_outputs_dir, file))
        except FileNotFoundError:
            print(f'Prediction file {file} not found in the given folder')
        if EXIT.is_set():
            return