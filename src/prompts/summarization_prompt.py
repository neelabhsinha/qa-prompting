import ast

import pandas as pd
import json

from const import qa_outputs_dir, prompt_elements_dir
from src.prompts.qa_prompts import QuestionAnsweringPrompt


class SummarizationPrompt:
    def __init__(self, model_name, baseline=False, top_k=5):
        self.top_k_questions = pd.read_csv(f'{qa_outputs_dir}/{model_name}/top_k_columns.csv',
                                           index_col='domain')
        self.top_k_questions['top_k_columns'] = self.top_k_questions['top_k_columns'].apply(lambda x: x.split(', '))
        self.top_k_questions = self.top_k_questions.fillna('')
        with open(f'{prompt_elements_dir}/{model_name}/prompt_elements.json', 'r') as f:
            self.prompt_elements = json.load(f)
        self.questions = QuestionAnsweringPrompt().questions
        self.top_k = top_k
        self.baseline = baseline
        if self.baseline and top_k > 0:
            raise ValueError("Baseline should not be used with top_k > 0")
        if top_k > 0:
            self.prompt_prefix = ("Given the following article, first answer the questions. Then, "
                                  "using the article and answers as key pointers, generate a summary of the article.\n")
        else:
            self.prompt_prefix = "Given the following article, generate a summary of the article.\n"

    def get_icl_examples(self, icl_examples, top_k_questions):
        prompt = ""
        for example in icl_examples:
            prompt += self.generate_prompt_text(example['input'], top_k_questions)
            prompt += ' '
            for i, q in enumerate(top_k_questions):
                prompt += f"A{i + 1}: {example[q]}. "
            reference = (ast.literal_eval(example['reference']))[0]
            prompt += f"\nSummary: {reference}\n\n"
        return prompt

    def generate_prompt_text(self, text, top_k_questions):
        prompt = self.prompt_prefix + 'Article: ' + text
        for i, q in enumerate(top_k_questions):
            prompt += f"\nQ{i + 1}: {self.questions[q]}"
        prompt += "\nA:" if not self.baseline else "\nSummary:"
        return prompt

    def get_prompt(self, text, task_file, domain):
        try:
            top_k_questions = self.top_k_questions.loc[domain, 'top_k_columns']
        except KeyError:
            domain = domain.split(' -> ')[0]
            top_k_questions = self.top_k_questions.loc[domain, 'top_k_columns']
        top_k_questions = top_k_questions[:self.top_k] if top_k_questions else []
        prompt_elements_for_task = self.prompt_elements[task_file]
        if not self.baseline:
            prompt = self.get_icl_examples(prompt_elements_for_task, top_k_questions)
        else:
            prompt = ""
        prompt += self.generate_prompt_text(text, top_k_questions)
        return prompt
