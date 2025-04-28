import os

import pandas as pd
import sys
from tqdm import tqdm
import torch

from const import qa_outputs_dir, cache_dir
from src.handler.exit_handler import EXIT
from src.utils.model_pass import get_lm_response
from src.utils.results_io_util import get_source_dataset
from src.loader.super_natural_instructions_loader import SuperNaturalInstructionsLoader
from src.metrics.qa_precision_score import QAPrecisionScore
from src.models.model import LanguageModel
from src.prompts.qa_prompts import QuestionAnsweringPrompt


def generate_answers(model_name, batch_size, instance_per_task=50000, checkpoint=None):
    parameters_dict = {'model_name': model_name, 'instance_per_task': instance_per_task, 'checkpoint': checkpoint}
    print('Parameters -')
    print(str(parameters_dict) + '\n\n')
    data_loader = SuperNaturalInstructionsLoader(split='train', categories=['Summarization'],
                                                 instance_per_task=instance_per_task,
                                                 batch_size=batch_size)
    prompt_util = QuestionAnsweringPrompt()
    model_builder = LanguageModel(f'{cache_dir}/{checkpoint}' if checkpoint is not None else model_name)
    model = model_builder.get_model()
    tokenizer = model_builder.get_tokenizer()
    if 'gpt' not in model_name and 'gemini' not in model_name:
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
    if 'gpt' in model_name:
        model_name = 'openai/' + model_name
    elif 'gemini' in model_name:
        model_name = 'google/' + model_name
    name = checkpoint if checkpoint is not None else ('pretrained--' + model_name.replace('/', '--'))
    results_path = f'{qa_outputs_dir}/{name}'
    results_df, metrics_df = execute(data_loader, prompt_util, tokenizer, model, model_name,
                                     'Super Natural Instructions')
    os.makedirs(results_path, exist_ok=True)
    results_df.to_csv(results_path + '/outputs.csv', index=False)
    metrics_df.to_csv(results_path + '/metrics.csv', index=False)


def execute(data_loader, prompt_util, tokenizer, model, model_name, dataset_name):
    qa_precision_score = QAPrecisionScore()
    if 'gpt' not in model_name and 'gemini' not in model_name:
        model.generation_config.pad_token_ids = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    results = {'task_file': [], 'instance_number': [], 'root_dataset': [], 'input': [], 'reference': []}
    metrics = {'task_file': [], 'instance_number': [], 'root_dataset': []}
    for question_id in prompt_util.questions:
        results[question_id] = []
        metrics[question_id] = []
    pbar = tqdm(data_loader, total=len(data_loader),
                desc=f'Generating Answers for {dataset_name} using {model_name}')
    count = 0
    for batch in pbar:
        try:
            texts = [instance['input'] for instance in batch]
            prompts = prompt_util.get_prompt(texts)
            candidate_batch = get_lm_response(prompts, model, tokenizer, model_name, max_new_tokens=64)
            if 'llama' in model_name.lower():
                candidate_batch = [candidate.split('\n')[0].strip() for candidate in candidate_batch]
            if 'mistral' in model_name.lower():
                candidate_batch = [candidate.split('Explanation')[0].strip() for candidate in candidate_batch]
            for i, instance in enumerate(batch):
                results['task_file'].append(instance['task_file'])
                results['instance_number'].append(instance['instance_number'])
                results['root_dataset'].append(get_source_dataset(instance['task_file']))
                metrics['task_file'].append(instance['task_file'])
                metrics['instance_number'].append(instance['instance_number'])
                metrics['root_dataset'].append(get_source_dataset(instance['task_file']))
                results['input'].append(instance['input'])
                results['reference'].append(instance['output'])
                for j, key in enumerate(prompt_util.questions):
                    results[key].append(candidate_batch[len(prompt_util.questions) * i + j])
                    precision = qa_precision_score.get_score(candidate_batch[len(prompt_util.questions) * i + j],
                                                             instance['output'][0])
                    metrics[key].append(precision)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('\nCuda Out of Memory Error: Clearing Cache', file=sys.stderr)
        if EXIT.is_set():
            return
        count += 1
    results_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(metrics)
    return results_df, metrics_df
