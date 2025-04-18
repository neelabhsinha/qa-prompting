import os

import pandas as pd
import sys
from tqdm import tqdm
import torch

from const import summary_outputs_dir, cache_dir
from src.handler.exit_handler import EXIT
from src.metrics.rouge import RougeScore
from src.prompts.summarization_prompt import SummarizationPrompt
from src.utils.model_pass import get_lm_response
from src.loader.super_natural_instructions_loader import SuperNaturalInstructionsLoader
from src.models.model import LanguageModel
from src.utils.results_io_util import write_results, get_source_dataset


def generate_summary(model_name, batch_size, instance_per_task=50000, top_k=5, baseline=False, checkpoint=None, global_top_k=False):
    parameters_dict = {'model_name': model_name, 'instance_per_task': instance_per_task, 'checkpoint': checkpoint,
                       'top_k': top_k}
    print('Parameters -')
    print(str(parameters_dict) + '\n\n')
    data_loader = SuperNaturalInstructionsLoader(split='train', categories=['Summarization'],
                                                 instance_per_task=instance_per_task,
                                                 batch_size=batch_size)
    model_builder = LanguageModel(f'{cache_dir}/{checkpoint}' if checkpoint is not None else model_name)
    model = model_builder.get_model()
    tokenizer = model_builder.get_tokenizer()
    if 'gpt' not in model_name and 'gemini' not in model_name and 'deepseek' not in model_name:
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
    if 'gpt' in model_name:
        model_name = 'openai/' + model_name
    elif 'gemini' in model_name:
        model_name = 'google/' + model_name
    name = checkpoint if checkpoint is not None else ('pretrained--' + model_name.replace('/', '--'))
    prompt_util = SummarizationPrompt(name, baseline, top_k, global_top_k)
    name += f'--{top_k}' + ('--baseline' if baseline else '')
    name += f'--global' if global_top_k else ''
    results_path = f'{summary_outputs_dir}/{name}'
    results_df = execute(data_loader, prompt_util, tokenizer, model, model_name,
                         'Super Natural Instructions', top_k, baseline)
    os.makedirs(results_path, exist_ok=True)
    write_results(results_df, results_path, parameters_dict)


def execute(data_loader, prompt_util, tokenizer, model, model_name, dataset_name, top_k, baseline):
    rouge_score = RougeScore()
    if 'gpt' not in model_name and 'gemini' not in model_name:
        model.generation_config.pad_token_ids = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    results = {'task_file': [], 'instance_number': [], 'input': [], 'reference': [], 'candidate': [],
               'rouge1': [], 'rouge2': [], 'rougeL': []}
    pbar = tqdm(data_loader, total=len(data_loader),
                desc=f'Generating Summary for {dataset_name} using {model_name}')
    count = 0
    for batch in pbar:
        try:
            texts = [instance['input'] for instance in batch]
            domains = [instance['domains'][0] for instance in batch]
            task_files = [instance['task_file'] for instance in batch]
            instance_numbers = [instance['instance_number'] for instance in batch]
            prompts = [prompt_util.get_prompt(text, task_file, domain) for text, task_file, domain
                       in zip(texts, task_files, domains)]
            candidate_batch_raw = get_lm_response(prompts, model, tokenizer, model_name, max_new_tokens=512 + 32 * top_k)
            candidate_batch = []
            for candidate in candidate_batch_raw:
                try:
                    if baseline:
                        cand = candidate.split('\n')[0]
                    elif "Summary:" in candidate and len(candidate.split('Summary: ')) > 0:
                        cand = candidate.split('Summary: ')[1].split('\n\n')[0]
                    else:
                        cand = ""
                except:
                    cand = ""
                candidate_batch.append(cand)
            reference_batch = [instance['output'][0] for instance in batch]
            rouge_scores = rouge_score.get_score(candidate_batch, reference_batch)
            for task_file, instance_number, candidate, ref, inp, rouge in zip(task_files, instance_numbers,
                                                                              candidate_batch,
                                                                              reference_batch, texts,
                                                                              rouge_scores):
                results['task_file'].append(task_file)
                results['instance_number'].append(instance_number)
                results['root_dataset'].append(get_source_dataset(task_file))
                results['input'].append(inp)
                results['reference'].append(ref)
                results['candidate'].append(candidate)
                results['rouge1'].append(rouge['rouge1'])
                results['rouge2'].append(rouge['rouge2'])
                results['rougeL'].append(rouge['rougeL'])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('\nCuda Out of Memory Error: Clearing Cache', file=sys.stderr)
        if EXIT.is_set():
            return
        count += 1
    results_df = pd.DataFrame(results)
    return results_df
