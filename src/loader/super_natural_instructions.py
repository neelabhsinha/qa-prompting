import os
import pandas as pd
import json
from const import metadata_dir, source_dataset_dir
from collections import defaultdict
from src.handler.exit_handler import EXIT


def convert_to_list(string):
    result = string.split(', ')
    for i in range(len(result)):
        result[i] = result[i].split(' -> ')[0]
    return result


def list_to_string(lst):
    if isinstance(lst, list):
        return ','.join(map(str, lst))
    return lst


class SuperNaturalInstructions:
    def __init__(self):
        self._dataset_dir = source_dataset_dir
        self._dataset_metadata_filepath = os.path.join(metadata_dir, 'dataset_metadata_sni.csv')
        self._task_metadata = self._load_dataset_metadata()
        self._train_split = self._get_split('train')
        self._test_split = self._get_split('test')
        if len(self._test_split) == 0:
            split_fraction = int(len(self._train_split) * 0.2)
            self._test_split = self._train_split[:split_fraction]

    def _get_split(self, split):
        file_path = os.path.join(self._dataset_dir, 'splits', 'default', split + '_tasks.txt')
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() in self._task_metadata.index]
            f.close()
        return lines

    def _load_dataset_metadata(self):
        columns_to_convert = ['categories', 'domains', 'reasoning', 'input_language', 'output_language',
                              'instruction_language']
        converters = {col: convert_to_list for col in columns_to_convert}
        if not os.path.exists(self._dataset_metadata_filepath):
            print('No metadata found. Creating Config from source dataset.')
            metadata = []
            task_path = os.path.join(self._dataset_dir, 'tasks')
            task_files = [f for f in os.listdir(task_path)
                          if (os.path.isfile(os.path.join(task_path, f)) and f.endswith('.json'))]
            task_files.sort()
            for task_file in task_files:
                with open(os.path.join(task_path, task_file), 'r', encoding='utf-8') as f:
                    task = json.load(f)
                    task_metadata = {
                        'name': os.path.splitext(task_file)[0],
                        'categories': ', '.join(task['Categories']),
                        'domains': ', '.join(task['Domains']),
                        'reasoning': ', '.join(task['Reasoning']),
                        # 'split': 'train' if os.path.splitext(task_file)[0] in self._train_split else 'test',
                        'input_language': ', '.join(task['Input_language']),
                        'output_language': ', '.join(task['Output_language']),
                        'instruction_language': ', '.join(task['Instruction_language']),
                        'count_positive_examples': len(task['Positive Examples']),
                        'count_negative_examples': len(task['Negative Examples']),
                        'count_instances': len(task['Instances'])
                    }
                    f.close()
                    metadata.append(task_metadata)
                    if EXIT.is_set():
                        return
            result_df = pd.DataFrame(metadata).set_index('name')
            result_df.to_csv(self._dataset_metadata_filepath)
        result_df = pd.read_csv(self._dataset_metadata_filepath, converters=converters).set_index('name')
        return result_df

    def get_task_metadata(self):
        return self._task_metadata

    def get_split(self, split_name):
        if split_name == 'train':
            return self._train_split
        elif split_name == 'test':
            return self._test_split

    def get_data(self, batch_tuples):
        data_instances = []
        file_groups = defaultdict(list)
        for file_name, instance_number in batch_tuples:
            file_groups[file_name].append(instance_number)
        for file_name, instance_numbers in file_groups.items():
            file_path = os.path.join(self._dataset_dir, 'tasks', file_name + '.json')
            with open(file_path, 'r') as f:
                task = json.load(f)
                task['file_name'] = file_name
                num_tasks = self._task_metadata.loc[file_name, 'count_instances'].astype(int)
                for instance_number in instance_numbers:
                    if instance_number < num_tasks:
                        data_instances.append(self.reformat_data(task, instance_number))
                f.close()
        return data_instances

    @staticmethod
    def reformat_data(task, instance_number):
        input = task['Instances'][instance_number]['input']
        output = task['Instances'][instance_number]['output']
        definition = task['Definition'][0]
        positive_examples = task['Positive Examples']
        negative_examples = task['Negative Examples']
        input_language = task['Input_language']
        output_language = task['Output_language']
        instruction_language = task['Instruction_language']
        categories = task['Categories']
        domains = task['Domains']
        reasoning = task['Reasoning']
        data_instance = {
            'input': input,
            'output': output,
            'definition': definition,
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'input_language': input_language,
            'output_language': output_language,
            'instruction_language': instruction_language,
            'categories': categories,
            'domains': domains,
            'reasoning': reasoning,
            'task_file': task['file_name'],
            'instance_number': instance_number
        }
        return data_instance
