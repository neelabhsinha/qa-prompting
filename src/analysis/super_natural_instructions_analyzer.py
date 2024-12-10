from collections import Counter
import os

import pandas as pd
from src.loader.super_natural_instructions import SuperNaturalInstructions
from src.loader.super_natural_instructions_loader import filter_metadata
from const import dataset_analysis_dir


class SuperNaturalInstructionsAnalyzer:
    def __init__(self, split='test', instance_per_task=50):
        self.dataset = SuperNaturalInstructions()
        self.split_name = split
        self.split = self.dataset.get_split(split)
        self.dataset_metadata = filter_metadata(self.dataset.get_task_metadata(), None, None, None)
        self.instance_per_task = instance_per_task

    def _count_tasks(self):
        return len(self.split)

    def _count_instances(self):
        return self.dataset_metadata.loc[self.split, 'count_instances'].sum()

    def _count_instances_after_clipping(self):
        return self.dataset_metadata.loc[self.split, 'count_instances'].apply(
            lambda x: min(x, self.instance_per_task)).sum()

    def _get_min_instance_per_task(self):
        return min(self.dataset_metadata.loc[self.split, 'count_instances'])

    def _group_task_and_instance_count_by_aspect(self, aspect):
        df = self.dataset_metadata.loc[self.split, [aspect, 'count_instances']]
        df['capped_instances'] = df['count_instances'].apply(lambda x: min(x, self.instance_per_task))
        exploded_df = df.explode(aspect)
        summary_df = exploded_df.groupby(aspect).agg(number_of_tasks=pd.NamedAgg(column=aspect, aggfunc='size'),
                                                     total_instances=pd.NamedAgg(column='capped_instances',
                                                                                 aggfunc='sum')
                                                     ).reset_index()
        total_tasks = summary_df['number_of_tasks'].sum()
        total_instances = summary_df['total_instances'].sum()
        avg_tasks = total_tasks / len(summary_df)
        avg_instances = total_instances / len(summary_df)
        total_row = pd.DataFrame([['Total', total_tasks, total_instances]], columns=summary_df.columns)
        avg_row = pd.DataFrame([['Average', avg_tasks, avg_instances]], columns=summary_df.columns)
        summary_df = pd.concat([summary_df, total_row, avg_row], ignore_index=True).reset_index(drop=True)
        file_path = f'{dataset_analysis_dir}/{self.split_name}/category_counts_{aspect}.csv'
        summary_df.to_csv(file_path)

    def save_analysis_results(self):
        os.makedirs(f'{dataset_analysis_dir}/{self.split_name}', exist_ok=True)
        print('Saving analysis results...')
        total_tasks = self._count_tasks()
        total_instances = self._count_instances()
        min_instance_count = self._get_min_instance_per_task()
        total_instances_after_clipping = self._count_instances_after_clipping()
        avg_instance_per_task = total_instances / total_tasks
        avg_instance_per_task_after_clipping = total_instances_after_clipping / total_tasks
        with(open(f'{dataset_analysis_dir}/{self.split_name}/total_tasks_and_instances.txt', 'w')) as f:
            f.write(f'Total Tasks: {total_tasks}\n')
            f.write(f'Minimum Instance in a Task: {min_instance_count}\n')
            f.write(f'Total Instances: {total_instances}\n')
            f.write(f'Total Instances after clipping: {total_instances_after_clipping}\n')
            f.write(f'Average Instances per Task: {avg_instance_per_task}\n')
            f.write(f'Average Instances per Task after clipping: {avg_instance_per_task_after_clipping}\n')
        print('Grouping task and instance count by aspects...')
        self._group_task_and_instance_count_by_aspect('categories')
        self._group_task_and_instance_count_by_aspect('domains')
        self._group_task_and_instance_count_by_aspect('reasoning')
