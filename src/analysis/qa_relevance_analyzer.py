import json
import os.path

import pandas as pd
from matplotlib import pyplot as plt

from src.loader.super_natural_instructions import SuperNaturalInstructions
from const import qa_outputs_dir, prompt_elements_dir


class QARelevanceAnalyzer:
    def __init__(self):
        self._super_natural_instructions = SuperNaturalInstructions()
        self._metadata = self._super_natural_instructions.get_task_metadata()

    @staticmethod
    def save_top_k_examples(results_path, k=3):
        complete_file_path = os.path.join(qa_outputs_dir, results_path, 'outputs.csv')
        outputs_df = pd.read_csv(complete_file_path)
        metrics_df = pd.read_csv(os.path.join(qa_outputs_dir, results_path, 'metrics.csv'))
        exclude_columns = ['task_file', 'instance_number', 'input', 'reference']
        columns_to_check = [col for col in outputs_df.columns if col not in exclude_columns]
        total_score = metrics_df[columns_to_check].sum(axis=1)
        metrics_df['total_score'] = total_score
        # get top_k examples for each unique task_file
        top_k_examples = {}
        for task_file in outputs_df['task_file'].unique():
            task_df = outputs_df[outputs_df['task_file'] == task_file]
            task_metrics_df = metrics_df[metrics_df['task_file'] == task_file]
            task_metrics_df = task_metrics_df.sort_values(by='total_score', ascending=False)
            top_k_instances = task_metrics_df['instance_number'].head(k).values
            top_k_df = task_df[task_df['instance_number'].isin(top_k_instances)]
            top_k_df = top_k_df.to_dict(orient='records')
            top_k_examples[task_file] = top_k_df
        output_file_path = os.path.join(prompt_elements_dir, results_path, 'prompt_elements.json')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(top_k_examples, f, indent=4)

    def analyze_model(self, results_path):
        complete_file_path = os.path.join(qa_outputs_dir, results_path, 'metrics.csv')
        metrics_df = pd.read_csv(complete_file_path)
        exclude_columns = ['task_file', 'instance_number']
        columns_to_check = [col for col in metrics_df.columns if col not in exclude_columns]
        metrics_df = metrics_df.loc[~(metrics_df[columns_to_check] == 0.0).all(axis=1)]
        metrics_df['domains'] = metrics_df['task_file'].map(self._metadata['domains'])
        metrics_df = metrics_df.explode('domains')
        grouped_stats_df = metrics_df.groupby(['domains'])[columns_to_check].mean()
        return grouped_stats_df

    @staticmethod
    def get_top_k_columns(grouped_stats_df, k=10):
        # get top_k columns for each row
        result = {'domain': [], 'top_k_columns': []}
        for index, row in grouped_stats_df.iterrows():
            top_k_columns = row.nlargest(k).index
            result['domain'].append(index)
            result['top_k_columns'].append(', '.join(top_k_columns))
        top_k_df = pd.DataFrame(result)
        return top_k_df

    @staticmethod
    def save_results(results_df, top_k_df, results_path):
        results_df.to_csv(os.path.join(qa_outputs_dir, results_path, 'grouped_stats.csv'), index=True)
        top_k_df.to_csv(os.path.join(qa_outputs_dir, results_path, 'top_k_columns.csv'), index=False)
        print(f'Results saved to {results_path}')

    @staticmethod
    def plot_dataframe_bars(df, results_path, output_prefix='graph_output'):
        plt.figure(figsize=(14, 8), dpi=300)
        x = range(len(df.columns))
        width = 0.15
        font_size = 14
        plt.rc('font', size=font_size)
        for i, (index, row) in enumerate(df.iterrows()):
            plt.bar([pos + width * i for pos in x], row, width=width, label=index)
        plt.xticks([pos + width * (len(df) / 2 - 0.5) for pos in x], df.columns, rotation=45, ha='right')
        plt.ylabel("Precision Score")
        plt.title("Precision Score by Domain")
        plt.legend(title="Domains")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(qa_outputs_dir, results_path, f'{output_prefix}.png'), format='png')
        plt.savefig(os.path.join(qa_outputs_dir, results_path, f'{output_prefix}.pdf'), format='pdf')
        plt.close()

    def collect_all_results(self):
        for results_path in os.listdir(qa_outputs_dir):
            if os.path.isdir(os.path.join(qa_outputs_dir, results_path)):
                grouped_stats_df = self.analyze_model(results_path)
                top_k_df = self.get_top_k_columns(grouped_stats_df)
                self.save_results(grouped_stats_df, top_k_df, results_path)
                self.save_top_k_examples(results_path)
                self.plot_dataframe_bars(grouped_stats_df, results_path, f'grouped_stats_graph')
