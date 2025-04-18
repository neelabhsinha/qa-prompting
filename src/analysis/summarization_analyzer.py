import os
import re

import pandas as pd
from matplotlib import pyplot as plt

from const import summary_outputs_dir, aggregated_results_dir
from src.loader.super_natural_instructions import SuperNaturalInstructions


def camel_to_hyphen(camel_str):
    hyphen_str = re.sub(r'([a-z])([A-Z])', r'\1-\2', camel_str).lower()
    return hyphen_str


class SummarizationAnalyzer:
    def __init__(self, metric):
        self.results_metadata = self.get_result_file_metadata()
        self.metric = metric
        self._super_natural_instructions = SuperNaturalInstructions()
        self._metadata = self._super_natural_instructions.get_task_metadata()

    @staticmethod
    def get_result_file_metadata():
        dirs = os.listdir(summary_outputs_dir)
        model_metadata = {}
        for dir in dirs:
            details = dir.split('--')
            model_name = details[2]
            top_k = details[3]
            baseline = True if len(details) > 4 and 'baseline' in details[4] else False
            global_top_k = True if len(details) > 4 and 'global' in details[4] else False
            if model_name not in model_metadata:
                model_metadata[model_name] = []
            model_metadata[model_name].append({'top_k': top_k, 'baseline': baseline, 'global': global_top_k,
                                               'results_dir': dir})
        for model_name, settings in model_metadata.items():
            settings = [setting for setting in settings if not setting['baseline'] and not setting['global']]
            settings = sorted(settings, key=lambda x: int(x['top_k']))
            model_metadata[model_name] = settings
        return model_metadata

    def tabulate_all_results(self):
        results_summary = []
        top_k_summary = []
        for model_name, settings in self.results_metadata.items():
            for setting in settings:
                results_dir = setting['results_dir']
                results_file = os.path.join(summary_outputs_dir, results_dir, 'result_statistics.csv')
                df = pd.read_csv(results_file, index_col=0)
                mean_values = df.loc['mean', :]
                top_k = setting['top_k']
                details = {'model_name': model_name, 'top_k': top_k}
                # all columns except instance_number
                cols = [col for col in df.columns if col != 'instance_number']
                for metric in cols:
                    details[metric + '_mean'] = mean_values[metric]
                results_summary.append(details)
        results_df = pd.DataFrame(results_summary)
        for model_name in results_df['model_name'].unique():
            model_results = results_df[results_df['model_name'] == model_name]
            best_result = model_results.loc[model_results['rouge1_mean'].idxmax()]
            data = {
                'model_name': model_name,
                'best_top_k': best_result['top_k'],
            }
            for metric in results_df.columns:
                if metric.endswith('_mean'):
                    data[metric] = best_result[metric]
            top_k_summary.append(data)
        top_k_df = pd.DataFrame(top_k_summary)
        path = os.path.join(aggregated_results_dir, 'results_summary.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        top_k_df.to_csv(path, index=False)

    def model_wise_analysis(self):
        for model_name, settings in self.results_metadata.items():
            results_summary_df = None
            for setting in settings:
                results_dir = setting['results_dir']
                results_file = os.path.join(summary_outputs_dir, results_dir, 'predictions.csv')
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    df['domain'] = df['task_file'].map(self._metadata['domains'])
                    df = df.explode('domain')
                    grouped_df = df.groupby('domain')[self.metric].mean().reset_index()
                    count_df = df.groupby('domain').size().reset_index()
                    grouped_df.columns = ['domain', setting['top_k']]
                    if results_summary_df is None:
                        results_summary_df = grouped_df
                    else:
                        results_summary_df = results_summary_df.merge(grouped_df, on='domain', how='outer')
            path = os.path.join(aggregated_results_dir, self.metric, model_name, 'k_variation.csv')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            results_summary_df.to_csv(path, index=False)
            self.generate_line_graph(results_summary_df, path.replace('.csv', '.png'))

    def generate_line_graph(self, df, path):
        plt.figure(figsize=(14, 8), dpi=300)
        font_size = 15
        plt.rc('font', size=font_size)
        for _, row in df.iterrows():
            domain = row['domain']
            x_values = df.columns[1:]
            y_values = row[x_values].values
            plt.plot(x_values, y_values, label=domain, marker='o')
        plt.xlabel('k')
        plt.ylabel(camel_to_hyphen(self.metric))
        plt.legend(
            title="Domain",
            loc='lower center',  # Position the legend at the bottom
            bbox_to_anchor=(0.5, -0.2),  # Center the legend below the plot
            ncol=len(df['domain'].unique()),  # Display legend entries in one row
        )
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.savefig(path.replace('.png', '.pdf'), format='pdf')
        plt.close()

    def save_results(self):
        self.tabulate_all_results()
        self.model_wise_analysis()
