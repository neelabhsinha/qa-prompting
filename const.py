project_root = '.'  # project root directory
project_name = 'qa-prompting'  # project name

# Dataset location (change the directory path here to the dataset)
source_dataset_dir = 'ADD PATH TO YOUR DATASET HERE'  # source dataset directory

# Directories (edit the source_dataset_dir as needed)
metadata_dir = f'{project_root}/metadata'  # metadata directory
cache_dir = f'{project_root}/cache'  # cache directory
results_dir = f'{project_root}/results'  # results directory
qa_outputs_dir = f'{project_root}/qa_outputs'  # QA outputs directory
prompt_elements_dir = f'{project_root}/prompt_elements'  # prompt elements directory
summary_outputs_dir = f'{project_root}/summary_outputs'  # summary outputs directory
aggregated_results_dir = f'{project_root}/aggregated_results'
dataset_analysis_dir = f'{project_root}/dataset_analysis'

tasks = ['qa_generate', 'analyze_dataset', 'analyze_qa_relevance', 'summary_generate', 'analyze_summarization',
         'compute_metrics']  # list of available tasks
