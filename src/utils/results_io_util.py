import os
import re


def write_results(results_df, dir_path, parameters_dict=None):
    if results_df is None or dir_path is None:
        return
    description = results_df.describe()
    os.makedirs(dir_path, exist_ok=True)
    if parameters_dict is not None:
        with open(f'{dir_path}/parameters.txt', 'w') as f:
            for key, value in parameters_dict.items():
                f.write(f'{key}: {value}\n')
    description.to_csv(f'{dir_path}/result_statistics.csv')
    results_df.to_csv(f'{dir_path}/predictions.csv', index=False)
    print(f'Results saved to {dir_path}')


def get_source_dataset(task_name):
    base_name = os.path.splitext(task_name)[0]
    cleaned = re.sub(r"^task\d{4}_", "", base_name)
    cleaned = re.sub(r"_?summarization$", "", cleaned)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned
