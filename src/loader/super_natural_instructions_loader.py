import random

from src.loader.super_natural_instructions import SuperNaturalInstructions


def filter_metadata(filtered_df, categories, domains, reasonings):
    # Filter by categories if specified
    if categories:
        filtered_df = filtered_df[
            filtered_df['categories'].apply(lambda x: any(item in x for item in categories))
        ]
    # Filter by domains if specified
    if domains:
        filtered_df = filtered_df[
            filtered_df['domains'].apply(lambda x: any(item in x for item in domains))
        ]
    # Filter by reasonings if specified
    if reasonings:
        filtered_df = filtered_df[
            filtered_df['reasoning'].apply(lambda x: any(item in x for item in reasonings))
        ]
    # Filter by input_language containing 'English'
    filtered_df = filtered_df[
        filtered_df['input_language'].apply(lambda x: 'English' in x)
    ]
    # Filter by output_language containing 'English'
    filtered_df = filtered_df[
        filtered_df['output_language'].apply(lambda x: 'English' in x)
    ]
    # Filter by instruction_language containing 'English'
    filtered_df = filtered_df[
        filtered_df['instruction_language'].apply(lambda x: 'English' in x)
    ]

    return filtered_df


class SuperNaturalInstructionsLoader:
    def __init__(self, split, categories=None, domains=None, reasonings=None, instance_per_task=None, batch_size=4,
                 shuffle=False):
        self._categories = categories
        self._domains = domains
        self._reasonings = reasonings
        self._instance_per_task = instance_per_task
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._current_batch_index = 0
        self.dataset = SuperNaturalInstructions()
        self._split = self.dataset.get_split(split)
        self._metadata = filter_metadata(self.dataset.get_task_metadata(), categories, domains, reasonings)
        selected_splits = self._metadata.index.tolist()
        self._split = [task for task in self._split if task in selected_splits]
        self._sampled_indices = self._get_file_and_instance_number()
        print('Loader initialization complete. Length = ', str(len(self._sampled_indices)))

    def _get_file_and_instance_number(self):
        all_indices = []
        for file in self._split:
            num_instances = self._metadata.loc[file, 'count_instances'].astype(int)
            num_instances = min(num_instances, self._instance_per_task) if self._instance_per_task else num_instances
            for instance_number in range(num_instances):
                all_indices.append((file, instance_number))
        if self._shuffle:
            random.shuffle(all_indices)
        else:
            all_indices = sorted(all_indices, key=lambda x: x[0])
        sampled_indices = [all_indices[i:i + self._batch_size] for i in range(0, len(all_indices), self._batch_size)]
        return sampled_indices

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_batch_index >= len(self._sampled_indices):
            raise StopIteration
        batch = self._sampled_indices[self._current_batch_index]
        self._current_batch_index += 1
        return self.dataset.get_data(batch)

    def __len__(self):
        return len(self._sampled_indices)
