import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import icgen
from src.available_datasets import all_datasets
#all_datasets = ['caltech101']
all_datasets = ['oxford_iiit_pet', 'deep_weeds', 'visual_domain_decathlon/dtd']
import os
import argparse
import random
import json
import numpy as np

config_parser = parser = argparse.ArgumentParser(description='Dataset Creation', add_help=False)
parser.add_argument('-n', '--n_augmentations', default=15, type=int, help='Number of augmentations to generate.')
parser.add_argument('-o', '--output_dir', default='datasets', type=str, help='Path where the datasets will be exported to.')
parser.add_argument('-s', '--skip_augmentation', default=0, type=int, help='Specifies which of the augmentation dir is already processed and can be skipped')

args = parser.parse_args()


# for d in all_new_tfds:
#     if d not in tfds.list_builders():
#         print(f"dataset {d} not supported.")

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    

for i in range(1, args.n_augmentations+1):
    random.seed(i)
    np.random.seed(i)
    dir_path = f"{args.output_dir}/{i}"
    
    if i <= args.skip_augmentation:
        print(f"skipping augmentation dir: {dir_path}")
        continue
        
    os.makedirs(dir_path, exist_ok=True)
    
    for dataset in all_datasets:
    
        print(f"processing {dir_path}/{dataset}")
        
        dataset_generator = icgen.ICDatasetGenerator(
                data_path=dir_path,
                min_resolution=16,
                max_resolution=512,
                max_log_res_deviation=1,  # Sample only 1 log resolution from the native one
                min_classes=2,
                max_classes=100,
                min_examples_per_class=20,
                max_examples_per_class=100_000,
            )
    
        (dev_data, test_data, dataset_info), identifier = dataset_generator.get_dataset(
                    dataset=dataset, augment=True, download=True
                    )
    
        # print(identifier['class_to_dev_samples'])
        # print(identifier['class_to_test_samples'])
    
        identifier["seed"] = i
        identifier["number_classes"] = len(identifier["classes"])
        identifier["number_train_samples_per_class"] = len(list(identifier['class_to_dev_samples'].values())[0])
        identifier["number_test_samples_per_class"] = len(list(identifier['class_to_test_samples'].values())[0])
        
        print(f"number of classes sampled: {identifier['number_classes']}")
        print(f"number of train samples per class sampled: {identifier['number_train_samples_per_class']}")
        print(f"number of test samples per class sampled: {identifier['number_test_samples_per_class']}")
    
        identifier_path = f"{dir_path}/{dataset}/icgen_info_{dataset.replace('/','_')}.json"
    
        with open(identifier_path, 'w', encoding='utf8') as identifier_file:
            json.dump(identifier, identifier_file, cls=SetEncoder)
            print(f"dumped {identifier_path}")