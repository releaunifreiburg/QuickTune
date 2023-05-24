import json
import numpy as np
import os

experiment_args_batch = "batch24"
meta_search_space0 = {"hidden_dim": [8, 16, 32, 64],
                     "output_dim": [8, 16, 32, 64],
                     "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                     "meta_learning_rate": [0.0001, 0.001, 0.01, 0.1],
                     "acqf_fc": ["ei", "ucb"],
                     "explore_factor": [0.0, 0.01, 0.1],
                     "freeze_feature_extractor": [1, 0],
                     "with_scheduler": [1, 0],
                     "include_metafeatures": [1, 0],
                     "meta_train" : [1, 0],
                     "output_dim_metafeatures": [0, 4, 8, 16],
                     "load_only_dataset_descriptors": [1, 0],
                     "use_encoders_for_model": [1, 0],
                     "cost_aware": [1, 0]
                     }

meta_search_space1 = {"hidden_dim": [32],
                     "output_dim": [32],
                     "learning_rate": [0.0001],
                     "meta_learning_rate": [0.0001],
                     "acqf_fc": ["ei"],
                     "explore_factor": [0.0],
                     "freeze_feature_extractor": [0],
                     "with_scheduler": [1],
                     "include_metafeatures": [1],
                     "meta_train" : [1,0],
                     "output_dim_metafeatures": [4],
                     "load_only_dataset_descriptors": [1],
                     "use_encoders_for_model": [1],
                     "cost_aware": [1],
                     "target_model" : ['beit_large_patch16_512',
                                        'volo_d5_512',
                                        'volo_d5_448',
                                        'volo_d4_448',
                                        'swinv2_base_window12to24_192to384_22kft1k',
                                        'beit_base_patch16_384',
                                        'volo_d3_448',
                                        'tf_efficientnet_b7_ns',
                                        'convnext_small_384_in22ft1k',
                                        'tf_efficientnet_b6_ns',
                                        'volo_d1_384',
                                        'xcit_small_12_p8_384_dist',
                                        'deit3_small_patch16_384_in21ft1k',
                                        'tf_efficientnet_b4_ns',
                                        'xcit_tiny_24_p8_384_dist',
                                        'xcit_tiny_12_p8_384_dist',
                                        'edgenext_small',
                                        'xcit_nano_12_p8_384_dist',
                                        'mobilevitv2_075',
                                        'edgenext_x_small',
                                        'mobilevit_xs',
                                        'edgenext_xx_small',
                                        'mobilevit_xxs',
                                        'dla46x_c'],
                     "test_generalization_to_model": [1],
                     "observe_cost": [1],
                     "budget_limit": [14400],
                     "aft_set": ["mini"],
                     "use_only_target_model": [1]
                     }

meta_search_space = {"hidden_dim": [32],
                     "output_dim": [32],
                     "learning_rate": [0.0001],
                     "meta_learning_rate": [0.0001],
                     "acqf_fc": ["ei"],
                     "explore_factor": [0.0],
                     "freeze_feature_extractor": [0],
                     "with_scheduler": [1],
                     "include_metafeatures": [1],
                     "meta_train" : [1],
                     "output_dim_metafeatures": [4],
                     "load_only_dataset_descriptors": [1],
                     "use_encoders_for_model": [1],
                     "cost_aware": [1],
                     "dataset_id_in_split" : [0,1,2,3,4,5],
                     "split_id": [0, 1, 2, 3, 4],
                     #"aft_set": ["micro", "mini", "extended"],
                     "aft_set": [ "extended"],
                     "conditioned_time_limit": [1]
                     }

meta_search_space2 = {"hidden_dim": [32],
                     "output_dim": [32],
                     "learning_rate": [0.0001],
                     "meta_learning_rate": [0.0001],
                     "acqf_fc": ["ei"],
                     "explore_factor": [0.0],
                     "freeze_feature_extractor": [0],
                     "with_scheduler": [1],
                     "include_metafeatures": [1],
                     "meta_train" : [1],
                     "output_dim_metafeatures": [4],
                     "load_only_dataset_descriptors": [1],
                     "use_encoders_for_model": [1],
                     "cost_aware": [1],
                     "subsample_models_in_hub" : [5, 10, 15, 20],
                     "aft_set": ["micro", "mini", "extended"],
                     }

meta_search_space4 = {"hidden_dim": [32],
                     "output_dim": [32],
                     "learning_rate": [0.0001],
                     "meta_learning_rate": [0.0001],
                     "acqf_fc": ["ei"],
                     "explore_factor": [0.0],
                     "freeze_feature_extractor": [0],
                     "with_scheduler": [1],
                     "include_metafeatures": [1],
                     "meta_train" : [1],
                     "output_dim_metafeatures": [4],
                     "load_only_dataset_descriptors": [1],
                     "use_encoders_for_model": [1],
                     "cost_aware": [1],
                     "use_only_target_model": [1],
                     "target_model": ['beit_large_patch16_512', 'xcit_small_12_p8_384_dist', 'dla46x_c'],
                     "aft_set": ["micro", "mini", "extended"],
                     }


#create text file with randoly sampled args for each experiment
budget_limit = 200
args_list = []
random = False

if random:
    for i in range(100):
        experiment = f"dyhpo{i+1100}"
        args = f"--experiment_id {experiment}"


        for key, values in meta_search_space.items():
            value = np.random.choice(values)
            args += f" --{key} " + str(value)

        args_list.append(args)
else:
    #cartesian product of lists
    from itertools import product
    for i, values in enumerate(product(*meta_search_space.values())):
        values = list(values)
        experiment = f"dyhpo{i+2400}"
        args = f"--experiment_id {experiment}"

        id1 = list(meta_search_space.keys()).index("meta_train")
        id2 = list(meta_search_space.keys()).index("cost_aware")

        if values[id1] ==0:
            values[id2] = 0

        for key, value in zip(meta_search_space.keys(), values):
            args += f" --{key} " + str(value)

        args_list.append(args)

#get rootdir path
rootdir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(rootdir, "..", "experiments", "output", "hpo", "experiments_args", experiment_args_batch + ".args")

with open(output_dir, "w") as f:
    for args in args_list:
        f.write(args)
        f.write("\n")
