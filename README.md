# QuickTune
## Download meta-data

Download QuickTune meta-dataset from [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/NxeXnnfGeGzFq9f). Move the content to a folder called *agg_curves* inside the *meta_data* folder.

```
cd meta_data 
mkdir agg_curves
cd agg_curves
wget https://rewind.tf.uni-freiburg.de/index.php/s/NxeXnnfGeGzFq9f
unzip QT_metadataset.zip
```



## Prepare environment
Create environment and install requirements:

`
conda -n quick_tune python=3.9
`

Install dependencies for running *QuickTune*:

`
pip install -r requirements.txt
`

## Fine-tune Network
You can fine-tune network by providing any hyperparameter as follows:

`
python finetune.py --model dla46x_c --pct_to_freeze 0.8
`


## Run Quick-Tune on Meta-dataset

```
cd hpo/optimizers/quick_tune --experiment_id qt_test
                             --aft_set micro 
                             --hidden_dim 32 
                             --output_dim 32 
                             --learning_rate 0.0001 
                             --meta_learning_rate 0.0001 
                             --meta_train 1 
                             --use_encoders_for_model 1 
                             --cost_aware 1 
                             --budget_limit 3600
```



