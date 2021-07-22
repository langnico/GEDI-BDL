#!/bin/bash

#BSUB -W 2:00
#BSUB -o /cluster/home/nlang/HCS_project/output/gedi_crossval.%J.%I.txt
#BSUB -e /cluster/home/nlang/HCS_project/output/gedi_crossval.%J.%I.txt
#BSUB -R "rusage[mem=6000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_model0==GeForceGTX1080Ti]"
#BSUB -n 1
#BSUB -J "GEDI_crossval[1-100]"
##BSUB -u nlang@ethz.ch

# load modules on cluster
module load python_gpu/3.7.1
module load hdf5/1.10.1


# job index (set this to your system job variable e.g. for parallel job arrays)
# used to set model_idx and test_fold_idx below.
#index=0   # index=0 --> model_idx=0, test_fold_idx=0
index=$((LSB_JOBINDEX - 1))

inputs_path=demo_data/GEDI_BDL_demo/GEDI_BDL_demo_subset_neon.npy

target_key='als_rh098'
min_gt=0
max_gt=100

input_key='rxwaveform'
sample_length=1420
noise_mean_key='noise_mean_corrected'

model_name='SimpleResNet_8blocks'
loss_key='gaussian_nll'
n_models=10
n_folds=10
normalize_targets=true

batch_size=16  # the batch size was reduced to obtain a stable optimization with the small demo dataset
nb_epoch=200
base_lr=0.0001

# data augmentation
shift_left=0.2
shift_right=0.2

# quality flags to filter different expected noise levels
setting_idx=3           # 0: power-night, 1: power-night + power-day, 2: power-night + power-day + coverage-night, 3: all
# filtering for complete crossover data including waveform matching information, otherwise all data is used
use_quality_flag=true
pearson_thresh=0.95

# select the model index for the model ensemble
model_idx=$(( $index % ${n_models} ))

# select the test fold index
test_fold_idx=$(( $index / ${n_models} ))

out_dir=output_demo/testfold_${test_fold_idx}/model_${model_idx}

echo job index: $index
echo model_idx: $model_idx
echo test_fold_idx: ${test_fold_idx}
echo output directory: ${out_dir}

# train and test
python3 torch_code/train.py --out_dir=${out_dir} \
                            --n_folds=${n_folds} \
                            --test_fold_idx=${test_fold_idx} \
                            --min_gt=${min_gt} \
                            --max_gt=${max_gt} \
                            --batch_size=${batch_size} \
                            --nb_epoch=${nb_epoch} \
                            --base_learning_rate=${base_lr} \
                            --loss_key=${loss_key} \
                            --sample_length=${sample_length} \
                            --inputs_path=${inputs_path} \
                            --input_key=${input_key} \
                            --target_key=${target_key} \
                            --shift_left=${shift_left} \
                            --shift_right=${shift_right} \
                            --model_name=${model_name}\
                            --setting_idx=${setting_idx} \
                            --normalize_targets=${normalize_targets} \
                            --pearson_thresh=${pearson_thresh} \
                            --noise_mean_key=${noise_mean_key}

