#!/bin/bash

# ----- CONFIGURATION -----

# path to base directory with model_XX subdirectories
ensemble_dir=demo_data/GEDI_BDL_demo/output_demo/testfold_0/

model_name='SimpleResNet_8blocks'
n_models=10
batch_size=8192  # Note: Reduce the batch size if the GPU is out of memory.

# set output directory for predictions
prediction_dir='output_demo_orbit_prediction'
# NOTE that in predict.py:
# if prediction_dir is None:
#     args.prediction_dir = os.path.dirname(args.file_path_L1B).replace('/L1B', '/pred_RH98')

# set L1B file
file_path_L1B=demo_data/GEDI_BDL_demo/DEMO_orbit_files/L1B/GEDI01_B_2019224233051_O03775_T03020_02_003_01.h5
# set corresponding L2A file (used for quality filtering)
file_path_L2A=demo_data/GEDI_BDL_demo/DEMO_orbit_files/L2A/processed_GEDI02_A_2019224233051_O03775_T03020_02_001_01.h5

echo L1B_path:
echo ${file_path_L1B}

echo L2A_path:
echo ${file_path_L2A}

echo output directory:
echo ${prediction_dir}

python3 torch_code/predict.py --ensemble_dir=${ensemble_dir} \
                              --n_models=${n_models} \
                              --batch_size=${batch_size} \
                              --file_path_L1B=${file_path_L1B} \
                              --file_path_L2A=${file_path_L2A} \
                              --prediction_dir=${prediction_dir} \
                              --model_name=${model_name}
