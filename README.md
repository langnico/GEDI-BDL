# Global canopy height estimation with GEDI LIDAR waveforms and Bayesian deep learning

This repository provides the code used to create the results presented in [Global canopy height estimation with GEDI LIDAR waveforms and Bayesian deep learning](https://arxiv.org/abs/2103.03975).

## Installation
The code has been tested with `Python 3.8.5`.

### Setup a virtual environment 
See details in the [venv documentation](https://docs.python.org/3/library/venv.html).
 
**Example on linux:**
 
Create a new virtual environment called `GEDI_BDL_env`.
```
python3 -m venv /path/to/new/virtual/environment/GEDI_BDL_env
```

Activate the new environment:
```
source /path/to/new/virtual/environment/GEDI_BDL_env/bin/activate
```

### Install python packages
After activating the venv, install the python packages with:
```
pip install -r requirements.txt
```

## Download data for the DEMO scripts
Please download the zip file `GEDI_BDL_demo.zip` from [here](https://share.phys.ethz.ch/~pf/nlangdata/GEDI_BDL_demo.zip).

Extract and save it in this repository such that path reads like this: `GEDI-BDL/demo_data/GEDI_BDL_demo/`.
The demo scripts will refer to this relative path.

This demo dataset contains a subset of the ALS crossover training data used in the paper. It consists of 6,868 samples and is based on the publicly available ALS data from the [National Ecological Observatory Network (NEON)](https://data.neonscience.org/data-products/explore) in the United States.

Note: The purpose of this demo dataset is to setup the code. Models trained on this subset may *not* generalize as described in the paper.
More information on the demo dataset in the readme file: `demo_data/GEDI_BDL_demo/README.txt`.

## Running the code

### Train and test a single CNN (or an ensemble)
This example runs the regression of RH98 (proxy for the canopy top height) from the input L1B waveform. It runs the first model of the first random cross-validation fold. 

Running this script multiple times with job indices from 0-9 will train and test a full ensemble of 10 models for the first cross-validation fold. Job indices 10-19 will run the ensemble for the second fold and so on.
```
bash DEMO_GEDI_regression_crossval_ensemble.sh
```
Alternative run a parallel job array on an IBM LSF batch system:
```
bsub < cluster/job_array_regression_crossval_ensemble.sh
```

Launch tensorboard to look at the training and validation loss curves:

```tensorboard --logdir output_demo --port 7777```

#### Collect ensemble predictions from all cross-validation folds
Here we run it for the ensemble demo output that was already included in the .zip file. 

```
python torch_code/collect_ensemble_preds.py demo_data/GEDI_BDL_demo/output_demo/
```

### Predict for all (quality) waveforms in an L1B orbit file 
This example demonstrates how a trained model can be deployed to a full orbit file of the GEDI Version 1 data. This script loads the ensemble trained on the demo dataset from here: `demo_data/GEDI_BDL_demo/output_demo/testfold_0/`.

The orbit files are loaded from `demo_data/GEDI_BDL_demo/DEMO_orbit_files`. The quality flag from the corresponding L2A file is used to filter the predictions.
```
bash DEMO_GEDI_orbit_prediction.sh
```


## Citation

If you use this code please cite our paper:

*Lang, N., Kalischek, N., Armston, J., Schindler, K., Dubayah, R., & Wegner, J. D. (2021). Global canopy height estimation with GEDI LIDAR waveforms and Bayesian deep learning. arXiv preprint [arXiv:2103.03975]((https://arxiv.org/abs/2103.03975)).*

BibTex:
```
@article{lang2021global,
  title={Global canopy height estimation with GEDI LIDAR waveforms and Bayesian deep learning},
  author={Lang, Nico and Kalischek, Nikolai and Armston, John and Schindler, Konrad and Dubayah, Ralph and Wegner, Jan Dirk},
  journal={arXiv preprint arXiv:2103.03975},
  year={2021}
}
```