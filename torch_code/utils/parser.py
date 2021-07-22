import argparse
import numpy as np


def setup_parser():
    """
    Setup parser with default settings.
    Returns: argparse.ArgumentParser() object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default='./tmp/', help="output directory for the experiment")

    # dataset file
    parser.add_argument("--dataset", help="Dataset type.", default='CROSSOVER_GEDI', choices=['CROSSOVER_GEDI'])
    parser.add_argument("--inputs_path", help="path to h5 file with input and target arrays and additional attributes")
    parser.add_argument("--input_key", default='rxwaveform', help="input waveform", choices=['rxwaveform'])
    parser.add_argument("--target_key", default='als_rh098', help="target variable that is estimated.")

    # for simulated data (args.dataset == 'SIMULATED_GEDI') with numpy files e.g. input_key.npy target_key.npy
    parser.add_argument("--data_dir", help="path to directory with npy files")

    # dataset preprocessing and filtering
    parser.add_argument("--sample_length", default=1420, help="Waveform length. GEDI waveform max length is 1420.", type=int)
    parser.add_argument("--setting_idx", default=3, type=int, help="0: power-night, 1: power-night + power-day, 2: power-night + power-day + coverage-night, 3: all")
    parser.add_argument("--use_quality_flag", type=str2bool, nargs='?', const=True, default=True, help="True: only use samples with quality_flag==1")
    parser.add_argument("--min_gt", default=-np.inf, help="Filter target range: Keep samples >= min_gt", type=float)
    parser.add_argument("--max_gt", default=np.inf, help="Filter target range: Keep samples <= max_gt", type=float)
    parser.add_argument("--pearson_thresh", default=0.95, help="scalar (float) [0,1], quality criteria to filter data ", type=float)
    parser.add_argument("--noise_mean_key", default='noise_mean_corrected', help="noise mean key")
    parser.add_argument("--normalize_targets", type=str2bool, nargs='?', const=True, default=True, help="normalize target labels with mean and std")

    # model architecture
    parser.add_argument("--model_name", help="model names (functions) defined in models.py", default='SimpleResNet_8blocks')
    parser.add_argument("--num_outputs", default=2, help="Number of outputs. Set to 2 for regressing mean and variance.")

    # training params
    parser.add_argument("--skip_training", type=str2bool, nargs='?', const=True, default=False, help="do not optimize parameters (i.e. run test only)")
    parser.add_argument("--num_workers", default=8, help="Number of workers for pytorch Dataloader")
    parser.add_argument("--loss_key", default='gaussian_nll', help="Loss keys", choices=['MSE', 'MAE', 'gaussian_nll', 'laplacian_nll'])
    parser.add_argument("--batch_size", default=64, help="batch size at train/val time. (number of samples per iteration)", type=int)
    parser.add_argument("--nb_epoch", default=200, help="number of epochs to train", type=int)
    parser.add_argument("--base_learning_rate", default=0.0001, help="initial learning rate", type=float)
    parser.add_argument("--l2_lambda", default=0.0, help="L2 regularizer on weights hyperparameter", type=float)
    parser.add_argument("--optimizer", default='ADAM', help="optimizer name", choices=['ADAM', 'SGD'])
    parser.add_argument("--momentum", default=0.0, help="momentum for SGD ", type=float)

    # data augmentation for training
    parser.add_argument("--shift_left", default=None, help="Augmentation: scalar (float) [0,1], relative shift w.r.t waveform length", type=str2none)
    parser.add_argument("--shift_right", default=None, help="Augmentation: scalar (float) [0,1], relative shift w.r.t waveform length", type=str2none)

    # additional augmentation at train time for robustness tests. NOTE: currently not implemented see todos run.py Trainer class
    parser.add_argument("--label_noise", default=0.0, help="scalar (float) [0,1], relative label noise ", type=float)
    parser.add_argument("--label_noise_distribution", default='uniform', help="label noise distribution", choices=['uniform', 'normal'])

    # data splits and generalization experiment settings
    parser.add_argument("--data_split", default='randCV', help="attribute_name used to split data into train/test", choices=['randCV', 'attrCV'])
    parser.add_argument("--n_folds", default=10, help="Number of folds. ", type=int)
    parser.add_argument("--test_fold_idx", default=0, help="Fold split index that is used for testing. ", type=int)
    parser.add_argument("--range_to_remove", default=None, nargs='+', type=float, help="if data_split=='randCV': removes a target range (min max) to evaluate the epistemic uncertainty")
    parser.add_argument("--ood_attribute", default=None, help="if data_split=='randCV': attribute_name used to remove OOD data e.g. 'continental_region_1km' ")
    parser.add_argument("--ood_value", default=None, help="if data_split=='randCV': attribute value float defined as OOD (will be only in test data)", type=float)
    parser.add_argument("--ood_value_string", default=None, help="if data_split=='randCV': attribute value string defined as OOD (will be only in test data)")
    parser.add_argument("--split_attribute", default=None, help="if data_split=='attrCV': attribute_name used to split data into train/test")
    parser.add_argument("--test_attribute_value", default=None, nargs='+', type=float, help="if data_split=='attrCV': attribute value to hold-out for testing")
    parser.add_argument("--model_weights_path", help="Pre-trained model weights path (e.g. weights_best.h5) ")

    # test params
    parser.add_argument("--model_dir", help="Model directory with weights_best.h5, train_mean.npy, train_std.npy.")

    # prediction params
    parser.add_argument("--ensemble_dir", help="path to directory with subdirectories called model_i")
    parser.add_argument("--file_path_L1B", help="path to L1B h5 file for prediction.")
    parser.add_argument("--file_path_L2A", help="path to L2A h5 file for prediction.", type=str_or_none)
    parser.add_argument("--n_models", default=10, help="Number of models in the ensemble. ", type=int)
    parser.add_argument("--prediction_dir", default=None, help="output directory for the predictions")

    return parser


# --- Helper functions to parse arguments ---

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return float(v)


def str_or_none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return str(v)

class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(StoreAsArray, self).__call__(parser, namespace, values, option_string)
