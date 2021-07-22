import os
import numpy as np
import h5py
import time
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from utils.parser import setup_parser
from dataset import GediDataOrbitMem
from tqdm import tqdm

from utils.transforms import Normalize, ToTensor, denormalize
from models.models import Models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(model, dl_pred, mean_target_train, std_target_train):
    """
    Predict with trained model.
    Args:
        model: torch model (trained weights must be loaded)
        dl_pred: torch dataloader with (unlabeled) data to predict.
        mean_target_train: train mean of target variable (to denormalize predictions)
        std_target_train: train standard deviation (std) of target variable (to denormalize predictions)

    Returns: Dict with torch tensors 'predictions', 'variances' in original scale (denormalized).

    """

    # init validation results for current epoch
    out_dict = {'predictions': [], 'log_variances': []}

    with torch.no_grad():
        for step, (inputs) in enumerate(tqdm(dl_pred, ncols=100, desc='pred')):

            inputs = inputs.to(device)

            predictions = model.forward(inputs).squeeze(dim=-1)
            predictions, log_variances = predictions[:, 0], predictions[:, 1]

            out_dict['predictions'] += list(predictions)
            out_dict['log_variances'] += list(log_variances)

        for key in out_dict.keys():
            if out_dict[key]:
                out_dict[key] = torch.stack(out_dict[key], dim=0)
                print("out_dict['{}'].shape: ".format(key), out_dict[key].shape)

    # compute variance from log_variance
    out_dict['variances'] = torch.exp(out_dict['log_variances'])
    del out_dict['log_variances']

    # convert torch tensor to numpy
    for key in out_dict.keys():
        out_dict[key] = out_dict[key].data.cpu().numpy()

    # denormalize model outputs
    out_dict['predictions'] = denormalize(out_dict['predictions'], mean_target_train, std_target_train)
    out_dict['variances'] = out_dict['variances'] * std_target_train ** 2

    return out_dict


if __name__ == "__main__":

    # set parameters
    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    if args.file_path_L2A is not None:
        if not os.path.exists(args.file_path_L2A):
            raise ValueError('L2A file does not exists: {}'.format(args.file_path_L2A))
    
    start_time_loading = time.time()

    # load input mean and std
    mean_input_train = np.load(os.path.join(args.ensemble_dir, 'model_0', 'mean_input_train.npy'))
    std_input_train = np.load(os.path.join(args.ensemble_dir, 'model_0', 'std_input_train.npy'))
    
    # load target mean and std (for denormalizing the predictions)
    mean_target_train = np.load(os.path.join(args.ensemble_dir, 'model_0', 'mean_target_train.npy'))
    std_target_train = np.load(os.path.join(args.ensemble_dir, 'model_0', 'std_target_train.npy'))

    # setupt preprocessing input_transforms
    input_transforms = Compose([Normalize(mean=mean_input_train, std=std_input_train), ToTensor()])

    # create dataset
    ds_orbit = GediDataOrbitMem(args.file_path_L1B, args.file_path_L2A, sample_length=1420,
                                input_transforms=input_transforms, noise_mean_key=args.noise_mean_key)

    print('orbit data loaded.')
    end_time_loading = time.time()
    duration_loading = end_time_loading - start_time_loading 
    
    start_time_predicting = time.time()

    # create dataloader
    dl_pred = DataLoader(ds_orbit, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model architecture
    # Setup model
    model_chooser = Models()
    model = model_chooser(args.model_name)(num_outputs=args.num_outputs)
    model.to(device)

    # set model to eval model
    model.eval()

    # initialize predictions
    pred_means = []
    pred_vars = []

    # Loop through all models
    for i in range(args.n_models):
        print('predicting with model_{}'.format(i))
        # get model_dir_i
        model_dir_i = os.path.join(args.ensemble_dir, 'model_{}'.format(i))
        
        # load weights
        model_weights_path = os.path.join(model_dir_i, 'best_weights.pt')
        model.load_state_dict(torch.load(model_weights_path))

        # predict
        out_dict_i = predict(model, dl_pred, mean_target_train, std_target_train)

        pred_means.append(out_dict_i['predictions'])
        pred_vars.append(out_dict_i['variances'])

    # convert to np array
    pred_means = np.array(pred_means, dtype=np.float32)
    pred_vars = np.array(pred_vars, dtype=np.float32)
    print('pred_means.shape', pred_means.shape)

    out_dict = {}

    # compute mean across ensemble mean predictions
    out_dict['pred_ensemble'] = np.mean(pred_means, axis=0)

    # compute variances
    epistemic_var = np.var(pred_means, axis=0)
    aleatoric_var = np.mean(pred_vars, axis=0)
    predictive_var = epistemic_var + aleatoric_var

    out_dict['predictive_std'] = np.sqrt(predictive_var)
    out_dict['aleatoric_std'] = np.sqrt(aleatoric_var)
    out_dict['epistemic_std']= np.sqrt(epistemic_var)

    end_time_predicting = time.time()
    duration_predicting = end_time_predicting - start_time_predicting

    # copy gedi keys to out_dict
    if ds_orbit.data_L2A:
        gedi_keys = ['shot_number', 'lat_lowestmode', 'lon_lowestmode', 'modis_nonvegetated']
        for key in gedi_keys:
            out_dict[key] = ds_orbit.data_L2A[key]
    else:
        out_dict['shot_number'] = ds_orbit.data_L1B['shot_number']

    # output directory to save predictions
    if args.prediction_dir is None:
        args.prediction_dir = os.path.dirname(args.file_path_L1B).replace('/L1B', '/pred_RH98')

    if not os.path.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)

    # save as h5 file
    out_path = os.path.join(args.prediction_dir, os.path.basename(args.file_path_L1B).replace('GEDI01_B', 'GEDI02_RH98'))
    print('writing to hdf5 file:')
    print(out_path)
    with h5py.File(out_path, 'w') as f:
        for key in out_dict.keys():
            print(key, out_dict[key].shape)
            f.create_dataset(key, data=out_dict[key])
    
    print('time loading data: ', time.strftime("%Hh%Mm%Ss", time.gmtime(duration_loading)) )
    print('time predicting:   ', time.strftime("%Hh%Mm%Ss", time.gmtime(duration_predicting)) )
    print('DONE!')

