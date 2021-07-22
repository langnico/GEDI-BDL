"""
Collect ensemble predictions and results from all cross-validation folds.

This script creates:

    i) a new subdirectory in each testfold directory e.g. "/experiment_dir/testfold_XX/ensemble",
    containing the ensemble results for the particular test fold.

    ii) a new subdirectory in the experiment base directory e.g. "/experiment_dir/ensemble_collected",
    containing the collected ensemble predictions and results from all disjoint test folds.

"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt


from utils.error_metrics import get_metrics_dict
from utils.plots import plot_hist2d, plot_calibration, plot_precision_recall


def collect_CV_ensemble_predictions(experiment_dir_base, collected_dir, n_folds, n_models,
                                    metrics_dict_fun, model_indices_to_exclude=None, filter_negative_preds=True):

    # Add testfold subdir template
    experiment_dir_base = os.path.join(experiment_dir_base, 'testfold_{}')

    # ## Collect results from cross-validation
    results = {'pred_ensemble': [],
               'targets': [],
               'epistemic_std': [],
               'aleatoric_std': [],
               'predictive_std': [],
               'test_indices': []}

    error_metrics_folds = {}
    for metric in metrics_dict_fun.keys():
        error_metrics_folds[metric] = []

    for fold_i in np.arange(n_folds):
        print('*****************************************************')
        print('fold: ', fold_i)
        experiment_dir = experiment_dir_base.format(fold_i)

        out_dir = os.path.join(experiment_dir, 'ensemble')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #  init lists
        if model_indices_to_exclude is None:
            model_indices_to_exclude = []

        pred_means, pred_var = [], []

        for i in range(n_models):
            if i in model_indices_to_exclude:
                continue
            model_i_dir = os.path.join(experiment_dir, 'model_{}'.format(i))
            pred_means.append(np.load(os.path.join(model_i_dir, 'predictions.npy')))
            pred_var.append(np.load(os.path.join(model_i_dir, 'variances.npy')))

            if i == 0:
                targets = np.load(os.path.join(model_i_dir, 'targets.npy'))
                test_indices = np.load(os.path.join(model_i_dir, 'test_indices.npy'))
                print('mean_target_train: ', np.load(os.path.join(model_i_dir, 'mean_target_train.npy')))
                print('std_target_train: ', np.load(os.path.join(model_i_dir, 'std_target_train.npy')))

        # convert the ensemble list to numpy
        pred_means = np.array(pred_means)
        pred_var = np.array(pred_var)

        # final predictions (average over model ensemble)
        pred_ensemble = np.mean(pred_means, axis=0)

        print(pred_means.shape)
        print(pred_var.shape)
        print(targets.shape)
        print(pred_ensemble.shape)

        print('pred_var: min: {}, max: {}'.format(np.min(pred_var), np.max(pred_var)))
        print('pred_ensemble: min: {}, max: {}'.format(np.min(pred_ensemble), np.max(pred_ensemble)))

        # save as npy
        np.save(os.path.join(out_dir, 'pred_means.npy'), pred_means)
        np.save(os.path.join(out_dir, 'pred_var.npy'), pred_var)
        np.save(os.path.join(out_dir, 'targets.npy'), targets)
        np.save(os.path.join(out_dir, 'pred_ensemble.npy'), pred_ensemble)

        # compute performance of ensemble
        if filter_negative_preds:
            # remove samples with negative pred_ensemble
            valid_indices = pred_ensemble >= 0
        else:
            # all samples are valid
            valid_indices = pred_ensemble == pred_ensemble

        x = targets[valid_indices]  # ground truth
        y = pred_ensemble[valid_indices]  # prediction

        for metric in metrics_dict_fun.keys():
            print('{}: {:.1f}'.format(metric, metrics_dict_fun[metric](x, y)))

        error_metrics = {}
        for metric in metrics_dict_fun.keys():
            error_metrics[metric] = metrics_dict_fun[metric](x, y)
            print(metric, error_metrics[metric])

        with open(os.path.join(out_dir, 'error_metrics.json'), 'w') as f:
            json.dump(error_metrics, f)

        # collect error metrics for all folds
        for metric in error_metrics.keys():
            error_metrics_folds[metric].append(error_metrics[metric])

        # compute epistemic and aleatoric over model ensemble
        epistemic_var = np.var(pred_means, axis=0)
        aleatoric_var = np.mean(pred_var, axis=0)
        predictive_var = epistemic_var + aleatoric_var

        aleatoric_std = np.sqrt(aleatoric_var)
        epistemic_std = np.sqrt(epistemic_var)
        predictive_std = np.sqrt(predictive_var)

        print('epistemic_std.shape', epistemic_std.shape)
        print('aleatoric_std.shape', aleatoric_std.shape)
        print('predictive_std.shape', predictive_std.shape)

        # save as npy
        np.save(os.path.join(out_dir, 'epistemic_std.npy'), epistemic_std)
        np.save(os.path.join(out_dir, 'aleatoric_std.npy'), aleatoric_std)
        np.save(os.path.join(out_dir, 'predictive_std.npy'), predictive_std)

        results['pred_ensemble'].append(pred_ensemble)
        results['targets'].append(targets)
        results['epistemic_std'].append(epistemic_std)
        results['aleatoric_std'].append(aleatoric_std)
        results['predictive_std'].append(predictive_std)
        results['test_indices'].append(test_indices)

    #  concatenate to numpy array
    for key in results:
        results[key] = np.concatenate(results[key])

    # errors folds to numpy
    for key in error_metrics_folds:
        error_metrics_folds[key] = np.array(error_metrics_folds[key])

    for key in results:
        print(results[key].shape, results[key].dtype)

    for key in error_metrics_folds:
        print(key)
        print(error_metrics_folds[key])

    # Save the collected folds (JOIN THE FOLDS)
    for key in results:
        np.save(os.path.join(collected_dir, '{}.npy'.format(key)), results[key])

    np.save(os.path.join(collected_dir, 'error_metrics_folds.npy'), error_metrics_folds)

    # compute mean and std of error metrics over all 10 folds
    error_metrics = {}
    for key in error_metrics_folds:
        error_metrics[key + '_mean'] = np.mean(error_metrics_folds[key])
        error_metrics[key + '_std'] = np.std(error_metrics_folds[key])
    print(error_metrics)

    with open(os.path.join(collected_dir, 'error_metrics.json'), 'w') as f:
        json.dump(error_metrics, f)

    return results


if __name__ == "__main__":

    # Set path to experiment base directory containing all test fold subdirectories
    experiment_dir_base = sys.argv[1]
    # experiment_dir_base = "demo_data/GEDI_BDL_demo/output_demo/"
    
    filter_negative_preds=True  # remove negative height predictions for evaluation (negative predictions are still included in the collected data)

    n_folds = 10
    n_models = 10

    # make new subdirectory for collected predictions
    collected_dir = os.path.join(experiment_dir_base, 'ensemble_collected')
    if not os.path.exists(collected_dir):
        os.makedirs(collected_dir)

    print('collected_dir:')
    print(collected_dir)

    # get all metrics functions
    metrics_dict_fun = get_metrics_dict()

    # collect results
    results = collect_CV_ensemble_predictions(experiment_dir_base=experiment_dir_base,
                                              collected_dir=collected_dir,
                                              n_folds=n_folds,
                                              n_models=n_models,
                                              metrics_dict_fun=metrics_dict_fun,
                                              filter_negative_preds=filter_negative_preds)

    # ------ PLOT RESULTS ------

    # -- plot confusion ground truth vs. prediction --
    ma = 100
    plot_hist2d(x=results['targets'], y=results['pred_ensemble'], ma=ma, step=ma/10,
                out_dir=collected_dir, figsize=(8, 6), xlabel='Ground truth [m]', ylabel='Prediction [m]')

    # -- plot precision recall curve --
    # (Note that here we do not use the adaptive thresholding described in the paper)
    metric = 'RMSE'
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    plot_precision_recall(predictions=results['pred_ensemble'], targets=results['targets'],
                          uncertainties=results['predictive_std'], metric=metric,
                          ax=ax, label=None, ylabel='RMSE [m]', style='k-')
    plt.grid()
    plt.legend(loc='upper left')
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(fname=os.path.join(collected_dir, 'PR_curves_RMSE_predictive_std.png'), dpi=300)

    # -- plot calibration --
    step = 1
    bins = np.arange(0, 16, step)
    xticks = np.arange(0, 16, 2)
    min_bin_count = 200
    fig, ax = plot_calibration(predictions=results['pred_ensemble'], targets=results['targets'],
                               uncertainties=results['predictive_std'], metric='RMSE', bins=bins, step=step,
                               min_bin_count=min_bin_count, xlabel='Predictive STD [m]', ylabel='Empirical RMSE [m]')
    ax.set_xticks(xticks)
    plt.tight_layout()
    fig.savefig(fname=os.path.join(collected_dir, 'calibration.png'), dpi=300)

    print('Collected results and plots are saved in:')
    print(collected_dir)

