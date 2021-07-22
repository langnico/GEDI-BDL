import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from dataset import CrossOverDataMem, CustomSubset
from utils.transforms import Normalize, RandomShift, ToTensor, denormalize, RandomLabelNoise
from utils.loss import RMSELoss, MELoss, GaussianNLL, LaplacianNLL
from pathlib import Path
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model, log_dir, args):
        """
        Initialize a Trainer object to train and test the model.
        Args:
            model: pytorch model
            log_dir: path to directory to save tensorboard logs
            args: argparse object (see setup_parser() in utils.parser)
        """
        self.model = model
        self.args = args
        self.out_dir = args.out_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.do_shift_target = self.args.input_key != 'delta_Z0_ground'
        self.shift_interval = (-self.args.shift_left, self.args.shift_right) \
            if self.args.shift_left and self.args.shift_right else None
        self.train_indices, self.val_indices, self.test_indices = None, None, None

        self.ds_train, self.ds_val, self.ds_test, \
        self.mean_input_train, self.std_input_train, \
        self.mean_target_train, self.std_target_train= self._setup_dataset()

        self.optimizer = self._setup_optimizer()
        self.error_metrics = self._setup_metrics()

        print('self.mean_target_train', self.mean_target_train)
        print('self.std_target_train', self.std_target_train)

    def _setup_metrics(self):
        error_metrics = {'MSE': torch.nn.MSELoss(),
                         'RMSE': RMSELoss(),
                         'MAE': torch.nn.L1Loss(),
                         'ME': MELoss()}

        if self.args.num_outputs == 2:
            error_metrics['gaussian_nll'] = GaussianNLL()
            error_metrics['laplacian_nll'] = LaplacianNLL()

        print('error_metrics.keys():', error_metrics.keys())
        error_metrics['loss'] = error_metrics[self.args.loss_key]
        return error_metrics

    def _setup_optimizer(self):
        if self.args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_learning_rate,
                                         weight_decay=self.args.l2_lambda)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.base_learning_rate,
                                        weight_decay=self.args.l2_lambda)
        else:
            raise ValueError("Solver '{}' is not defined.".format(self.args.optimizer))
        return optimizer

    def _setup_transforms(self, mean_input_train, std_input_train, mean_target_train, std_target_train):
        input_transforms = Compose([Normalize(mean=mean_input_train, std=std_input_train), ToTensor()])

        if self.args.normalize_targets:
            target_transforms = Compose([Normalize(mean=mean_target_train, std=std_target_train)])
        else:
            target_transforms = None

        # data augmentation for training
        augmentation_transforms = Compose([RandomShift(shift_interval=self.shift_interval,
                                                       do_shift_target=self.do_shift_target)])

        # TODO Option to add random label noise as an augmentation to carry out robustness experiments
        #                              RandomLabelNoise(rel_label_noise=self.args.label_noise,
        #                                      distribution=self.args.label_noise_distribution)])

        return input_transforms, target_transforms, augmentation_transforms

    def _setup_dataset(self):
        if self.args.dataset == 'CROSSOVER_GEDI':
            dataset = CrossOverDataMem(input_path=self.args.inputs_path,
                                       target_key=self.args.target_key,
                                       input_key=self.args.input_key,
                                       min_gt=self.args.min_gt,
                                       max_gt=self.args.max_gt,
                                       sample_length=self.args.sample_length,
                                       settings_index=self.args.setting_idx,
                                       pearson_thresh=self.args.pearson_thresh,
                                       split_attribute_name=self.args.ood_attribute,
                                       noise_mean_key=self.args.noise_mean_key)

        else:
            raise ValueError('Dataset {} is undefined.'.format(self.args.dataset))

        # print target range:
        print('overall target min: {}, max: {}'.format(np.min(dataset.targets), np.max(dataset.targets)))

        # Split dataset into three subsets

        if self.args.data_split == 'randCV':
            # Random n-fold cross validation split
            self.train_indices, self.val_indices, self.test_indices = self._split_dataset_random_nfold_CV(len_dataset=len(dataset),
                                                                                                          n_folds=self.args.n_folds,
                                                                                                          test_fold_idx=self.args.test_fold_idx,
                                                                                                          quality_indices_train=dataset.quality_indices_train,
                                                                                                          quality_indices_valtest=dataset.quality_indices_valtest)

            if self.args.range_to_remove is not None:
                # Remove specific target range from training and validation indices.
                # We keep that range in the test data to evaluate the epistemic uncertainty
                print('TRAIN & VAL data is filtered in the range: ', self.args.range_to_remove)
                self.train_indices, out_dist_indices_train = self.filter_subset_indices_by_attribute_range(self.train_indices,
                                                                                dataset_attribute=dataset.split_attribute,
                                                                                range_to_remove=self.args.range_to_remove)
                self.val_indices, out_dist_indices_val = self.filter_subset_indices_by_attribute_range(self.val_indices,
                                                                              dataset_attribute=dataset.split_attribute,
                                                                              range_to_remove=self.args.range_to_remove)

                # append OOD indices to test_indices
                print('num samples OOD: ', len(out_dist_indices_train) + len(out_dist_indices_val))
                print('test_indices.shape before', self.test_indices.shape)
                self.test_indices = np.concatenate((self.test_indices, out_dist_indices_train, out_dist_indices_val))
                print('test_indices.shape with ood', self.test_indices.shape)
                
                # Check test indices
                test_indices_in_dist, test_indices_out_dist = self.filter_subset_indices_by_attribute_range(self.test_indices,
                                                                              dataset_attribute=dataset.split_attribute,
                                                                              range_to_remove=self.args.range_to_remove)
                print('test_indices_in_dist.shape', test_indices_in_dist.shape)
                print('test_indices_out_dist.shape', test_indices_out_dist.shape)

            elif self.args.ood_attribute is not None:
                # Remove a specific attribute value from train and val data.
                # We keep that attribute in the test data to evaluate the epistemic uncertainty for in- and out of distribution.

                if self.args.ood_value_string is not None:
                    self.args.ood_value = self.args.ood_value_string

                print('TRAIN & VAL data is filtered by attribute: {} and value: {}'.format(self.args.ood_attribute, self.args.ood_value))
                self.train_indices, out_dist_indices_train = self.filter_subset_indices_by_attribute(self.train_indices,
                                                                                                     dataset_attribute=dataset.split_attribute,
                                                                                                     out_dist_value=self.args.ood_value)
                self.val_indices, out_dist_indices_val = self.filter_subset_indices_by_attribute(self.val_indices,
                                                                                                   dataset_attribute=dataset.split_attribute,
                                                                                                   out_dist_value=self.args.ood_value)

                # append OOD indices to test_indices
                print('num samples OOD: ', len(out_dist_indices_train) + len(out_dist_indices_val))
                print('test_indices.shape before', self.test_indices.shape)
                self.test_indices = np.concatenate((self.test_indices, out_dist_indices_train, out_dist_indices_val))
                print('test_indices.shape with ood', self.test_indices.shape)

                 # Check test indices
                test_indices_in_dist, test_indices_out_dist = self.filter_subset_indices_by_attribute(self.test_indices,
                                                                              dataset_attribute=dataset.split_attribute,
                                                                              out_dist_value=self.args.ood_value)
                print('test_indices_in_dist.shape', test_indices_in_dist.shape)
                print('test_indices_out_dist.shape', test_indices_out_dist.shape)

        elif self.args.data_split == 'attrCV':
            # Split by attribute: Hold-out a specific attribute value
            self.train_indices, self.val_indices, self.test_indices = self._split_dataset_by_attribute(attribute=dataset.split_attribute,
                                                                                                       test_attribute=self.args.test_attribute_value,
                                                                                                       quality_indices_train=dataset.quality_indices_train,
                                                                                                       quality_indices_valtest=dataset.quality_indices_valtest)
        else:
            raise ValueError("self.args.data_split = '{}' is not defined".format(self.args.data_split))

        # check that data splits do not overlap
        if not set(self.train_indices).isdisjoint(set(self.val_indices)):
            raise ValueError("TRAIN indices overlap with VAL indices.")
        if not set(self.train_indices).isdisjoint(set(self.test_indices)):
            raise ValueError("TRAIN indices overlap with TEST indices.")
        if not set(self.test_indices).isdisjoint(set(self.val_indices)):
            raise ValueError("TEST indices overlap with VAL indices.")

        # Calculate mean and std of training set and setup transforms.
        mean_input_train = np.mean(dataset.inputs[self.train_indices])
        std_input_train = np.std(dataset.inputs[self.train_indices])

        mean_target_train = np.mean(dataset.targets[self.train_indices])
        std_target_train = np.std(dataset.targets[self.train_indices])

        # save training mean and std
        np.save(Path(self.args.out_dir) / 'mean_input_train.npy', mean_input_train)
        np.save(Path(self.args.out_dir) / 'std_input_train.npy', std_input_train)

        np.save(Path(self.args.out_dir) / 'mean_target_train.npy', mean_target_train)
        np.save(Path(self.args.out_dir) / 'std_target_train.npy', std_target_train)

        input_transforms, target_transforms, augmentation_transforms = self._setup_transforms(mean_input_train=mean_input_train,
                                                                                              std_input_train=std_input_train,
                                                                                              mean_target_train=mean_target_train,
                                                                                              std_target_train=std_target_train)

        ds_train = CustomSubset(dataset, self.train_indices,
                                input_transforms=input_transforms,
                                target_transforms=target_transforms,
                                augmentation_transforms=augmentation_transforms)
        ds_val = CustomSubset(dataset, self.val_indices,
                              input_transforms=input_transforms,
                              target_transforms=target_transforms)
        ds_test = CustomSubset(dataset, self.test_indices,
                               input_transforms=input_transforms,
                               target_transforms=target_transforms)
        return ds_train, ds_val, ds_test, mean_input_train, std_input_train, mean_target_train, std_target_train

    def train(self):
        """
        A routine to train and validated the model for several epochs.
        """
        # Initialize train and validation loader
        dl_train = DataLoader(self.ds_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        dl_val = DataLoader(self.ds_val, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        # Init best losses for weights saving.
        loss_val_best = np.inf
        best_epoch = None

        if self.args.model_weights_path is not None:
            # load best model weights
            print('ATTENTION: loading pretrained model weights from:')
            print(self.args.model_weights_path)
            self.model.load_state_dict(torch.load(self.args.model_weights_path))

        # Start training
        for epoch in range(self.args.nb_epoch):
            epoch += 1
            print('Epoch: {} / {} '.format(epoch, self.args.nb_epoch))

            # optimize parameters
            training_metrics = self.optimize_epoch(dl_train)
            # validated performance
            val_dict, val_metrics = self.validate(dl_val)

            # -------- LOG TRAINING METRICS --------
            metric_string = 'TRAIN: '
            for metric in self.error_metrics.keys():
                # tensorboard logs
                self.writer.add_scalar('{}/train'.format(metric), training_metrics[metric], epoch)
                metric_string += ' {}: {:.3f},'.format(metric, training_metrics[metric])
            print(metric_string)

            # -------- LOG VALIDATION METRICS --------
            metric_string = 'VAL:   '
            for metric in self.error_metrics:
                # tensorboard logs
                self.writer.add_scalar('{}/val'.format(metric), val_metrics[metric], epoch)
                metric_string += ' {}: {:.3f},'.format(metric, val_metrics[metric])
            print(metric_string)

            # logging the estimated variance
            if 'log_variances' in val_dict:
                val_dict['variances'] = torch.exp(val_dict['log_variances'])

                if self.args.normalize_targets:
                    # denormalize the variance
                    val_dict['variances'] = val_dict['variances'] * self.std_target_train**2

                self.writer.add_scalar('var_mean/val', torch.mean(val_dict['variances']), epoch)
                self.writer.add_scalar('std_mean/val', torch.mean(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_min/val', torch.min(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_max/val', torch.max(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('var_count_infinite_elements/val', self.count_infinite_elements(val_dict['variances']), epoch)

                print('VAL: Number of infinite elements in variances: ', self.count_infinite_elements(val_dict['variances']))

            if val_metrics['loss'] < loss_val_best:
                loss_val_best = val_metrics['loss']
                best_epoch = epoch
                # save and overwrite the best model weights:
                path = Path(self.out_dir) / 'best_weights.pt'
                torch.save(self.model.state_dict(), path)
                print('Saved weights at {}'.format(path))

            # stop training if loss is nan
            if np.isnan(training_metrics['loss']) or np.isnan(val_metrics['loss']):
                raise ValueError("Training loss is nan. Stop training.")

        # TODO: Currently we save only the best and last epoch weights --> maybe want to save every nth epoch.
        print('Best val loss: {} at epoch: {}'.format(loss_val_best, best_epoch))
        # save model weights after last epoch:
        path = Path(self.out_dir) / 'weights_last_epoch.pt'
        torch.save(self.model.state_dict(), path)
        print('Saved weights at {}'.format(path))

    def optimize_epoch(self, dl_train):
        """
        Run the optimization for one epoch.

        Args:
            dl_train: torch dataloader with training data.

        Returns: Dict with error metrics on training data (including the loss). Used for tensorboard logs.
        """
        # init running error
        training_metrics = {}
        for metric in self.error_metrics:
            training_metrics[metric] = 0

        total_count_infinite_var = 0

        # set model to training mode
        self.model.train()
        for step, (inputs, labels) in enumerate(tqdm(dl_train, ncols=100, desc='train')):
            inputs, labels = inputs.to(device), labels.to(device)
            # Run forward pass
            predictions = self.model.forward(inputs).squeeze(dim=-1)

            if self.args.num_outputs == 2:
                predictions, log_variances = predictions[:, 0], predictions[:, 1]
                # pass predicted mean and log_variance to e.g. gaussian_nll
                loss = self.error_metrics['loss'](predictions, log_variances, labels)

                # debug
                variances = torch.exp(log_variances)
                count_infinite = self.count_infinite_elements(variances)
                total_count_infinite_var += count_infinite

            else:
                loss = self.error_metrics['loss'](predictions, labels)

            # Run backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute metrics on every batch and add to running sum
            for metric in self.error_metrics:
                if self.args.num_outputs == 2 and metric in ['gaussian_nll', 'laplacian_nll', 'loss']:
                    training_metrics[metric] += self.error_metrics[metric](predictions, log_variances, labels).item()
                else:
                    if self.args.normalize_targets:
                        # denormalize labels and predictions
                        predictions_ = denormalize(predictions, self.mean_target_train, self.std_target_train)
                        labels_ = denormalize(labels, self.mean_target_train, self.std_target_train)
                        training_metrics[metric] += self.error_metrics[metric](predictions_, labels_).item()
                    else:
                        training_metrics[metric] += self.error_metrics[metric](predictions, labels).item()

        # debug
        if total_count_infinite_var > 0:
            print('TRAIN DEBUG: ATTENTION: count infinite elements in variances is: {}'.format(total_count_infinite_var))

        # average over number of batches
        for metric in self.error_metrics.keys():
            training_metrics[metric] /= len(dl_train)
        return training_metrics

    def validate(self, dl_val):
        """
        Validate the model on validation data.

        Args:
            dl_val: torch dataloader with validation data

        Returns:
            val_dict: Dict with torch tensors for 'predictions', 'targets', 'log_variances'.
            val_metrics: Dict with error metrics on validation data (including the loss). Used for tensorboard logs.
        """
        # set model to eval model
        self.model.eval()

        # init validation results for current epoch
        val_dict = {'predictions': [], 'targets': []}

        if self.args.num_outputs == 2:
            val_dict['log_variances'] = []

        with torch.no_grad():
            for step, (inputs, labels) in enumerate(dl_val):  # for each training step

                inputs = inputs.to(device)
                labels = labels.to(device)

                predictions = self.model.forward(inputs).squeeze(dim=-1)
                if self.args.num_outputs == 2:
                    predictions, log_variances = predictions[:, 0], predictions[:, 1]
                    val_dict['log_variances'] += list(log_variances)

                val_dict['predictions'] += list(predictions)
                val_dict['targets'] += list(labels)

            for key in val_dict.keys():
                if val_dict[key]:
                    val_dict[key] = torch.stack(val_dict[key], dim=0)
                    print("val_dict['{}'].shape: ".format(key), val_dict[key].shape)

        val_metrics = {}

        for metric in self.error_metrics:
            if self.args.num_outputs == 2 and metric in ['gaussian_nll', 'laplacian_nll', 'loss']:
                val_metrics[metric] = self.error_metrics[metric](val_dict['predictions'],
                                                                 val_dict['log_variances'],
                                                                 val_dict['targets']).item()
            else:
                # denormalize labels and predictions
                if self.args.normalize_targets:
                    predictions_ = denormalize(val_dict['predictions'], self.mean_target_train, self.std_target_train)
                    targets_ = denormalize(val_dict['targets'], self.mean_target_train, self.std_target_train)
                    val_metrics[metric] = self.error_metrics[metric](predictions_, targets_).item()
                else:
                    val_metrics[metric] = self.error_metrics[metric](val_dict['predictions'],
                                                                     val_dict['targets']).item()
        return val_dict, val_metrics

    def test(self, model_weights_path=None, dl_test=None):
        """
        Test trained model on test data.

        Args:
            model_weights_path: path to trained model weights. Default: "best_weights.pt"
            dl_test: torch dataloader with test data. Default: self.ds_test is loaded.

        Returns:
            test_metrics: Dict with error metrics on test data (including the loss). Used for tensorboard logs.
            test_dict: Dict with torch tensors for 'predictions', 'targets', 'variances'.
            metric_string: formatted string to print test metrics.
        """
        if dl_test is None:
            dl_test = DataLoader(self.ds_test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        # test performance

        if model_weights_path is None:
            model_weights_path = Path(self.out_dir) / 'best_weights.pt'

        # load best model weights
        self.model.load_state_dict(torch.load(model_weights_path))

        test_dict, test_metrics = self.validate(dl_test)

        # convert log(var) to var
        if self.args.num_outputs == 2:
            test_dict['variances'] = torch.exp(test_dict['log_variances'])
            del test_dict['log_variances']

        # denormalize predictions and targets
        if self.args.normalize_targets:
            test_dict['predictions'] = denormalize(test_dict['predictions'], self.mean_target_train, self.std_target_train)
            test_dict['targets'] = denormalize(test_dict['targets'], self.mean_target_train, self.std_target_train)
            if self.args.num_outputs == 2:
                # denormalize the variances by multiplying with the target variance
                test_dict['variances'] = test_dict['variances'] * self.std_target_train**2

        if self.args.num_outputs == 2:
            print('TEST: Number infinite elements in variances: ', self.count_infinite_elements(test_dict['variances']))

        # convert torch tensor to numpy
        for key in test_dict.keys():
            test_dict[key] = test_dict[key].data.cpu().numpy()

        metric_string = 'TEST:   '
        for metric in self.error_metrics:
            metric_string += ' {}: {:.3f},'.format(metric, test_metrics[metric])
        print(metric_string)
        return test_metrics, test_dict, metric_string

    def count_infinite_elements(self, x):
        return torch.sum(torch.logical_not(torch.isfinite(x))).item()

    @staticmethod
    def _split_dataset_random_nfold_CV(len_dataset, n_folds=10, test_fold_idx=0, quality_indices_train=None, quality_indices_valtest=None):
        """
        Split data into n folds for cross-validation.
        """
        # shuffle the samples randomly to create train, val and test sets
        indices = np.arange(len_dataset)
        # always generate the same random numbers with random seed for testing the code
        np.random.seed(2401)
        np.random.shuffle(indices)

        # split indices int n folds:
        indices_list = np.array_split(indices, n_folds)

        test_indices = indices_list[test_fold_idx]

        # all except indices from fold test fold  i
        trainval_indices = indices_list[:test_fold_idx] + indices_list[test_fold_idx + 1:]
        trainval_indices = np.concatenate(trainval_indices)

        # split training into 90% train and 10% val
        train_indices = trainval_indices[:int(0.9 * len(trainval_indices))]
        val_indices = trainval_indices[int(0.9 * len(trainval_indices)):]

        def filter_subset_indices(subset_indices, quality_indices):
            return subset_indices[quality_indices[subset_indices]]

        if quality_indices_train is not None:
            train_indices = filter_subset_indices(train_indices, quality_indices_train)
            val_indices = filter_subset_indices(val_indices, quality_indices_valtest)
            test_indices = filter_subset_indices(test_indices, quality_indices_valtest)

        return train_indices, val_indices, test_indices


    @staticmethod
    def _split_dataset_by_attribute(attribute, test_attribute, quality_indices_train=None, quality_indices_valtest=None):
        """
        Split data using all samples with a specific attribute as test data.
        """

        test_indices = np.argwhere(attribute == test_attribute)
        trainval_indices = np.argwhere(attribute != test_attribute)

        # split training into 80% train and 20% val
        train_indices = trainval_indices[:int(0.9 * len(trainval_indices))]
        val_indices = trainval_indices[int(0.9 * len(trainval_indices)):]

        def filter_subset_indices(subset_indices, quality_indices):
            return subset_indices[quality_indices[subset_indices]]

        if quality_indices_train is not None:
            train_indices = filter_subset_indices(train_indices, quality_indices_train)
            val_indices = filter_subset_indices(val_indices, quality_indices_valtest)
            test_indices = filter_subset_indices(test_indices, quality_indices_valtest)

        return train_indices, val_indices, test_indices

    def filter_subset_indices_by_attribute_range(self, subset_indices, dataset_attribute, range_to_remove):
        """
        Returns subset_indices that are within and outside the specified target range
        """
        subset_targets = dataset_attribute[subset_indices]
        range_indices = (subset_targets > range_to_remove[0]) & (subset_targets < range_to_remove[1])

        in_dist_indices = subset_indices[~range_indices]
        out_dist_indices = subset_indices[range_indices]
        return in_dist_indices, out_dist_indices

    def filter_subset_indices_by_attribute(self, subset_indices, dataset_attribute, out_dist_value):
        """
        Returns subset_indices that have a specific attribute value as out_dist_indices and
        the remaining samples as in_dist_indices.
        """
        subset_attribute = dataset_attribute[subset_indices]
        in_dist_indices = subset_indices[subset_attribute != out_dist_value]
        out_dist_indices = subset_indices[subset_attribute == out_dist_value]
        return in_dist_indices, out_dist_indices

