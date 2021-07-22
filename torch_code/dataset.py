import numpy as np
from pathlib import Path
import h5py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from utils.transforms import pad_waveforms
from utils.utils import get_quality_indices


# parsing the geolocated waveforms
def parse_L1B_hdf5(filepath,
                   quality_dict,
                   use_coverage=True,
                   keys_beam=('shot_number',),
                   noise_mean_key='noise_mean_corrected'):

    # initialize output dictionary
    keys_beam = list(keys_beam)
    keys_beam.append(noise_mean_key)

    out_dict = {}
    keys = keys_beam + ['rxwaveform']
    for key in keys:
        out_dict[key] = []

    # select beam names
    coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
    power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
    if use_coverage:
        beam_names = coverage_beams + power_beams
    else:
        beam_names = power_beams

    with h5py.File(filepath, 'r') as f:

        # if quality_dict=None, it is not given by the L2A file: set quality=1 for all samples
        if not quality_dict:
            quality_dict = {}
            for beam in beam_names:
                quality_dict[beam] = np.ones_like(f[beam]['shot_number'][:]).astype(np.bool)

        for beam in beam_names:
            print(beam)
            if not beam in f.keys():
                print('Beam: {} does not exist. continue...'.format(beam))
                continue

            for k in keys_beam:
                out_dict[k] = out_dict[k] + list(f[beam][k][:][quality_dict[beam]])

            # parse the waveforms (rxwaveform is a single array)
            print('parsing full rxwaveform array...')
            rxwaveform_all = np.array(f[beam]['rxwaveform'][:])
            print('rxwaveform_all.shape', rxwaveform_all.shape)
            print('extracting waveform starts and ends')
            # we cut it into parts belonging to each of the valid laser shot
            # index starts at 1 correct for python indexing starting at 0
            start = f[beam]['rx_sample_start_index'][:][quality_dict[beam]] - 1
            count = f[beam]['rx_sample_count'][:][quality_dict[beam]]
            end = start + count

            print('max count', np.max(count))

            print('cutting into single waveforms...')
            size_rxwaveform = len(rxwaveform_all)
            print('size_rxwaveform', size_rxwaveform)
            
            rxwaveform = []
            for i in range(len(start)):
                wave = rxwaveform_all[start[i]:end[i]]  # this is a view, not a copy
                rxwaveform.append(wave)

            out_dict['rxwaveform'] += rxwaveform

    # convert to numpy arrays
    for k in out_dict.keys():
        if k not in ['shot_number', 'rxwaveform']:
            out_dict[k] = np.array(out_dict[k], dtype=np.float32)
    out_dict['shot_number'] = np.array(out_dict['shot_number'], dtype=np.uint64)
    return out_dict


def parse_L2A_hdf5(filepath, use_coverage=True,
                   keys_beam=('shot_number', 'quality_flag', 'lat_lowestmode', 'lon_lowestmode'),
                   keys_land_cover=('modis_nonvegetated',)):

    keys_beam = list(keys_beam)
    keys_land_cover = list(keys_land_cover)

    # initialize output dictionary
    out_dict = {}
    keys = keys_beam + keys_land_cover
    for key in keys:
        out_dict[key] = []

    # select beam names
    coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
    power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

    if use_coverage:
        beam_names = coverage_beams + power_beams
    else:
        beam_names = power_beams

    # init quality dictionary (for each beam)
    quality_dict = {}

    with h5py.File(filepath, 'r') as f:
        for beam in beam_names:
            if not beam in f.keys():
                print('Beam: {} does not exist. continue...'.format(beam))
                continue

            quality_dict[beam] = np.array(f[beam]['quality_flag'][:], dtype=np.bool)

            for k in keys_beam:
                out_dict[k] = out_dict[k] + list(f[beam][k][:][quality_dict[beam]])

            for k in keys_land_cover:
                out_dict[k] = out_dict[k] + list(f[beam]['land_cover_data'][k][:][quality_dict[beam]])

    # convert to numpy arrays
    for k in out_dict.keys():
        if k != 'shot_number':
            out_dict[k] = np.array(out_dict[k])
    out_dict['shot_number'] = np.array(out_dict['shot_number'], dtype=np.uint64)
    out_dict['quality_flag'] = np.array(out_dict['quality_flag'], dtype=np.bool)
    return out_dict, quality_dict


def filter_quality(data_dict, quality_indices):
    for key in data_dict.keys():
        data_dict[key] = data_dict[key][quality_indices]
    return data_dict


def pad_waveform(waveform, pad_constant, out_size=1420):
    pad_after = out_size-len(waveform)
    padded_waveform = np.pad(waveform, pad_width=((0, pad_after),), mode='constant', constant_values=(0, pad_constant))
    return padded_waveform


class GediDataOrbitMem(Dataset):
    """
    This dataset class loads all waveforms from an L1B orbit h5 file and filters based on the corresponding L2A quality_flag.
    """

    def __init__(self, file_path_L1B, file_path_L2A=None, sample_length=1420, input_transforms=None,
                 noise_mean_key='noise_mean_corrected'):

        super(GediDataOrbitMem, self).__init__()

        self.file_path_L1B = file_path_L1B
        self.file_path_L2A = file_path_L2A
        self.input_transforms = input_transforms
        self.sample_length = sample_length

        if self.file_path_L2A:

            # parse L2A data
            print('Loading L2A...')
            self.data_L2A, self.quality_dict = parse_L2A_hdf5(filepath=self.file_path_L2A)

            for key in self.data_L2A.keys():
                print(key, self.data_L2A[key].shape, self.data_L2A[key].dtype)
        else:
            self.data_L2A, self.quality_dict = None, None

        # parse L1B data
        print('Loading L1B...')
        self.data_L1B = parse_L1B_hdf5(filepath=self.file_path_L1B, quality_dict=self.quality_dict,
                                       noise_mean_key=noise_mean_key)

        self.inputs = self.data_L1B['rxwaveform']
        self.mean_noise_lvls = self.data_L1B[noise_mean_key]

        print('preprocessing samples...')
        self._preprocess_samples()

        # expand dimension of waveforms
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.inputs = self.inputs[..., None]
        
        self.data_L1B['rxwaveform'] = self.inputs
        
        for key in self.data_L1B.keys():
            print(key, self.data_L1B[key].shape, self.data_L1B[key].dtype)

        print('inputs.shape:', self.inputs.shape)
        print('mean_noise_lvls.shape:', self.mean_noise_lvls.shape)

        if self.file_path_L2A:
            assert np.array_equal(self.data_L1B['shot_number'], self.data_L2A['shot_number']), 'shot_number in L1B and L2A do not have the same order.'


    def _preprocess_samples(self):
        """
        Applies three different transformations to the waveforms
        """

        for i in range(len(self.inputs)):
            
            # pad first with mean: 
            self.inputs[i] = pad_waveform(self.inputs[i], pad_constant=self.mean_noise_lvls[i], out_size=1420)
            # subtract noise level
            self.inputs[i] = self.inputs[i] - self.mean_noise_lvls[i]
            # normalize integral to 1 (total energy return)
            self.inputs[i] = self.inputs[i] / np.sum(self.inputs[i])

    def __getitem__(self, index):
        """
        Returns the index-th waveform and the corresponding target.
        :param index: index within 0 and len(self)
        :return: index-th waveform
        """
        sample = self.inputs[index]

        if self.input_transforms:
            sample = self.input_transforms(sample)
        return sample

    def __len__(self):
        return len(self.inputs)


class AbstractDataMem(Dataset, ABC):
    """
        This dataset class loads all data into memory.
    """

    def __init__(self, input_path, target_path='', min_gt=-np.inf, max_gt=np.inf, sample_length=1420,
                 input_transforms=None, target_transforms=None):
        """
        Constructor, inherits from torch.utils.data.Dataset.
        :param input_path: numpy file of waveforms
        :param target_path: numpy file of targets
        :param max_gt: upper bound of ground truth height
        :param sample_length: maximum sample length
        :param input_transforms: torchvision.transforms.Compose or single transform
        :param target_transforms: torchvision.transforms.Compose or single transform
        """
        super(AbstractDataMem, self).__init__()
        self.input_path = Path(input_path)
        self.target_path = Path(target_path)
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.min_gt = min_gt
        self.max_gt = max_gt
        self.sample_length = sample_length

        self.inputs, self.targets, self.mean_noise_lvls, self.quality_indices_train, self.quality_indices_valtest, self.split_attribute = self._get_data()
        self._preprocess_samples()

    @abstractmethod
    def _get_data(self):
        pass

    def _preprocess_samples(self):
        """
        Applies three different transformations to the waveforms
        """
        assert np.ndim(self.inputs) == 3, 'waveforms should have the shape (num_samples, sample_length, num_featuers=1) e.g. (N, 1420, 1)'
        # subtract noise level
        self.inputs = self.inputs - self.mean_noise_lvls[..., None, None]
        # normalize integral to 1 (total energy return)
        self.inputs = self.inputs / np.sum(self.inputs, axis=1)[..., None]
        # pad waveforms to a fixed length with zeros at the end
        self.inputs = pad_waveforms(waveforms=self.inputs, out_size=self.sample_length)

    def __getitem__(self, index):
        """
        Returns the index-th waveform and the corresponding target.
        :param index: index within 0 and len(self)
        :return: index-th waveform
        """
        sample, target, mean_noise_lvl = self.inputs[index], self.targets[index], self.mean_noise_lvls[index]

        # input: Normalize_in, ToTensor
        if self.input_transforms:
            sample = self.input_transforms(sample)
        # target: Normalize_target
        if self.target_transforms:
            target = self.target_transforms(target)
        return sample, target

    def __len__(self):
        return len(self.inputs)


class CrossOverDataMem(AbstractDataMem):
    """
    This dataset class loads all crossover waveforms into memory.
    This class differs from the main class as it loads only one big numpy dictionary including all inputs AND targets.
    """
    def __init__(self, input_path, target_path='', min_gt=-np.inf, max_gt=np.inf, sample_length=1420,
                 input_transforms=None, target_transforms=None,
                 input_key='rxwaveform', target_key='als_rh098', settings_index=3, pearson_thresh=0.95,
                 split_attribute_name=None, noise_mean_key='noise_mean'):
        """
        :param input_key: input key to use (see args.input_key)
        :param target_key: target key to use (see args.target_key)
        :param settings_index: what setting to use (see self._get_data)
        """
        self.input_key = input_key
        self.target_key = target_key
        self.noise_mean_key = noise_mean_key
        self.settings_index = settings_index
        self.pearson_thresh = pearson_thresh
        self.split_attribute_name = split_attribute_name
        super(CrossOverDataMem, self).__init__(input_path, target_path, min_gt, max_gt, sample_length, input_transforms,
                                               target_transforms)

    def _get_data(self):
        """
        Load numpy files into memory as np.float32 arrays.
        For cross over data, all samples are within one file including targets.
        :return: loaded input and target without preprocessing.
        """
        if not self.input_path.exists():
            raise FileNotFoundError('The file {} does not exist.'.format(self.input_path))
        if not self.target_path.exists():
            raise FileNotFoundError('The file {} does not exist.'.format(self.target_path))

        # Load samples and ground truth
        print('Start loading dataset.')
        data = np.load(self.input_path, allow_pickle=True).item()
        inputs = np.array(data[self.input_key], dtype=np.float32, copy=False)[..., None]
        targets = np.array(data[self.target_key], dtype=np.float32, copy=False)
        mean_noise_lvls = np.array(data[self.noise_mean_key], dtype=np.float32, copy=False)

        if not self.split_attribute_name is None:
            split_attribute = np.array(data[self.split_attribute_name], copy=False)
        else:
            split_attribute = None

        settings = [{'night_strong': True, 'day_strong': False, 'night_coverage': False, 'day_coverage': False},
                    {'night_strong': True, 'day_strong': True, 'night_coverage': False, 'day_coverage': False},
                    {'night_strong': True, 'day_strong': True, 'night_coverage': True, 'day_coverage': False},
                    {'night_strong': True, 'day_strong': True, 'night_coverage': True, 'day_coverage': True}]

        setting = settings[self.settings_index]

        # Filter samples within valid range
        print('Start filtering dataset.')
        valid_indices = np.logical_and(targets >= self.min_gt, targets <= self.max_gt)
        # filter night and strong beams
        night_coverage_indices, filter_string = self.filter_shots(data=data,
                                                                  night_strong=setting['night_strong'],
                                                                  day_strong=setting['day_strong'],
                                                                  night_coverage=setting['night_coverage'],
                                                                  day_coverage=setting['day_coverage'])
        valid_indices = np.logical_and(valid_indices, night_coverage_indices)

        # filter by quality criteria
        do_filter_quality = True
        # check if all quality criteria keys exists (difference between crossover v1 and v2)
        for key in ['ground_elev_cog', 'dz_pearson', 'dz_count', 'pearson']:
            do_filter_quality = do_filter_quality & (key in data)
        print('do_filter_quality:', do_filter_quality)

        if do_filter_quality:
            # to allow a different pearson threshold between training and testing
            quality_indices_train = get_quality_indices(data_crossover=data, pearson_thresh=self.pearson_thresh)
            quality_indices_valtest = get_quality_indices(data_crossover=data, pearson_thresh=0.95)

            quality_indices_train = quality_indices_train[valid_indices]
            quality_indices_valtest = quality_indices_valtest[valid_indices]
        else:
            quality_indices_train = None
            quality_indices_valtest = None

        inputs = inputs[valid_indices]
        targets = targets[valid_indices]
        mean_noise_lvls = mean_noise_lvls[valid_indices]
        if split_attribute is not None:
            split_attribute = split_attribute[valid_indices]

        print('inputs.shape', inputs.shape)
        print('targets.shape', targets.shape)
        print('mean_noise_lvls.shape', mean_noise_lvls.shape)
        if split_attribute is not None:
            print('split_attribute.shape', split_attribute.shape)

        print('Done loading dataset.')
        return inputs, targets, mean_noise_lvls, quality_indices_train, quality_indices_valtest, split_attribute

    @staticmethod
    def filter_shots(data, night_strong=True, day_strong=True, night_coverage=True, day_coverage=True):
        night_indices = data['solar_elevation'] < 0
        day_indices = ~night_indices
        coverage_indices = data['coverage_flag'] == 1
        strong_indices = ~coverage_indices

        out_str = ''

        # init: all points are invalid
        valid_indices = np.repeat(0, repeats=len(data['shot_number']))

        if night_strong:
            indices = np.logical_and(night_indices, strong_indices)
            valid_indices = np.logical_or(valid_indices, indices)
            out_str += 'night-strong'
        if day_strong:
            indices = np.logical_and(day_indices, strong_indices)
            valid_indices = np.logical_or(valid_indices, indices)
            out_str += '_day-strong'
        if night_coverage:
            indices = np.logical_and(night_indices, coverage_indices)
            valid_indices = np.logical_or(valid_indices, indices)
            out_str += '_night-coverage'
        if day_coverage:
            indices = np.logical_and(day_indices, coverage_indices)
            valid_indices = np.logical_or(valid_indices, indices)
            out_str += '_day-coverage'

        return valid_indices, out_str


class CustomSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, input_transforms=None, target_transforms=None, augmentation_transforms=None):
        """
        Constructor, inherits from torch.utils.data.Dataset.
        :param dataset: pytorch dataset object (e.g. SimulatedDataMem)
        :param indices: numpy file of sample indices defining the subset
        :param input_transforms: torchvision.transforms.Compose or single transform
        :param target_transforms: torchvision.transforms.Compose or single transform
        :param augmentation_transforms: torchvision.transforms.Compose or single transform. Expects a tuple (input_, target)
        """
        super(CustomSubset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.augmentation_transforms = augmentation_transforms

    def __getitem__(self, idx):
        input_, target = self.dataset[self.indices[idx]]

        # augmentation
        # Note: since we adjust the target in its original scale, augmentation is applied before normalization.
        if self.augmentation_transforms:
            input_, target = self.augmentation_transforms((input_, target))

        if self.input_transforms is not None:
            input_ = self.input_transforms(input_)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return input_, target

    def __len__(self):
        return len(self.indices)

