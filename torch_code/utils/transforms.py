import numpy as np
import torch
from scipy import ndimage


def pad_waveforms(waveforms, out_size=1420, axis=1, value=0):
    """
    Pads the waveforms on the right side to out_size (this preserves the georeference of the first returns)
    :param waveforms: array (N, waveform_length) with N samples
    :param out_size: expected waveform length
    :param axis: int, axis along to pad
    :param value: constant value used for padding
    :return: array (N, out_size) padded waveforms
    """

    if waveforms.shape[axis] == out_size:
        return waveforms
    else:
        waveform_length = waveforms.shape[axis]
        pad_after = out_size-waveform_length
        pad_widths = [(0, 0) for i in range(len(waveforms.shape))]
        pad_widths[axis] = (0, pad_after)
        padded_waveforms = np.pad(waveforms, pad_width=pad_widths, mode='constant', constant_values=value)
        return padded_waveforms


def denormalize(x, mean, std):
    x = x * std
    x = x + mean
    return x


class ToTensor(object):
    """
    Turn numpy array into torch tensor with CxW
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        x = x.permute((1, 0)).contiguous()
        return x


class Normalize(object):
    """normalize the input tensor with training mean and std.
    """

    def __init__(self, mean, std):
        """
        :param mean: scalar (float)
        :param std: scalar (float)
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x - self.mean
        x = x / self.std
        return x


# -------- DATA AUGMENTATION --------
class RandomShift(object):
    """
    Randomly shift the waveform by a fraction uniformly sampled from [-shift, shift].
    """

    def __init__(self, shift_interval, do_shift_target):
        """
        :param shift_interval: tuple (float, float) i.e. (-shift_left, shift_right) each in the range of [0, 1]
        :param do_shift_target: bool True: adjust the target elevation (e.g. delta = Z0 - ground) False: keep the target
        """
        self.shift_interval = shift_interval
        self.elev_resolution = 0.15  # GEDI waveforms have a resolution of 0.15 m between two returns
        self.do_shift_target = do_shift_target

    def __call__(self, sample):
        x, y = sample
        if self.shift_interval:
            # TODO: change to torch.rand
            rel_shift = np.random.uniform(self.shift_interval[0], self.shift_interval[1])
            abs_shift = int(rel_shift * len(x))
            x = ndimage.interpolation.shift(x, abs_shift, mode='nearest')

            # this is used to update the elevation target (delta = Z0 - ground)
            if self.do_shift_target:
                elev_shift = abs_shift * self.elev_resolution
                y = y + elev_shift
        return x, y


class RandomLabelNoise(object):
    """
    Add random noise to the target variable.
    """

    def __init__(self, rel_label_noise, distribution='uniform'):
        """
        :param rel_label_noise: scalar (float) fraction of label [0, 1].
        :param distribution: string choices= ['uniform', 'normal']. If normal, then rel_label_noise defines the std
        """
        self.rel_label_noise = rel_label_noise
        self.distribution = distribution

    def __call__(self, sample):
        x, y = sample
        if self.rel_label_noise:
            if self.distribution == 'uniform':
                # TODO: change to torch.rand
                rel_noise = np.random.uniform(-self.rel_label_noise, self.rel_label_noise)
            elif self.distribution == 'normal':
                # TODO: change to torch.rand
                rel_noise = np.random.normal(loc=0, scale=self.rel_label_noise)
            else:
                raise ValueError("This distribution is not impelmented. Use 'uniform' or 'normal'.")
            abs_noise = rel_noise * y
            y = y + abs_noise
        return x, y
