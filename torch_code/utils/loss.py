import torch
import torch.nn as nn


class MELoss(nn.Module):
    def __init__(self):
        super(MELoss, self).__init__()

    def __call__(self, prediction, target):
        return torch.mean(prediction - target)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def __call__(self, prediction, target):
        return torch.sqrt(torch.mean((prediction - target)**2))


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """
    def __init__(self):
        super(GaussianNLL, self).__init__()
        self.eps = 1e-8

    def __call__(self, prediction, log_variance, target):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        # add a small constant to the variance for numeric stability
        variance = torch.exp(log_variance) + self.eps
        return torch.mean(0.5 / variance * (prediction - target)**2 + 0.5 * torch.log(variance))


class LaplacianNLL(nn.Module):
    """
    Laplacian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
    Furthermore, the constant log term is discarded.
    """
    def __init__(self):
        super(LaplacianNLL, self).__init__()
        self.eps = 1e-8

    def __call__(self, prediction, log_variance, target):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        # add a small constant to the variance for numeric stability
        variance = torch.exp(log_variance) + self.eps
        return torch.mean(1 / variance * torch.abs(prediction - target) + torch.log(variance))




