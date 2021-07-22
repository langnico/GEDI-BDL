import numpy as np

def mse(x, y):
    return np.nanmean(np.square(y - x))


def rmse(x, y):
    return float(np.sqrt(np.nanmean(np.square(y - x))))


def mae(x, y):
    return float(np.nanmean(np.abs(y - x)))


def me(x, y):
    return float(np.nanmean(y - x))


def rmspe(x, y):
    return float(np.sqrt(np.nanmean(np.square((y - x) / x))) * 100)


def mape(x, y):
    return float(np.nanmean(np.abs((y - x) / x)) * 100)


def mpe(x, y):
    return float(np.nanmean((y - x) / x) * 100)


def get_metrics_dict():
    metrics_dict_fun = {'RMSE': rmse,
                        'MAE': mae,
                        'ME': me,
                        'RMSPE': rmspe,
                        'MAPE': mape,
                        'MPE': mpe}
    return metrics_dict_fun

