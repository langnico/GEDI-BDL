import numpy as np

np.random.seed(2401)


def get_quality_indices(data_crossover, pearson_thresh=0.95):
    """

    Args:
        data_crossover: dict with arrays
        pearson_thresh: minimum threshold on pearson correlation

    Returns: bool array flagging quality samples

    """
    quality = ~np.isnan(data_crossover['ground_elev_cog']) & \
              (data_crossover['dz_pearson'] > 0.9) & \
              (data_crossover['dz_count'] > 5) & \
              (data_crossover['pearson'] > pearson_thresh)
    return quality


def filter_shots(data, night_strong=True, day_strong=True, night_coverage=True, day_coverage=True):
    """
    Get indices to create subsets of samples from specific noise level groups.
    Day vs. night and strong (full power beams) vs. coverage beams.

    Args:
        data: dict with data arrays
        night_strong: bool, return samples from this group if True
        day_strong: bool, return samples from this group if True
        night_coverage: bool, return samples from this group if True
        day_coverage: bool, return samples from this group if True

    Returns:
        valid_indices: bool array
        out_str: string to identify the settings
    """
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


# attention: this function returns bool arrays to filter the data (not indices)
def filter_subset_indices_by_target_range(subset_attribute, range_to_remove):
    """
    Returns bool arrays to create subsets that are within and outside the specified target range

    Note: While this function returns bool arrays, the pendant
          Trainer.filter_subset_indices_by_attribute_range (in run.py) returns indices.

    Args:
        subset_attribute: array of attribute which is used for splitting
        range_to_remove: tuple (min_value, max_value)

    Returns:
        in_dist_indices: bool array
        out_dist_indices: bool array
    """
    range_indices = (subset_attribute > range_to_remove[0]) & (subset_attribute < range_to_remove[1])
    print(len(range_indices))
    in_dist_indices = ~range_indices
    out_dist_indices = range_indices
    return in_dist_indices, out_dist_indices


def filter_subset_indices_by_attribute(subset_attribute, out_dist_value):
    """
    Returns bool arrays to create subsets that have a specific attribute value as out_dist_indices and
        the remaining samples as in_dist_indices.

    Note: While this function returns bool arrays, the pendant
          Trainer.filter_subset_indices_by_attribute (in run.py) returns indices.

    Args:
        subset_attribute: array of attribute which is used for splitting
        out_dist_value: attribute value considered to be out of distribution

    Returns:
        in_dist_indices: bool array
        out_dist_indices: bool array
    """
    in_dist_indices = subset_attribute != out_dist_value
    out_dist_indices = subset_attribute == out_dist_value
    return in_dist_indices, out_dist_indices

