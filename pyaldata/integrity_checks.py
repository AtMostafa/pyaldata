import numpy as np
import pandas as pd

from . import utils


def trials_are_same_length(trial_data: pd.DataFrame, ref_field: str = None) -> bool:
    """
    Check if all trials of a dataset have the same length

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used
    """
    # Vectorized: use get_trial_lengths directly instead of iterrows
    trial_lengths = utils.get_trial_lengths(trial_data, ref_field=ref_field)
    return len(np.unique(trial_lengths)) == 1


def all_integer(arr: np.ndarray) -> bool:
    """
    Check if all the values in arr are approximately integers
    """
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        return False
    return np.all(np.isclose(arr, arr.astype(int)))
