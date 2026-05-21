import warnings
from typing import Callable

import numpy as np
import pandas as pd

from . import tools, utils

warnings.simplefilter("always", UserWarning)


__all__ = [
    "extract_interval_from_signal",
    "generate_epoch_fun",
    "restrict_to_interval",
    "slice_around_index",
    "slice_around_point",
    "slice_between_points",
    "slice_in_trial",
]


@utils.copy_td
def restrict_to_interval(
    trial_data: pd.DataFrame,
    start_point_name: str = None,
    end_point_name: str = None,
    rel_start: int = 0,
    rel_end: int = 0,
    before: int = None,
    after: int = None,
    epoch_fun: Callable[[pd.Series], slice] = None,
    warn_per_trial: bool = False,
    ignore_warnings: bool = False,
    reset_index: bool = True,
    ref_field: str = None,
):
    """
    Restrict time-varying fields to an interval around a time point or between two time points (inclusive on both ends)

    trial_data : pd.DataFrame
        data in trial_data format
    start_point_name : str, optional
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start : int, default 0
        when to start extracting relative to the starting time point
        replaces the 'before' option
    rel_end : int, default 0
        when to stop extracting relative to the ending time point
        replaces the 'after' option
    before (deprecated) : int, optional, default None
        number of time points to extract before the starting time point
        Please use rel_start instead.
    after (deprecated): int, optional, default None
        number of time points to extract after the ending time point
        Please use rel_end instead.
    epoch_fun : function, optional
        function that takes a trial and returns the epoch to extract
    warn_per_trial : bool, optional, default False
        give more detailed warnings about indexing in each problematic trial
    ignore_warnings : bool, optional, default False
        Won't print anything if True
    reset_index : bool, optional, default True
        whether to reset the dataframe index to [0,1,2,...]
        or keep the original indices of the kept trials
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    data in trial_data format
    """
    if before is not None:
        if not ignore_warnings:
            warnings.warn("'before' is deprecated. Use 'rel_start' instead.")
        rel_start = -before
    if after is not None:
        if not ignore_warnings:
            warnings.warn("'after' is deprecated. Use 'rel_end' instead.")
        rel_end = after

    assert (start_point_name is None) ^ (
        epoch_fun is None
    ), "Give either start_point_name or epoch_fun."

    idx_fields = [col for col in trial_data.columns.values if col.startswith("idx")]
    time_fields = utils.get_time_varying_fields(trial_data, ref_field)

    # generate epoch_fun if the interval is given with time points
    if start_point_name is not None:
        epoch_fun = generate_epoch_fun(start_point_name, end_point_name, rel_start, rel_end)

    # Single iterrows pass: compute epoch slices and validity together.
    # This avoids re-calling epoch_fun N*n_fields times later.
    all_slices: list = []
    kept_trials_mask_list: list = []
    for (_, trial) in trial_data.iterrows():
        sl = epoch_fun(trial)
        all_slices.append(sl)
        kept_trials_mask_list.append(slice_in_trial(trial, sl, warn_per_trial))
    kept_trials_mask = np.array(kept_trials_mask_list, dtype=bool)

    # warn about dropping the problematic trials
    if not ignore_warnings:
        if np.any(~kept_trials_mask):
            warnings.warn(
                f"Dropping {len(trial_data.trial_id.values[~kept_trials_mask])} trials because of invalid time indexing. For more information, try warn_per_trial=True",
                stacklevel=3,
            )

    # keep only the slices for valid trials
    kept_slices = [sl for sl, keep in zip(all_slices, kept_trials_mask) if keep]

    # only keep trials in which the indexing works properly
    trial_data = tools.select_trials(trial_data, kept_trials_mask, reset_index)

    # cut time varying signals using precomputed slices (no extra epoch_fun calls)
    trim_temp = {
        col: [arr[sl, ...] for arr, sl in zip(trial_data[col], kept_slices)]
        for col in time_fields
    }
    trial_data = trial_data.assign(**trim_temp)

    # adjust idx fields
    def _adjust_field(val, new_T):
        if isinstance(val, (np.ndarray, list)):
            return np.array([np.nan if (idx < 0 or idx > new_T) else idx for idx in val])
        elif (val < 0) or (val > new_T):
            return np.nan
        else:
            return val

    new_time_lengths = [arr.shape[0] for arr in trial_data[time_fields[0]]]
    # zero_points come directly from precomputed slices — no extra iterrows
    zero_points = [sl.start for sl in kept_slices]

    # Save absolute session indices before the loop shifts them
    if 'idx_trial_start' in idx_fields:
        abs_trial_starts = [
            int(np.asarray(gs).item()) + zp
            for gs, zp in zip(trial_data['idx_trial_start'].to_numpy(), zero_points)
        ]

    for col in idx_fields:
        trial_data[col] = [
            idx - zero_point for (idx, zero_point) in zip(trial_data[col], zero_points)
        ]

        # set indices that are now invalid (i.e. not in the restricted interval) to nan
        trial_data[col] = [
            _adjust_field(idx, new_T)
            for (idx, new_T) in zip(trial_data[col], new_time_lengths)
        ]

    # Restore idx_trial_start and idx_trial_end as absolute session indices
    if 'idx_trial_start' in idx_fields:
        trial_data['idx_trial_start'] = abs_trial_starts
    if 'idx_trial_end' in idx_fields:
        trial_data['idx_trial_end'] = [
            s + t - 1 for s, t in zip(abs_trial_starts, new_time_lengths)
        ]

    if 'trial_length' in trial_data.columns:
        trial_data['trial_length'] = new_time_lengths

    return trial_data


def slice_around_index(idx: int, before: int, after: int) -> slice:
    """
    Return a slice around an index
    Length will be before + after + 1

    Parameters
    ----------
    idx : int
        index around which to create the interval
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice(idx-before, idx+after+1)
    """
    start = idx - before
    end = idx + after + 1

    if np.isfinite(start):
        start = int(np.asarray(start).item())
    if np.isfinite(end):
        end = int(np.asarray(end).item())

    return slice(start, end)


def slice_around_point(trial: pd.Series, point_name: str, before: int, after: int) -> slice:
    """
    Return a slice around a time point in the trial
    Length will be before + after + 1

    Parameters
    ----------
    trial : pd.Series
        a row from a trial_data dataframe
        representing a trial
    point_name : str
        name of the time point around which to create the interval
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice object
    """
    start = trial[point_name] - before
    end = trial[point_name] + after + 1

    if np.isfinite(start):
        start = int(np.asarray(start).item())
    if np.isfinite(end):
        end = int(np.asarray(end).item())

    return slice(start, end)


def slice_between_points(
    trial: pd.DataFrame, start_point_name: str, end_point_name: str, before: int, after: int
) -> slice:
    """
    Return a slice that starts before start_point_name and ends after end_point_name

    Parameters
    ----------
    trial : pd.Series
        a row from a trial_data dataframe
        representing a trial
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str
        name of the time point around which the interval ends
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice object
    """
    start = trial[start_point_name] - before
    end = trial[end_point_name] + after + 1

    if np.isfinite(start):
        start = int(np.asarray(start).item())
    if np.isfinite(end):
        end = int(np.asarray(end).item())

    return slice(start, end)


def generate_epoch_fun(
    start_point_name: str, end_point_name: str = None, rel_start: int = 0, rel_end: int = 0
) -> Callable[[pd.Series], slice]:
    """
    Return a function that slices a trial around/between time points

    Parameters
    ----------
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start : int, default 0
        when to start extracting relative to the starting time point
        replaces the 'before' option
    rel_end : int, default 0
        when to stop extracting relative to the ending time point
        replaces the 'after' option

    Returns
    -------
    generated_epoch_fun : function
        function that can be used to extract the interval from a trial
    """
    if end_point_name is None:

        def generated_epoch_fun(trial):
            return slice_around_point(trial, start_point_name, -rel_start, rel_end)
    else:

        def generated_epoch_fun(trial):
            return slice_between_points(
                trial, start_point_name, end_point_name, -rel_start, rel_end
            )

    return generated_epoch_fun


def extract_interval_from_signal(
    trial_data: pd.DataFrame,
    signal: str,
    epoch_fun: Callable[[pd.Series], slice],
) -> list[np.ndarray]:
    """
    Extract an interval from a time-varying signal in the dataset

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        name of the signal to extract
    epoch_fun : function
        function that takes a trial and returns the epoch to extract

    Returns
    -------
    list of the extracted np.arrays
    """
    # Use zip over column values + apply for epoch computation instead of iterrows
    slices = trial_data.apply(epoch_fun, axis=1)
    return [arr[sl, ...] for arr, sl in zip(trial_data[signal].values, slices)]


def slice_in_trial(trial: pd.Series, sl: slice, warn: bool = False) -> bool:
    """
    Check if the slice is within the trial's time indices

    Parameters
    ----------
    trial : pd.Series
        trial to check
    sl : slice
        slice to check
    warn : bool, optional, default False
        whether to warn if the slice is outside
        the trial's time index

    Returns
    -------
    is_inside : bool
    """
    T = utils.get_trial_length(trial)

    is_inside = True

    if sl.start < 0:
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Trying to access index {sl.start} < 0"
            )
    if sl.stop > T:
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Trying to access index {sl.stop-1} >= {T}"
            )

    if not np.isfinite(sl.start):
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Starting point is {sl.start}"
            )
    if not np.isfinite(sl.stop):
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. End point is {sl.stop}"
            )

    return is_inside
