import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

from . import data_cleaning

__all__ = ["mat2dataframe",
           "load_pyaldata"]


def _struct_array_to_dict_of_lists(arr) -> dict:
    """Convert a numpy struct array (one record per trial) into a dict of
    per-trial lists, suitable for ``pd.DataFrame``.

    ``pd.DataFrame`` of a struct array whose fields hold 2-D arrays (e.g. spikes)
    raises "Per-column arrays must each be 1-dimensional", so the column dict is
    built explicitly. Used for MATLAB v7.3 files read via ``hdf5storage``.
    """
    if isinstance(arr, dict):  # hdf5storage may hand back a scalar struct as a dict
        return {k: [v] for k, v in arr.items()}
    arr = np.atleast_1d(arr).ravel()
    return {name: list(arr[name]) for name in arr.dtype.names}


def mat2dataframe(path: str, shift_idx_fields: bool, td_name: str = None) -> pd.DataFrame:
    """
    Load a trial_data .mat file and turn it into a pandas DataFrame

    Parameters
    ----------
    path : str
        path to the .mat file to load
        "Can also pass open file-like object."
    td_name : str, optional
        name of the variable under which the data was saved
    shift_idx_fields : bool
        whether to shift the idx fields
        set to True if the data was exported from matlab
        using its 1-based indexig

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe replicating the trial_data format
        each row is a trial
    """
    is_hdf5 = False
    try:
        mat = scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:
        # MATLAB v7.3 (HDF5). Read with hdf5storage
        try:
            import hdf5storage
        except ImportError:
            raise ImportError(
                "Must have hdf5storage installed to load MATLAB v7.3 files."
            )
        mat = hdf5storage.loadmat(str(path))
        is_hdf5 = True

    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]

    if td_name is None:
        if len(real_keys) == 0:
            raise ValueError("Could not find dataset name. Please specify td_name.")
        elif len(real_keys) > 1:
            raise ValueError("More than one datasets found. Please specify td_name.")

        assert len(real_keys) == 1

        td_name = real_keys[0]

    td = mat[td_name]
    df = pd.DataFrame(_struct_array_to_dict_of_lists(td) if is_hdf5 else td)

    df = data_cleaning.clean_0d_array_fields(df)
    df = data_cleaning.clean_integer_fields(df)

    if shift_idx_fields:
        df = data_cleaning.backshift_idx_fields(df)

    return df

def load_pyaldata(path: str, shift_idx_fields: bool = False, td_name: str = None) -> pd.DataFrame:
    """
    Load multiple pyal_data .mat files and turn it into a single pandas DataFrame

    Parameters
    ----------
    path : str
        path to the session directory, where the .mat files are saved
    td_name : str, optional
        name of the variable under which the data was saved
    shift_idx_fields : bool, optional
        whether to shift the idx fields
        set to True if the data was exported from matlab
        using its 1-based indexig

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe replicating the trial_data format
        each row is a trial
    """
    
    pyal_files = sorted(list(Path(path).glob("*.mat")))

    df = []
    for file in pyal_files:
        df_single = mat2dataframe(file, shift_idx_fields, td_name)
        df.append(df_single)
    df = pd.concat(df, ignore_index=True)
    
    return df