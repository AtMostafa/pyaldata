import pandas as pd
import scipy.io
from pathlib import Path

from . import data_cleaning

__all__ = ["mat2dataframe",
           "load_pyaldata"]


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
    try:
        mat = scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError("Must have mat73 installed to load mat73 files.")
        else:
            mat = mat73.loadmat(path)

    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]

    if td_name is None:
        if len(real_keys) == 0:
            raise ValueError("Could not find dataset name. Please specify td_name.")
        elif len(real_keys) > 1:
            raise ValueError("More than one datasets found. Please specify td_name.")

        assert len(real_keys) == 1

        td_name = real_keys[0]

    df = pd.DataFrame(mat[td_name])

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