"""
This is a utility module that contains functions and snippets I've written
that I have found helpful for various analysis. I have a similar store of functions
and analysis for R but none for python yet.
"""
import logging
import pandas as pd
import numpy as np


def get_top_corr_vars(df, n=5):
    """
    Get the top n correlated columns for each column.

    df: pd.DataFrame - Source Dataframe
    n:  int          - number of largest elements to return
    """
    corr_dict = dict()
    df_tmp = df.copy()
    for col in df_tmp:
        # get most highly correlated variables
        # TODO: Find a more concise way to accomplish this.
        df_tmp_filt = df_tmp[col].where(lambda _: abs(_) != 1).dropna()

        df_corr = pd.DataFrame({"col_corr": df_tmp_filt})
        df_corr["corr_abs"] = df_tmp[col].abs()
        corr_dict[col] = df_corr.nlargest(n, columns="corr_abs").col_corr
    return corr_dict


def get_cols_to_center(corr_dict, threshold=0.7):
    """
    Return columns that should be centered.
    corr_dict: dict   - <K, V> where K = column name and V = pd.Series where the values are correlation coefficients.
    threshold: float  - the correlation coefficient at which the correlation is considered significant.
    """
    cols_to_center = set()
    for k in corr_dict.keys():
        # strange that the pandas.Series.__str__ doesn't autoprint the string version when concatenating with other strings.
        # ====== Column Name ======
        # <pandas.Series>
        log.debug("\n ====== " + k + " ====== \n" + str(corr_dict[k]) + "\n")
        # row names are the column names we want
        s = set(corr_dict[k].where(lambda _: abs(_) >= threshold).dropna().index.values)

        log.debug("Adding values to set: {}".format(s))
        cols_to_center = cols_to_center.union(s)
    return cols_to_center


def center_columns(data, cols_to_center=None):
    """
    Center columns based on correlation coefficients.
    data: pd.DataFrame - Source pandas DataFrame

    TODO: Make this more generic.
    """
    df_corr = pd.DataFrame()
    cols = list(data.select_dtypes(include=np.number).columns)
    cols.remove("Id")

    # prevent creation of duplicate columns. e.g. TotalBsmtSF_log_center_center
    cols = list(
        filter(
            lambda c: all(c2 not in [c + "_log", c + "_center"] for c2 in cols), cols
        )
    )
    log.info("Final Log Columns: {}".format(cols))

    df_corr = data[cols].corr()

    if cols_to_center is None:
        corr_dict = get_top_corr_vars(df_corr)
        cols_to_center = get_cols_to_center(corr_dict)

        # don't center columns that have already been centered.
        # At that point, centering is not the answer
        cols_to_center = list(
            filter(lambda _: not _.endswith("_center"), cols_to_center)
        )
    log.info("Centering columns: {}".format(cols_to_center))

    for col in cols_to_center:
        data[col + "_center"] = data[col] - data[col].mean()
    return data
