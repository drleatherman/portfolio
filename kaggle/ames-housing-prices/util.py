"""
A utility module for functions used in the analysis of the Ames Housing Dataset.
"""


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import logging


def calc_vif(X):
    """Calculate Variance Inflation Factor for a given dataframe to detect multicollinearity between features. This is needed for regression models when there's a need to maintain interpretability of the resulting model (which should be pretty much always).


    :param X: pd.DataFrame.
    :returns: pd.DataFrame. "X" sorted by largest VIF first.

    """
    vif = pd.DataFrame()
    vif["variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False)


def get_top_zeroes(data, threshold=0.10):
    """For a given dataframe, return a Series containing the percentage of values that are 0 for each column. This is useful for exploratory data analysis.

    :param data: pd.DataFrame.
    :returns: pd.Series.

    """
    df = data.select_dtypes(include=np.number)
    top_zeroes = (
        # gets the 0-percentage for each column of a dataframe
        ((df == 0).sum() / len(df))
        .where(lambda _: _ > threshold)
        .dropna()
        .sort_values(ascending=False)
    )
    return top_zeroes


def get_logger(name=None, default_level=logging.INFO):
    """
    Create a Logger object to use instead of relying on print statements.

    :param name: str name of logger. If None, then root logger is used.
    :param default_level: Log Level
    :return logger: A Configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    ch = logging.StreamHandler()
    ch.setLevel(default_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
