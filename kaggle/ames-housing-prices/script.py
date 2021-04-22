#!/usr/bin/env jupyter-console
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import make_column_transformer
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import time
import logging
import sys
import os

KAGGLE_PROJ_PATH = os.path.join(os.getcwd(), "kaggle", "ames-housing-prices")

# append path so we can import functions from local modules
sys.path.append(KAGGLE_PROJ_PATH)

from util import get_logger  # noqa
from plot import generate_summary_plots, plot_regression_results, plot_regression_results2 # noqa


SEED = 12345

# Setup logging so we can avoid using print statements all over the place
global log
log = get_logger()


def apply_cleansing(df):
    """Apply data cleansing transformations and corrections to a dataframe. This is mostly centralized so data cleansing only happens in one place.

    :param df: pd.DataFrame. A pandas Dataframe representing the ames housing price dataset
    :returns: pd.DataFrame. A cleaned dataframe.

    """
    # Exterior2nd has mismatching values compared to Exterior1st
    df.Exterior2nd = (
        df.Exterior2nd.replace("Wd Shng", "WdShing")
        .replace("CmentBd", "CemntBd")
        .replace("Brk Cmn", "BrkComm")
    )
    return df


def apply_imputation(df, cols_to_exclude=[]):
    """For columns with missing data, apply imputation techniques to fill in the missing data. Note that this function is not needed when using the sklearn Pipeline API as imputation happens within the pipeline.

    :param df: pd.DataFrame. Data that contains all the columns needed to impute
    :param cols_to_exclude: list(str). A list of columns that should not be factored into imputation. For example, say that the predictor variable of interest is a part of "df". The imputed values should not take into consideration the predictor variable as it would cause data leakage and pollute the final results.
    :return df: pd.DataFrame. Dataframe with imputed values.

    """
    # store a copy of pre-imputed data just in case additional analysis is needed.
    data_pre_imp = (
        df.select_dtypes(include=np.number).copy().drop(
            cols_to_exclude, axis=1)
    )

    # impute mean using sklearn
    # TODO: check to make sure the imputations make sense afterwards
    imp_mean = IterativeImputer(random_state=SEED)
    data_trans = imp_mean.fit_transform(data_pre_imp)

    imputed_df = pd.DataFrame(data_trans, columns=data_pre_imp.columns)

    for col in imputed_df:
        df[col] = imputed_df[col]
    return df


def get_area_cols(df):
    """Return all of the columns that represent area measurements in the Ames Housing Dataset.

    :param df: pd.DataFrame. Ames Housing Dataset
    :returns: list(str). List of column names.

    """
    return list(filter(lambda _: any(x in _ for x in ["SF", "Area"]), df.columns)) + [
        "LotFrontage",
        "SalePrice",
    ]


def filter_suffix(ls, df, suffix="log"):
    """Filter a list of column names based on a provided suffix.

    :param ls: list(str). List of column names.
    :param df: pd.DataFrame. Dataframe containing columns that are being compared against.
    :param suffix: str. Suffix present in the column name that should be filtered for.
    :returns:

    """
    return list(
        map(lambda _: _ + "_" + suffix if (_ + suffix)
            not in df.columns else _, ls)
    )


def add_indicators(data):
    """Add indicator variables (using OneHotEncoder) to the Ames Housing Price Dataset. Note that this is done during a step when using the Pipeline API so this is only if building and fitting models without using the Pipeline API.

    :param data: pd.DataFrame. A dataframe representing the Ames Housing Dataset.
    :returns: pd.DataFrame. The source dataframe with indicator variables joined to it.

    """
    exterior_indicators = (
        # create indicator variables for both "Exterior" columns
        pd.get_dummies(data.Exterior1st, drop_first=True).add(
            pd.get_dummies(data.Exterior2nd, drop_first=True), fill_value=0
        )
    )
    exterior_indicators = (
        # rename them to be more descriptive
        exterior_indicators.rename(
            columns={
                c: "ExteriorHas" + c.replace(" ", "")
                for c in exterior_indicators.columns
            }
        )
        # in cases where both the Exterior1st and Exterior2nd was the same, we don't care so set to 1
        .replace(2, 1)
    )

    condition_indicators = pd.get_dummies(data.Condition1, drop_first=True).add(
        pd.get_dummies(data.Condition2, drop_first=True), fill_value=0
    )

    condition_indicators = (
        condition_indicators.rename(
            columns={
                c: "ConditionHas" + c.replace(" ", "")
                for c in condition_indicators.columns
            }
        )
        # in cases where both the Condition1 and Condition2 was the same, we don't care so set to 1
        .replace(2, 1)
    )

    # TODO: record base level for each variable for inference
    neighborhood_indicators = pd.get_dummies(
        data.Neighborhood, prefix="nghbrhd", drop_first=True
    )
    saletype_indicators = pd.get_dummies(
        data.SaleType, prefix="SaleType", drop_first=True
    )
    yearsold_indicators = pd.get_dummies(
        data.YrSold, prefix="YrSold", drop_first=True)
    monthsold_indicators = pd.get_dummies(
        data.MoSold, prefix="MoSold", drop_first=True)
    rootmatl_indicators = pd.get_dummies(
        data.RoofMatl, prefix="RoofMatl", drop_first=True
    )
    exterqual_indicators = pd.get_dummies(
        data.ExterQual, prefix="ExterQual", drop_first=True
    )
    poolqc_indicators = pd.get_dummies(
        data.PoolQC, prefix="PoolQC", drop_first=True)
    kitchenqual_indicators = pd.get_dummies(
        data.KitchenQual, prefix="KitchenQual", drop_first=True
    )
    fireplacequ_indicators = pd.get_dummies(
        data.FireplaceQu, prefix="FireplaceQu", drop_first=True
    )

    df = pd.DataFrame(
        data.join(exterior_indicators)
        .join(condition_indicators)
        .join(neighborhood_indicators)
        .join(saletype_indicators)
        .join(yearsold_indicators)
        .join(monthsold_indicators)
        .join(rootmatl_indicators)
        .join(exterqual_indicators)
        .join(poolqc_indicators)
        .join(kitchenqual_indicators)
        .join(fireplacequ_indicators)
    )

    # These indicator variables are all 0's so removing them.
    df = df.drop(
        [
            "ExterQual_Po",
            "ExterQual_NA",
            "PoolQC_TA",
            "PoolQC_Po",
            "KitchenQual_Po",
            "KitchenQual_NA",
        ],
        axis=1,
    )
    return df


def apply_trans(df):
    """Apply general transformations to a dataframe containing the Ames Housing Dataset. This is slightly different than apply_cleansing as it is focused less on resolving Data Quality issues and moreso preparing it for modeling.

    :param df: pd.DataFrame. A dataframe containing the Ames Housing Dataset.
    :returns: pd.DataFrame. The source dataframe with transformations applied.

    """
    # Replace "NA" categories with "MISSING"
    cat_cols = df.columns[df.dtypes == "O"]
    categories = [df[column].unique() for column in df[cat_cols]]

    for cat in categories:
        cat[cat == None] = "MISSING"

    # apply log transformations
    for col in ["1stFlrSF", "GrLivArea", "LotFrontage", "SalePrice"]:
        df[col + "_log"] = df[col].apply(np.log)

    return df


def get_dataset2(path):
    """Retrieve the Ames Housing dataset, apply data cleansing operations, and transformations to it. This is to avoid having to orchestrate this in the main function.

    :param path: str. A path to a csv that pd.read_csv() can understand.
    :returns: pd.DataFrame.

    """
    data = pd.read_csv(path)

    # apply data cleansing operations
    data = apply_cleansing(data)

    # apply transformations
    data = apply_trans(data)
    return data


def fit_OLS(X, y, K=10):
    """Fit an OLS model using KFold cross validation. This was a first attempt. After playing with doing more model types, the Pipeline API was much nicer to use. This is being kept around for future reference though.

    :param X: pd.DataFrame. Dataframe containing feature variables.
    :param y: pd.Series. Series containing the target variable.
    :param K: int. number of folds to use for Cross Validation.
    :returns: list(float). K-length list of R^2 statistics.

    """
    model = LinearRegression()
    scores = []
    kfold = KFold(n_splits=K, shuffle=True, random_state=12345)

    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train], y.iloc[train])
        score = model.score(X.iloc[test], y.iloc[test])
        scores.append(score)

    return scores


def fit_OLS_statsmodels(X, y):
    """Fit an OLS model using the statsmodels which is closer to traditional statistics.
    :param X: pd.DataFrame. Dataframe containing feature variables.
    :param y: pd.Series. Series containing the target variable.
    :returns: statsmodels.regression.linear_model.RegressionResults.

    """
    # Fit the regression model
    sm_model = sm.OLS(y, X)
    sm_ols = sm_model.fit()
    return sm_ols


def fit_RidgeCV(X, y, K=10):
    """Fit a Ridge Regression Model with scikit learn. X is standardized beforehand since Ridge Regression works best when coefficients are similar sized.

    Ridge Regression (with CV) was employed for the Ames Housing Price Dataset because the initial chosen variables have a high degree of multicollinearity (verified by investigating VIF). In this case, Ridge Regresssion can help resolve multicollinearity by penalizing coefficients that contain redundant information.
    :param X: pd.DataFrame. Dataframe containing feature variables.
    :param y: pd.Series. Series containing the target variable.
    :param K: int. number of folds to use for Cross Validation.
    :returns: (RidgeCV, list(float)). Fitted Ridge Regression Model and a K-length list of R^2 statistics.

    """
    n_alphas = 200
    alphas = np.logspace(-10, 6, n_alphas)

    # standardize features for Ridge Regression
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    rr = RidgeCV(alphas=alphas)
    rr.fit(X_trans, y)
    score = rr.score(X_trans, y)
    return rr, score

    """
    """


def model_summary(model):
    """Return a DataFrame containing model summary statistics.

    :param model: sm.regression.linear_model.RegressionResults
    :returns: pd.DataFrame. Dataframe containing summary statistics of the model.

    """
    model_df = pd.DataFrame()

    # model values
    model_df["fitted"] = model.fittedvalues
    # residuals
    model_df["residuals"] = model.resid
    # normalized residuals
    model_df["norm_residuals"] = model.get_influence(
    ).resid_studentized_internal
    # absoluate squared normalized residuals
    model_df["abs_sq_norm_residuals"] = model_df["norm_residuals"].apply(
        lambda _: np.sqrt(np.abs(_))
    )
    # absolute residuals
    model_df["abs_residuals"] = model_df["residuals"].apply(np.abs)
    # leverage
    model_df["leverage"] = model.get_influence().hat_matrix_diag
    # Cook's Distance
    model_df["cooks"] = model.get_influence().cooks_distance[0]

    model_df = model_df.reset_index()
    return model_df


def build_estimators(X, impute_missing_val=None):
    """Using the Pipeline API in sci-kit learn, build pipelines and associate them with estimators. Much of this code is source from https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py, which was extremely helpful and relevant to this Dataset. Some modifications have been made based on analysis I did.

    :param X: pd.DataFrame. Dataframe containing feature variables.
    :param impute_missing_val: The value to use to indicate that a variable is missing. This is used during the imputation transformer in the pieplines.
    :returns: list((str, sklearn.pipeline.Pipeline)). Sklearn Pipelines that can be executed.

    """
    cat_cols = X.columns[X.dtypes == "O"]
    num_cols = X.select_dtypes(include=np.number).columns
    categories = [X[column].unique() for column in X[cat_cols]]

    categories_clean = []
    for cat in categories:
        categories_clean.append(
            np.array(
                list(map(lambda _: "MISSING" if _ is np.nan else _, cat)),
                dtype=cat.dtype,
            )
        )
        # cat[cat == None] = "MISSING"
    categories = categories_clean
    cat_proc_nlin = make_pipeline(
        SimpleImputer(
            missing_values=impute_missing_val, strategy="constant", fill_value="MISSING"
        ),
        OrdinalEncoder(categories=categories),
    )

    cat_proc_lin = make_pipeline(
        SimpleImputer(
            missing_values=impute_missing_val, strategy="constant", fill_value="MISSING"
        ),
        OneHotEncoder(categories=categories),
    )

    num_proc_nlin = make_pipeline(IterativeImputer(random_state=SEED))

    num_proc_lin = make_pipeline(IterativeImputer(
        random_state=SEED), StandardScaler())

    # transformation to use for non-linear estimators
    processor_nlin = make_column_transformer(
        (cat_proc_nlin, cat_cols), (num_proc_nlin,
                                    num_cols), remainder="passthrough"
    )

    # transformation to use for linear estimators
    processor_lin = make_column_transformer(
        (cat_proc_lin, cat_cols), (num_proc_lin, num_cols), remainder="passthrough"
    )

    lasso_pipeline = make_pipeline(processor_lin, LassoCV())

    rf_pipeline = make_pipeline(
        processor_nlin, RandomForestRegressor(random_state=SEED)
    )

    gradient_pipeline = make_pipeline(
        processor_nlin, HistGradientBoostingRegressor(random_state=SEED)
    )

    ridge_pipeline = make_pipeline(processor_lin, RidgeCV())

    estimators = [
        ("Random Forest", rf_pipeline),
        ("Lasso", lasso_pipeline),
        ("Gradient Boosting", gradient_pipeline),
        ("Ridge", ridge_pipeline),
    ]
    return estimators


def get_x_y(data):
    """Return the features (X) and target (y) for Ames Housing Dataset.

    These values were chosen initially by investigating univariate comparisons with boxplots and scatter plots between features and Log Saleprice. Log transformed features are used in cases where a log transformation turned a right-skewed distribution into a normal distribution.

    :param data: pd.DataFrame. Dataframe containing the Ames Housing Dataset.
    :returns: (pd.DataFrame, pd.Series). The features and target variable

    """
    features = [
        "1stFlrSF_log",
        "2ndFlrSF",
        "TotalBsmtSF",
        "GarageArea",
        "GrLivArea_log",
        "LotArea_log",
        "LotFrontage_log",
        "MasVnrArea",
        "WoodDeckSF",
        "BsmtFinSF1",
        "BsmtUnfSF",
        "EnclosedPorch",
        "ScreenPorch",
        "FullBath",
        "TotRmsAbvGrd",
    ]
    indicator_cols_dirty = set(data.filter(like="_")).difference(set(features))
    indicator_cols_clean = list(
        filter(lambda _: "_log" not in _, indicator_cols_dirty))
    X = data[features + indicator_cols_clean]
    y = data["SalePrice_log"]

    return X, y


def get_x_y2():
    """Return the features (X) and target (y) for Ames Housing Dataset.

    This version does not pre-select variables (unlike get_x_y()). This is because I learned that it is best practice to do variable selection within the confines of Cross-Validation (rather than before hand). This was evident in some of the R^2 statistics returned previously as there were several that were fine but a few that were much lower. Elements of Statistical Learning has a great, concise explanation of why it is not best practice to filter variables outside of Cross Valdation. I refer you to that text (https://web.stanford.edu/~hastie/ElemStatLearn/) for more detail.

    :returns: (pd.DataFrame, pd.Series). The features and target variable

    """

    data = get_dataset2(os.path.join(KAGGLE_PROJ_PATH, "data", "train.csv"))
    y = data["SalePrice_log"]
    X = data.drop(
        ["1stFlrSF", "GrLivArea", "LotFrontage",
            "SalePrice", "SalePrice_log", "Id"],
        axis=1,
    )

    # treat YrSold and MoSold as indicator variables to avoid
    # dealing with autocorrelation
    time_ind_cols = ["YrSold", "MoSold"]

    X[time_ind_cols] = X[time_ind_cols].astype("str")
    X[X.select_dtypes(include="int64").columns] = X.select_dtypes(
        include="int64"
    ).astype("float")
    return X, y


def get_x_y3():
    """Return the features (X) and target (y) for Ames Housing Dataset.

    This version does not pre-select variables (unlike get_x_y()) and it uses a version of the dataset from OpenML. The example I was looking at it used to so I thought I'd keep it in here for future reference.

    :returns: (pd.DataFrame, pd.Series). The features and target variable

    """

    from sklearn.datasets import fetch_openml

    df = fetch_openml(name="house_prices", as_frame=True)
    X = df.data
    y = df.target

    log_cols = ["1stFlrSF", "GrLivArea", "LotFrontage"]
    X[log_cols] = X[log_cols].apply(np.log)

    X = X.drop(
        ["Id"],
        axis=1,
    )

    # treat YrSold and MoSold as indicator variables to avoid
    # dealing with autocorrelation
    time_ind_cols = ["YrSold", "MoSold"]

    X[time_ind_cols] = X[time_ind_cols].astype("str")

    return X, np.log(y)


def run(plot_summaries=True):
    """Initial entrypoint function to kick run all the models. Kept for posterity and future use.

    :param plot_summaries: bool. Generate summary plots.
    :returns:

    """
    # get cleansed dataset all ready to go
    data = get_dataset(
        os.path.join(os.getcwd(), "kaggle",
                     "ames-housing-prices", "data", "train.csv")
    )

    # generate all summary plots
    if plot_summaries:
        generate_summary_plots(data)

    X, y = get_x_y(data)

    score_ols = fit_OLS(X, y, K=10)
    log.info("R^2 OLS: {}".format(score_ols))

    sm_ols = fit_OLS_statsmodels(X, y)

    log.info(sm_ols.summary())

    rr_fitted, rr_score = fit_RidgeCV(X, y)
    log.info("Best alpha for Ridge Regression: {}".format(rr_fitted.alpha_))
    log.info("Best score for Ridge Regression: {}".format(rr_score))

    # pair coefficients back with column names
    rr_coefs = {k: v for k, v in zip(X.columns.values, rr_fitted.coef_)}
    log.info(rr_coefs)


def run_pipelines():
    """Second iteration of the entrypoint function. This builds and executes pipelines using the Pipeline APIs.

    :returns:

    """
    X, y = get_x_y2()

    estimators = build_estimators(X, np.nan)

    fig, axs = plt.subplots(2, 2, figsize=(9, 7), facecolor="white")
    axs = np.ravel(axs)

    try:

        for ax, (name, est) in zip(axs, estimators):
            start_time = time.time()
            score = cross_validate(
                est,
                X,
                y,
                scoring=["r2", "neg_mean_absolute_error"],
                n_jobs=-1,
                verbose=logging.DEBUG,
            )
            elapsed_time = time.time() - start_time

            log.info("Score: {}".format(score))
            y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)

            plot_regression_results(
                ax,
                y,
                y_pred,
                name,
                (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
                    np.mean(score["test_r2"]),
                    np.std(score["test_r2"]),
                    -np.mean(score["test_neg_mean_absolute_error"]),
                    np.std(score["test_neg_mean_absolute_error"]),
                ),
                elapsed_time,
            )

        plt.suptitle("Single predictors versus stacked predictors")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    except (Exception, ValueError) as e:
        log.exception(e)


if __name__ == "__main__":
    # run(False)
    run_pipelines()
