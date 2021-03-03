#!/usr/bin/env jupyter-console

import math
import logging
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from plotnine import *  # plotting
from itertools import islice
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ORD_COND_LS = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]

ORDINAL_VARS = [
    {"BsmtFullBath": None},
    {"BedroomAbvGr": None},
    {"TotRmsAbvGrd": None},
    {"OverallQual": None},
    {"OverallCond": None},
    {"HeatingQC": ORD_COND_LS},
    {"GarageCond": ORD_COND_LS},
    {"FireplaceQu": ORD_COND_LS},
    {"GarageCars": None},
    {"GarageQual": ORD_COND_LS},
    {"ExterCond": ORD_COND_LS},
    {"Fireplaces": None},
    {"Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]},
    {"BsmtQual": ORD_COND_LS},
    {"BsmtCond": ORD_COND_LS},
    {"KitchenQual": ORD_COND_LS},
    {"BsmtExposure": ["Gd", "Av", "Mn", "No", "NA"]},
    {"ExterQual": ORD_COND_LS},
    {"BsmtFullBath": None},
    {"FullBath": None},
    {"KitchenAbvGr": None},
    {"BsmtHalfBath": None},
    {"PavedDrive": ["Y", "P", "N"]},
    {"PoolQC": ORD_COND_LS},
    {"HalfBath": None},
    {"GarageFinish": ["Fin", "RFn", "Unf", "NA"]},
    {"LandSlope": ["Gtl", "Mod", "Sev"]},
    {"Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO"]},
]

CATEGORICAL_VARS = [
    "Neighborhood",
    "Exterior1st",
    "Exterior2nd",
    "MSSubClass",
    "SaleType",
    "Condition1",
    "Condition2",
    "RoofMatl",
    "HouseStyle",
    "Functional",
    "BsmtFinType1",
    "GarageFinish",
    "GarageType",
    "Heating",
    "BsmtFinType2",
    "SaleCondition",
    "Foundation",
    "RoofStyle",
    "LotConfig",
    "BldgType",
    "Electrical",
    "MSZoning",
    "LandContour",
    "MiscFeature",
    "LotShape",
    "MasVnrType",
    "CentralAir",
    "Alley",
    "Street",
]


def get_logger(name=None, default_level=logging.INFO):
    """
    Build a logger object
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


def draw_cat_plots(
    df, title="Categorical", size="regular", y="SalePrice_log", x="value"
):
    """
    Draw plot for categorical data. In this case, it's boxplots. This is mostly used as a utility for keeping plotting code DRY.

    df: pd.DataFrame - A DataFrame to plot. Has the following columns: SalePrice, variable, value
    title: str       - A phrase to insert into the plot's title.
    size: str        - The size of the png that should be produced. Useful in cases where we need to control real-estate by plot. options: regular, large.
    """
    plt = (
        ggplot(df)
        + aes(y=y, x=x)
        + geom_boxplot()
        + stat_summary(fun_y=np.mean, geom="point", color="red", fill="red")
        + facet_wrap("variable", scales="free")
        + labs(
            x="",
            y="Log Sale Price",
            title="Adjusted Sales Price vs {} Variables".format(title),
        )
    )
    if size == "regular":
        # adjust x and y axis' so that the scales show up nicely
        plt += theme(
            subplots_adjust={"hspace": 0.5, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(8, 6),
        )
    else:
        plt += theme(
            subplots_adjust={"hspace": 0.5, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(13, 10),
        )
    plt.draw()


def apply_cleansing(df):
    # Exterior2nd has mismatching values compared to Exterior1st
    df.Exterior2nd = (
        df.Exterior2nd.replace("Wd Shng", "WdShing")
        .replace("CmentBd", "CemntBd")
        .replace("Brk Cmn", "BrkComm")
    )
    return df


def apply_imputation(df, cols_to_exclude=[]):
    """
    Impute missing values
    """
    # store a copy of pre-imputed data just in case additional analysis is needed.
    # A few variables have been dropped for purposes of imputation
    # 1. SalePrice. Since this is the dependent variable, don't want to introduce data leakage by having the imputed values based off it.
    # 2. YrSold, MoSol. These are to be included as factors in the model to avoid dealing with autocorrelation, which would have to be accounted for here if used in as a numeric.
    # 3. Id. It's an identifier and meaningless in the context of regression.
    data_pre_imp = (
        df.select_dtypes(include=np.number).copy().drop(cols_to_exclude, axis=1)
    )

    # impute mean using sklearn
    # TODO: check to make sure the imputations make sense afterwards
    imp_mean = IterativeImputer(random_state=12345)
    data_trans = imp_mean.fit_transform(data_pre_imp)

    imputed_df = pd.DataFrame(data_trans, columns=data_pre_imp.columns)

    for col in imputed_df:
        df[col] = imputed_df[col]
    return df


def get_area_cols(df):
    return list(filter(lambda _: any(x in _ for x in ["SF", "Area"]), df.columns)) + [
        "LotFrontage",
        "SalePrice",
    ]


def filter_suffix(ls, df, suffix="log"):
    return list(
        map(lambda _: _ + "_" + suffix if (_ + suffix) not in df.columns else _, ls)
    )


def add_indicators(data):
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
    yearsold_indicators = pd.get_dummies(data.YrSold, prefix="YrSold", drop_first=True)
    monthsold_indicators = pd.get_dummies(data.MoSold, prefix="MoSold", drop_first=True)
    rootmatl_indicators = pd.get_dummies(
        data.RoofMatl, prefix="RoofMatl", drop_first=True
    )
    exterqual_indicators = pd.get_dummies(
        data.ExterQual, prefix="ExterQual", drop_first=True
    )
    poolqc_indicators = pd.get_dummies(data.PoolQC, prefix="PoolQC", drop_first=True)
    kitchenqual_indicators = pd.get_dummies(
        data.KitchenQual, prefix="KitchenQual", drop_first=True
    )
    fireplacequ_indicators = pd.get_dummies(
        data.FireplaceQu, prefix="FireplaceQu", drop_first=True
    )
    masvnrtype_indicators = pd.get_dummies(
        data.MasVnrType, prefix="MasVnrType", drop_first=True
    )
    heatingqc_indicators = pd.get_dummies(
        data.HeatingQC, prefix="HeatingQC", drop_first=True
    )
    street_indicators = pd.get_dummies(data.Street, prefix="Street", drop_first=True)
    garagetype_indicators = pd.get_dummies(
        data.GarageType, prefix="GarageType", drop_first=True
    )
    overallqual_indicators = pd.get_dummies(
        data.OverallQual, prefix="OverallQual", drop_first=True
    )
    miscfeature_indicators = pd.get_dummies(
        data.MiscFeature, prefix="MiscFeature", drop_first=True
    )
    housestyle_indicators = pd.get_dummies(
        data.HouseStyle, prefix="HouseStyle", drop_first=True
    )
    mssubclass_indicators = pd.get_dummies(
        data.MSSubClass, prefix="MSSubClass", drop_first=True
    )
    foundation_indicators = pd.get_dummies(
        data.Foundation, prefix="Foundation", drop_first=True
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
    # Add Log Transformations to area columns
    area_cols = get_area_cols(df)
    area_cols_renamed = filter_suffix(area_cols, df, suffix="log")
    df[area_cols_renamed] = df[area_cols].apply(np.log)

    # new columns
    df["HasExtraGarage"] = (df.MiscFeature == "Gar2").astype("int")
    df = add_indicators(df)
    return df


# Setup logging so we can avoid using print statements all over the place
global log
log = get_logger()


# Don't auto interpret NA as NaN. NA has an ordinal ranking with respect to other fields
# and converting it to NaN obfuscates that


def get_dataset(path):
    data = pd.read_csv(path)

    for _ in ORDINAL_VARS:
        for k, v in _.items():
            if v:
                # use the explicit Order defined for this column
                data[k] = pd.Categorical(data[k], categories=v, ordered=True)
            else:
                # use default ordering. Should be mostly integer columns here. e.g. 1 < 2 < 3 < 4
                data[k] = pd.Categorical(data[k], ordered=True)

    for _ in CATEGORICAL_VARS:
        data[_] = pd.Categorical(data[_])

    # make LotFrontage a continuous variable
    # data.LotFrontage = data.LotFron# tage.replace("NA", np.nan).astype("float")
    # data.MasVnrArea = data.MasVnrArea.replace("NA", np.nan).astype("float")

    # apply data cleansing operations
    data = apply_cleansing(data)

    # impute missing data
    data = apply_imputation(
        data, cols_to_exclude=["SalePrice", "YrSold", "MoSold", "Id"]
    )

    # apply transformations
    data = apply_trans(data)
    return data


def gen_logtrans_plots(data):
    # Unfortunately, there is not currently a good way to get plotnine plots to show up
    # side by side. matplotlib can do this but only with their plots.
    (
        ggplot(data)
        + aes(x="SalePrice")
        + geom_histogram()
        + labs(x="Sale Price (USD) (K)", y="Count", title="Sale Price")
        # + theme(figure_size = (6.4, 4.8))
    ).draw()

    (
        ggplot(data)
        + aes(x="SalePrice_log")
        + geom_histogram()
        + labs(x="Log Sale Price (USD)", y="Count", title="Adjusted Sale Price")
        # + theme(figure_size = (6.4, 4.8))
    ).draw()

    # do histograms of SalePrice for each neighborhood.
    (
        ggplot(data)
        + aes(x="SalePrice_log")
        + geom_histogram()
        + facet_wrap("Neighborhood")
        + labs(
            x="Log Sale Price (USD)",
            y="Count",
            title="Adjusted Sale Price by Neighborhood",
        )
        + theme(figure_size=(8, 6))
    )

    melted_df_raw = (
        data.select_dtypes(include=np.number)
        .melt(id_vars=["Id"], var_name="variable", value_name="value")
        .where(lambda _: ~_.variable.isin(["SalePrice", "SalePrice_log"]))
        .dropna()
    )

    cols = iter(list(melted_df_raw.variable.unique()))
    length_to_split = [12, 12]

    output = [list(islice(cols, elem)) for elem in length_to_split]

    log.debug(output)

    (
        ggplot(melted_df_raw.where(lambda _: _.variable.isin(output[0])).dropna())
        + aes(x="value")
        + geom_histogram()
        + facet_wrap("variable", scales="free")
        + theme(
            subplots_adjust={"hspace": 0.5, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(13, 10),
        )
    ).draw()

    (
        ggplot(melted_df_raw.where(lambda _: _.variable.isin(output[1])).dropna())
        + aes(x="value")
        # had trouble with the binwidth with all of these variables together.
        # Left this as-is since I was able to figure out what I needed from the graphs.
        + geom_histogram(binwidth=10)
        + facet_wrap("variable", scales="free")
        + theme(
            subplots_adjust={"hspace": 0.5, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(13, 10),
        )
    ).draw()

    area_cols = get_area_cols(data)
    area_cols_renamed = filter_suffix(area_cols, data, suffix="log")

    melted_df_log = (
        data.select_dtypes(include=np.number)
        .melt(id_vars=["Id"], var_name="variable", value_name="value")
        .where(lambda _: _.variable.isin(["SalePrice_log"] + area_cols_renamed))
        .dropna()
    )

    (
        # address plotting issues with -infty by replacing with 0
        ggplot(melted_df_log.replace(-np.infty, 0))
        + aes(x="value")
        # had trouble with the binwidth with all of these variables together.
        # Left this as-is since I was able to figure out what I needed from the graphs.
        + geom_histogram()
        + facet_wrap("variable", scales="free")
        + theme(
            subplots_adjust={"hspace": 0.5, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(13, 10),
        )
    ).draw()


def gen_scatter_plots(data):
    melted_df = (
        data.select_dtypes(include=np.number)
        .melt(id_vars=["Id", "SalePrice_log"], var_name="variable", value_name="value")
        .dropna()
    )

    (
        ggplot(melted_df)
        + aes(y="SalePrice_log", x="value")
        + geom_point()
        + geom_smooth(color="red")
        + facet_wrap("variable", scales="free")
        + labs(x="", y="Sale Price (K)", title="Sales Price vs Continuous Variables")
        +
        # adjust x and y axis' so that the scales show up nicely
        # set the figure size to be wider so it takes up the whole screen
        theme(
            subplots_adjust={"hspace": 0.55, "wspace": 0.25},
            axis_text_x=element_text(rotation=45),
            figure_size=(13, 10),
        )
    ).draw()


def gen_cat_plots(data):

    # not a great use of a generator. Mainly used to transform list(dict.keys()) -> list(keys) in a single line
    ord_colnames = list(map(lambda _: next(k for k, v in _.items()), ORDINAL_VARS))
    names = ord_colnames + CATEGORICAL_VARS + ["Id", "SalePrice_log"]

    amenities_ls = [
        "CentralAir",
        "Heating",
        "HeatingQC",
        "Electrical",
        "Fence",
        "Fireplaces",
        "FireplaceQu",
        "PoolQC",
        "Utilities",
    ]
    external_ls = [
        "HouseStyle",
        "ExterCond",
        "ExterQual",
        "Exterior1st",
        "Exterior2nd",
        "Foundation",
        "RoofMatl",
        "RoofStyle",
        "Functional",
        "PavedDrive",
        "Alley",
        "Street",
        "MasVnrType",
        "BldgType",
    ]
    internal_ls = [
        "KitchenAbvGr",
        "KitchenQual",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "FullBath",
        "HalfBath",
    ]
    land_ls = [
        "LandContour",
        "LandSlope",
        "LotConfig",
        "LotShape",
        "MSZoning",
        "Condition1",
        "Condition2",
    ]
    misc_ls = ["Neighborhood", "MiscFeature", "MiscVal"]
    melted_df2 = (
        data[names]
        .melt(id_vars=["Id", "SalePrice_log"], var_name="variable", value_name="value")
        .dropna()
    )

    df_bsmt = melted_df2.where(lambda _: _.variable.str.contains("Bsmt")).dropna()
    df_garage = melted_df2.where(lambda _: _.variable.str.contains("Garage")).dropna()
    df_amenities = melted_df2.where(lambda _: _.variable.isin(amenities_ls)).dropna()
    df_external = melted_df2.where(lambda _: _.variable.isin(external_ls)).dropna()
    df_internal = melted_df2.where(lambda _: _.variable.isin(internal_ls)).dropna()
    df_land = melted_df2.where(lambda _: _.variable.isin(land_ls)).dropna()
    df_misc = melted_df2.where(
        lambda _: ~_.variable.str.contains("Bsmt|Garage", regex=True)
        & ~_.variable.isin(amenities_ls + external_ls + internal_ls + land_ls + misc_ls)
    ).dropna()

    draw_cat_plots(df_bsmt, "Basement")
    draw_cat_plots(df_garage, "Garage")
    draw_cat_plots(df_amenities, "Amenities")
    draw_cat_plots(df_external, "External", "large")
    draw_cat_plots(df_internal, "Internal")
    draw_cat_plots(df_land, "Land")

    draw_cat_plots(df_misc)

    # Misc Feature vs Misc Value.
    (
        ggplot(
            data[["MiscFeature", "MiscVal", "SalePrice"]]
            .where(lambda _: _.MiscFeature != "NA")
            .dropna()
        )
        + aes(x="MiscFeature", y="MiscVal")
        + geom_boxplot()
        + geom_point(color="green")
        + labs(
            x="Feature",
            y="Feature Value (USD)",
            title="Estimated Value of Extra Property Features",
        )
        + theme(figure_size=(6.4, 4.8))
    ).draw()

    (
        ggplot(data)
        + aes(x="factor(YrSold)", y="SalePrice_log")
        + geom_boxplot()
        + labs(x="Year Sold", y="Sale Price (USD) (K)", title="Houses Sold by Year")
        + theme(figure_size=(6.4, 4.8))
    ).draw()


def generate_summary_plots(data):
    gen_logtrans_plots(data)
    gen_scatter_plots(data)
    gen_cat_plots(data)


def get_top_zeroes(data):
    df = data.select_dtypes(include=np.number)
    top_zeroes = (
        # gets the 0-percentage for each column of a dataframe
        ((df == 0).sum() / len(df))
        .sort_values(ascending=False)
        .where(lambda _: _ > 0.10)
        .dropna()
    )
    return top_zeroes


def get_x_y(data):
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
    indicator_cols_clean = list(filter(lambda _: "_log" not in _, indicator_cols_dirty))
    X = data[features + indicator_cols_clean]
    y = data["SalePrice_log"]

    return X, y


def fit_OLS(X, y, K=10):
    model = LinearRegression()
    scores = []
    kfold = KFold(n_splits=K, shuffle=True, random_state=12345)

    for i, (train, test) in enumerate(kfold.split(X, y)):
        print(train)
        print(test)
        model.fit(X.iloc[train], y.iloc[train])
        score = model.score(X.iloc[test], y.iloc[test])
        scores.append(score)

    return scores


def fit_OLS_statsmodels(X, y):
    # Fit the regression model
    sm_model = sm.OLS(y, X)
    sm_ols = sm_model.fit()
    return sm_ols


def fit_RidgeCV(X, y, K=10):
    n_alphas = 10
    alphas = np.logspace(-10, -2, n_alphas)

    # standardize features for Ridge Regression
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    rr = RidgeCV(alphas=alphas, cv=K)
    rr.fit(X_trans, y)
    score = rr.score(X_trans, y)
    return rr, score


def calc_vif(X):
    vif = pd.DataFrame()
    vif["variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False)


def run(plot_summaries=True):
    # get cleansed dataset all ready to go
    data = get_dataset(
        os.path.join(os.getcwd(), "kaggle", "ames-housing-prices", "data", "train.csv")
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
