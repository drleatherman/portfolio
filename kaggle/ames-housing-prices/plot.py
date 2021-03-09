"""
A module containing functions used for generating plots for the Ames Housing Dataset. Some of these are more generic but for much, fields within the dataset are baked in and would need to be slightly refactored if used in another project.

"""

from itertools import islice
from plotnine import *  # plotting
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.pyplot import figure

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


def draw_cat_plots(
    df, title="Categorical", size="regular", y="SalePrice_log", x="value"
):
    """
    Draw plot for categorical data. In this case, it's boxplots. This is mostly used as a utility for keeping plotting code DRY.

    :param df: pd.DataFrame - A DataFrame to plot. Has the following columns: SalePrice, variable, value
    :param title: str       - A phrase to insert into the plot's title.
    :param size: str        - The size of the png that should be produced. Useful in cases where we need to control real-estate by plot. options: regular, large.
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
        + labs(x="Log Sale Price (USD)", y="Count",
               title="Adjusted Sale Price")
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

    (
        ggplot(melted_df_raw.where(
            lambda _: _.variable.isin(output[0])).dropna())
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
        ggplot(melted_df_raw.where(
            lambda _: _.variable.isin(output[1])).dropna())
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
        + labs(x="", y="Sale Price (K)",
               title="Sales Price vs Continuous Variables")
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
    ord_colnames = list(
        map(lambda _: next(k for k, v in _.items()), ORDINAL_VARS))
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

    df_bsmt = melted_df2.where(
        lambda _: _.variable.str.contains("Bsmt")).dropna()
    df_garage = melted_df2.where(
        lambda _: _.variable.str.contains("Garage")).dropna()
    df_amenities = melted_df2.where(
        lambda _: _.variable.isin(amenities_ls)).dropna()
    df_external = melted_df2.where(
        lambda _: _.variable.isin(external_ls)).dropna()
    df_internal = melted_df2.where(
        lambda _: _.variable.isin(internal_ls)).dropna()
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
        + labs(x="Year Sold", y="Sale Price (USD) (K)",
               title="Houses Sold by Year")
        + theme(figure_size=(6.4, 4.8))
    ).draw()


def residual_plots(regression_res):
    """
    Generate the following residual plots:
    1. Fitted vs Residuals

    :regression_rs: statsmodels.regression.linear_model.RegressionResults - A fitted model using the statsmodels library
    """

    model_df = model_summary(regression_res)
    # determine outliers. This is not the best way to do it but it's clear from the residuals that
    # the outliers in this case match this logic.
    model_df["index_label"] = model_df.where(lambda _: abs(_.residuals) > 1)[
        "index"
    ].replace(np.nan, "")
    (
        ggplot(model_df)
        + aes(x="fitted", y="residuals")
        + geom_smooth(color="red", method="loess")
        + geom_hline(yintercept=0, linetype="dotted")
        + geom_point()
        # highlight outliers with text labels
        + geom_text(
            aes(label="index_label", size=10), show_legend=False, ha="left", va="top"
        )
        + labs(x="Fitted Values", y="Residuals", title="Fitted vs. Residuals")
    ).draw()


def generate_summary_plots(data):
    gen_logtrans_plots(data)
    gen_scatter_plots(data)
    gen_cat_plots(data)


def plot_ridge_path(X, y):
    """
    Taken from https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html
    """

    figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    # #############################################################################
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(-10, 6, n_alphas)
    # alphas = np.logspace(6, -3, n_alphas)
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a)
        ridge.fit(X_trans, y)
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [scores], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)
