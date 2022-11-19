# system imports
import glob
import os
import pickle


# import statements
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sns_settheme
sns.set_theme()

# change this to match your system paths, currently set to this folder only
ROOT_FILE_PATH = "."


def perform_regression(
    normed_save_path, interped_save_path, save_path, overwrite=False
):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)
    with open(normed_save_path, "rb") as file:
        curve = pickle.load(file)
    with open(interped_save_path, "rb") as file:
        curves = pickle.load(file)
    locations = curve["sites"].columns
    for p in locations:
        print(p)
    x = curves["x"]
    y = curves["y"]
    y_tlf = curves["y_tlf"]
    y_hlf = curves["y_hlf"]

    y_ones = np.ones(y[2].shape)

    # Desired coefficient matrix for LSF. First columns is of 1
    coeff_matrix = np.array([y_ones, y_tlf, y_hlf]).transpose()

    k = []
    for i in range(len(y)):
        k.append(np.linalg.lstsq(coeff_matrix, y[i], rcond="warn"))

    c_array = np.array([])
    for x in k:
        c_array = np.append(c_array, x[0], axis=0)
        print(x[0])

    c_array = c_array.reshape(len(y), coeff_matrix.shape[1])
    c_df = pd.DataFrame(data=c_array, columns=["c", "TLF", "HLF"])
    c_df["location"] = locations

    # Comparision with the CFU/ plate count
    df_cfu = pd.read_excel("Bagmati Water Details.xlsx", "After Averaging")
    df_cfu = df_cfu[["Sample ID", "CFU/100ml"]]
    df_cfu = df_cfu.set_index(df_cfu["Sample ID"], drop=True).drop("Sample ID", axis=1)

    cfu_list = []
    for location in locations:
        cfu_list.append(df_cfu.loc[location].values[0])

    # final data frame
    fdf = c_df.copy()
    fdf = fdf.set_index("location")

    cfulistnp = np.array(cfu_list)
    fdf["cfu"] = cfu_list
    with open(save_path, "wb") as file:
        pickle.dump(fdf, file)
    return fdf


def plot_regression(fdf):
    # with open(finaldf_path, "rb") as file:
    #  fdf = pickle.load(file)
    # following 10 lines of code control the font size of matplotlib plots explicitly. may affect other functions
    # that follow this

    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    ax[0].scatter(fdf["TLF"], fdf["cfu"])
    z1 = np.polyfit(fdf["TLF"], fdf["cfu"], 1)
    p1 = np.poly1d(z1)
    ax[0].plot(fdf["TLF"], p1(fdf["TLF"]), "r")
    ax[0].set_xlabel(r"$a_T$")
    ax[0].set_ylabel(r"CFU")
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["TLF"], fdf["cfu"]
    )
    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )
    ax[0].annotate(r"$r_p$:{:.2f}".format(r_value), (1.18, 1.25 * 1e6))
    ax[0].annotate("p-value:{:.2f}".format(p_value), (1.18, 1.15 * 1e6))
    ax[1].scatter(fdf["HLF"], fdf["cfu"])
    z2 = np.polyfit(fdf["HLF"], fdf["cfu"], 1)
    p2 = np.poly1d(z2)
    ax[1].plot(fdf["HLF"], p2(fdf["HLF"]), "r")
    ax[1].set_xlabel(r"$a_H$")
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["HLF"], fdf["cfu"]
    )
    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )
    ax[1].annotate(r"$r_p$: {:.2f}".format(r_value), (0.78, 1.25 * 1e6))
    ax[1].annotate("p-value: {:.2f}".format(p_value), (0.78, 1.15 * 1e6))

    plt.savefig("sCFUvsTLFnHLF.png", dpi=150, bbox_inches="tight")


# perform regression and add CFU to the final d
save_path_normed = "normed_dfs.pkl"
save_path_interped = "interped_dfs.pkl"
save_path_regressed = "regressed_df.pkl"
final_dataframe = perform_regression(
    save_path_normed, save_path_interped, save_path_regressed
)

plot_regression(final_dataframe)
