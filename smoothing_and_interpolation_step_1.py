# system imports
import glob
import os
import pickle

# 3rd party imports
import numpy as np
import math as mh
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from scipy import interpolate
import scipy
import os
from matplotlib.pyplot import cm
import seaborn as sns

# change this to match your system paths, currently set to this folder only
ROOT_FILE_PATH = "."
CURVES_INTERP_MIN = 300
CURVES_INTERP_MAX = 550

# sns_settheme
sns.set_theme()

# function definitions
def load_sites_data(filelist):
    """
    Return a dataframe given a list of files containing data in the form of x y_i
    fileist (list): List of files
    """
    all_sites_df = []
    for filename in filelist:
        column_1_name = "wv_" + os.path.basename(filename).replace(".txt", "")
        column_2_name = os.path.basename(filename).replace(".txt", "")
        dfx = pd.read_csv(
            filename,
            sep=" ",
            names=[column_1_name, column_2_name],
        )
        all_sites_df.append(dfx)

    wavelengths_df_1 = dfx[column_1_name]
    wavelengths_df_1 = wavelengths_df_1.rename("wv")
    combined_df_columns = []
    combined_df_columns.append(wavelengths_df_1)
    for df_i in all_sites_df:
        assert list(df_i[df_i.columns[0]]) == list(wavelengths_df_1)
        combined_df_columns.append(df_i[df_i.columns[1]])
    combined_df = pd.concat(combined_df_columns, axis=1)
    combined_df = combined_df.set_index(combined_df["wv"], drop=True)
    combined_df = combined_df.drop("wv", axis=1)
    return combined_df


def load_canonical_signals():
    df1 = pd.read_csv("trypt.csv")
    df1 = df1.set_index("Wavelength (nm)")

    df2 = pd.read_csv("hlf.txt", sep=" ")
    df2 = df2.set_index("wl")

    df3 = pd.read_csv("e_coli_spectrum.csv")
    df3 = df3.set_index("Wavelength (nm)")
    return df1, df2, df3


def get_smoothed_curves(sites_df, tlf_df, hlf_df, ecoli_df, save_path, sigma=5, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)

    def smooth_data_frame(dfx):
        for column in dfx:
            dfx[column] = scipy.ndimage.gaussian_filter1d(dfx[column], sigma=5, axis=0)
        return dfx

    sites_smoothed = smooth_data_frame(sites_df)
    tlf_smoothed = smooth_data_frame(tlf_df)
    hlf_smoothed = smooth_data_frame(hlf_df)
    ecoli_smoothed = smooth_data_frame(ecoli_df)
    result = {
        "sites": sites_smoothed,
        "hlf": hlf_smoothed,
        "tlf": tlf_smoothed,
        "ecoli": ecoli_smoothed,
    }
    with open(save_path, "wb") as file:
        pickle.dump(result, file)
    return result


def plot_and_save_smoothed_curves(smooth_curves_path):
    with open(smooth_curves_path, "rb") as file:
        curves = pickle.load(file)
    plt.figure(1)
    curves["sites"].plot(ax=plt.gca())
    curves["hlf"].plot(ax=plt.gca())
    curves["tlf"].plot(ax=plt.gca())
    curves["ecoli"].plot(ax=plt.gca())
    plt.savefig("smoothed_curves.png", bbox_inches="tight")
    plt.close()


def get_normalized_curves(smoothed_curves_path, save_path, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)

    with open(smoothed_curves_path, "rb") as file:
        smoothed_curves = pickle.load(file)

    def min_max_norm_df(dfx):
        dfx_scaled = pd.DataFrame.copy(dfx)

        def get_min_max_within_range(dfx):
            y_min, y_max = np.inf, -np.inf
            for x, y in zip(list(dfx.index), list(dfx[dfx.columns[0]])):
                if x < CURVES_INTERP_MIN:
                    continue
                if x > CURVES_INTERP_MAX:
                    continue
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            return y_min, y_max

        for column in dfx_scaled.columns:
            y_min, y_max = get_min_max_within_range(dfx_scaled)
            print(column, y_min, y_max)
            dfx_scaled[column] = (dfx_scaled[column] - y_min) / (y_max - y_min)
            print(np.min(dfx_scaled[column]), np.max(dfx_scaled[column]))
        return dfx_scaled

    def min_norm_df(dfx):
        column_min = {}
        dfx_scaled = pd.DataFrame.copy(dfx)
        for column in dfx_scaled.columns:
            y_min, y_max = get_min_max_within_range(dfx_scaled)
            print(column, y_min, y_max)
            dfx_scaled[column] = (dfx_scaled[column] - y_min) / (y_max - y_min)
            print(np.min(dfx_scaled[column]), np.max(dfx_scaled[column]))
        return dfx_scaled

    normed_signals = {
        "sites": smoothed_curves["sites"],
        "hlf": min_max_norm_df(smoothed_curves["hlf"]),
        "tlf": min_max_norm_df(smoothed_curves["tlf"]),
        "ecoli": min_max_norm_df(smoothed_curves["ecoli"]),
    }
    with open(save_path, "wb") as file:
        pickle.dump(normed_signals, file)
    return normed_signals


def plot_and_save_normed_curves(norm_curves_path):
    with open(norm_curves_path, "rb") as file:
        curves = pickle.load(file)
    plt.figure(1)
    curves["sites"].plot(ax=plt.gca())
    curves["hlf"].plot(ax=plt.gca())
    curves["tlf"].plot(ax=plt.gca())
    curves["ecoli"].plot(ax=plt.gca())
    plt.savefig("normed_curves.png", bbox_inches="tight")
    plt.close()


def get_interped_curves(normed_curves_path, save_path, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)

    with open(normed_curves_path, "rb") as file:
        smoothed_curves = pickle.load(file)

    # declare interp functions and apply to x
    x = np.arange(CURVES_INTERP_MIN, CURVES_INTERP_MAX, 0.5)
    y = []
    for location in smoothed_curves["sites"].columns:
        interp_f = interpolate.interp1d(
            smoothed_curves["sites"].index, smoothed_curves["sites"][location]
        )
        y.append(interp_f(x))
    y_tlf = interpolate.interp1d(
        smoothed_curves["tlf"].index, smoothed_curves["tlf"]["500ppb_500ms"]
    )(x)
    y_hlf = interpolate.interp1d(
        smoothed_curves["hlf"].index, smoothed_curves["hlf"]["hlf"]
    )(x)

    y_ecoli = interpolate.interp1d(
        smoothed_curves["ecoli"].index, smoothed_curves["ecoli"]["ecoli"]
    )(x)

    interped_signals = {
        "x": x,
        "y": y,
        "y_tlf": y_tlf,
        "y_hlf": y_hlf,
        "y_ecoli": y_ecoli,
    }

    with open(save_path, "wb") as file:
        pickle.dump(interped_signals, file)
    return interped_signals


def plot_and_save_interped_curves(interped_curves_path):
    with open(interped_curves_path, "rb") as file:
        curves = pickle.load(file)
    x = curves["x"]
    y = curves["y"]
    y_tlf = curves["y_tlf"]
    y_hlf = curves["y_hlf"]
    y_ecoli = curves["y_ecoli"]
    fig, ax = plt.subplots(figsize=(4, 3))

    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    color = iter(cm.rainbow(np.linspace(0, 1, 17)))

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax.text(CURVES_INTERP_MIN + 50, 0.6, r"TLF", fontsize=12)
    ax.text(CURVES_INTERP_MAX - 80, 0.6, r"HLF", fontsize=12)
    for i in range(len(y)):
        c = next(color)
        plt.plot(x, y[i] / np.max(y[i]), c=c, label=f"{i+1}")
    plt.plot(x, y_tlf, "--", c="black")
    plt.fill_between(x, y_tlf, color="blue", alpha=0.20)
    plt.plot(x, y_hlf, "--", c="black")
    plt.fill_between(x, y_hlf, color="magenta", alpha=0.20)
    plt.plot(x, y_ecoli, "--", c="black")
    plt.fill_between(x, y_ecoli, color="green", alpha=0.20)
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig("interped_curves.png", bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":
    # load data files
    data_file_paths = glob.glob(os.path.join(ROOT_FILE_PATH, "17loc/*.txt"))  # ./17loc
    print("total number of files: ", len(data_file_paths))
    df_signals = load_sites_data(data_file_paths)
    df_tlf, df_hlf, df_ecoli = load_canonical_signals()

    ############### SMOOTHING
    # smooth data files
    save_path_smooth = "smoothed_dfs.pkl"
    smoothed_dfs = get_smoothed_curves(df_signals, df_tlf, df_hlf, df_ecoli, save_path_smooth)
    plot_and_save_smoothed_curves(save_path_smooth)

    # ############### INTERPOLATING
    # normalize and save
    save_path_normed = "normed_dfs.pkl"
    normed_curves = get_normalized_curves(
        save_path_smooth, save_path_normed, overwrite=True
    )
    plot_and_save_normed_curves(save_path_normed)

    # interoplate and save
    save_path_interped = "interped_dfs.pkl"
    interped_curves = get_interped_curves(
        save_path_normed, save_path_interped, overwrite=True
    )
    plot_and_save_interped_curves(save_path_interped)
