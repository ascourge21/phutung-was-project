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
import os
from matplotlib.pyplot import cm

# change this to match your system paths, currently set to this folder only
ROOT_FILE_PATH = "."
CURVES_INTERP_MIN = 300
CURVES_INTERP_MAX = 550


def load_sites_data(filelist):
    """
    Return a dataframe given a list of files containing data in the form of x y_i
    fileist (list): List of files
    """
    df = []
    for filename in filelist:
        dfx = pd.read_csv(filename, sep=" ", names=["wv", filename[:-4]])
        dfx = dfx.set_index(dfx["wv"], drop=True)
        dfx = dfx.drop(dfx.columns[0], axis=1)
        df.append(dfx)
    return df


def load_canonical_signals():
    df1 = pd.read_csv("trypt.csv")
    df1 = df1.set_index("Wavelength (nm)")

    df2 = pd.read_csv("hlf.txt", sep=" ")
    df2 = df2.set_index("wl")
    return df1, df2


def get_smoothed_curves(sites_df, tlf_df, hlf_df, save_path, sigma=5, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)

    def smooth_data_frame(dfx):
        for column in dfx:
            smoothed_vals = np.zeros(dfx[column].shape)
            i = 0
            for x_position in dfx.index:
                kernel = np.exp(-((dfx.index - x_position) ** 2) / (2 * sigma**2))
                kernel = kernel / sum(kernel)
                smoothed_vals[i] = sum(dfx[column] * kernel)
                i += 1
            dfx[column] = smoothed_vals
        return dfx

    sites_smoothed = smooth_data_frame(sites_df)
    tlf_smoothed = smooth_data_frame(tlf_df)
    hlf_smoothed = smooth_data_frame(hlf_df)
    result = {
        "sites": sites_smoothed,
        "hlf": hlf_smoothed,
        "tlf": tlf_smoothed,
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
    plt.savefig("smoothed_curves.png")
    plt.close()


def get_normalized_curves(smoothed_curves_path, save_path, overwrite=False):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)

    with open(smoothed_curves_path, "rb") as file:
        smoothed_curves = pickle.load(file)

    def norm_df(dfx):
        scaler = preprocessing.MinMaxScaler()
        names = dfx.columns
        d = scaler.fit_transform(dfx)
        scaled_df = pd.DataFrame(d, columns=names)
        scaled_df = scaled_df.set_index(dfx.index)
        return scaled_df

    normed_signals = {
        "sites": norm_df(smoothed_curves["sites"]),
        "hlf": norm_df(smoothed_curves["hlf"]),
        "tlf": norm_df(smoothed_curves["tlf"]),
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
    plt.savefig("normed_curves.png")
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

    interped_signals = {
        "x": x,
        "y": y,
        "y_tlf": y_tlf,
        "y_hlf": y_hlf,
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
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

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
    ax = plt.gca()
    ax.text(350, 0.6, r"TLF", fontsize=12)
    ax.text(470, 0.6, r"HLF", fontsize=12)
    for i in range(len(y)):
        c = next(color)
        plt.plot(x, y[i], c=c, label=f"{i+1}")
    plt.plot(x, y_tlf, "--", c="black")
    plt.fill_between(x, y_tlf, color="blue", alpha=0.20)
    plt.plot(x, y_hlf, "--", c="black")
    plt.fill_between(x, y_hlf, color="magenta", alpha=0.20)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.savefig("interped_curves.png", bbox_inches="tight", dpi=150)
    plt.close()


# load data files
data_file_paths = glob.glob(os.path.join(ROOT_FILE_PATH, "17loc/*.txt"))  # ./17loc
print("total number of files: ", len(data_file_paths))
df_signals = pd.concat(load_sites_data(data_file_paths), axis=1)
df_tlf, df_hlf = load_canonical_signals()


############### SMOOTHING
# smooth data files
save_path_smooth = "smoothed_dfs.pkl"
smoothed_dfs = get_smoothed_curves(df_signals, df_tlf, df_hlf, save_path_smooth)
plot_and_save_smoothed_curves(save_path_smooth)


############### INTERPOLATING
# normalize and save
save_path_normed = "normed_dfs.pkl"
normed_curves = get_normalized_curves(save_path_smooth, save_path_normed)
plot_and_save_normed_curves(save_path_normed)

# interoplate and save
save_path_interped = "interped_dfs.pkl"
interped_curves = get_interped_curves(save_path_normed, save_path_interped)
plot_and_save_interped_curves(save_path_interped)
