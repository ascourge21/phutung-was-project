# system imports
import glob
import os
import pickle


# import statements
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sns_settheme
sns.set_theme()

PRESENCE_CUTOFF = 340000


def perform_regression(
    normed_save_path, interped_save_path, save_path, overwrite=False
):
    if not overwrite and os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return pickle.load(file)
    with open(normed_save_path, "rb") as file:
        normed_curves = pickle.load(file)
    with open(interped_save_path, "rb") as file:
        interped_curves = pickle.load(file)
    locations = normed_curves["sites"].columns

    x = interped_curves["x"]
    y = interped_curves["y"]
    y_tlf = interped_curves["y_tlf"]
    y_hlf = interped_curves["y_hlf"]

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
    c_df = pd.DataFrame(data=c_array, columns=["c", "a_TLF", "a_HLF"])
    c_df["location"] = locations

    # Comparision with the CFU/ plate count
    df_cfu = pd.read_excel("Bagmati Water Details.xlsx", "After Averaging")
    df_cfu = df_cfu[["Sample ID", "CFU/100ml", "TLF (ppb)", "Site number"]]
    df_cfu = df_cfu.set_index(df_cfu["Sample ID"], drop=True).drop("Sample ID", axis=1)

    cfu_list, tlf_measured_list, site_numbers = [], [], []
    for location in locations:
        row_vals = df_cfu.loc[location].values
        cfu_list.append(row_vals[0])
        tlf_measured_list.append(row_vals[1])
        site_numbers.append(int(row_vals[2]))

    # final data frame
    fdf = c_df.copy()
    fdf = fdf.set_index("location")

    cfulistnp = np.array(cfu_list)
    fdf["cfu"] = cfu_list
    fdf["tlf_independent"] = tlf_measured_list
    fdf["site_number"] = site_numbers
    with open(save_path, "wb") as file:
        pickle.dump(fdf, file)
    return fdf


def plot_regression(fdf, logscale=True):
    # with open(finaldf_path, "rb") as file:
    #  fdf = pickle.load(file)
    # following 10 lines of code control the font size of matplotlib plots explicitly. may affect other functions
    # that follow this

    im_save_path = "sCFUvsTLFnHLF.png"
    if logscale:
        fdf["cfu"] = fdf["cfu"].map(np.log10)
        im_save_path = "sCFUvsTLFnHLF_log.png"

    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("font", weight="bold")
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax[0].scatter(fdf["a_TLF"], fdf["cfu"])
    z1 = np.polyfit(fdf["a_TLF"], fdf["cfu"], 1)
    p1 = np.poly1d(z1)
    ax[0].plot(fdf["a_TLF"], p1(fdf["a_TLF"]), "r")
    ax[0].set_xlabel(r"$a_T$", fontsize=16)
    if logscale:
        ax[0].set_ylabel(r"log(CFU/100ml)", fontsize=16)
    else:
        ax[0].set_ylabel(r"CFU", fontsize=16)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["a_TLF"], fdf["cfu"]
    )
    pearson_r = stats.pearsonr(fdf["a_TLF"], fdf["cfu"])
    spearman_r = stats.spearmanr(fdf["a_TLF"], fdf["cfu"])
    print(pearson_r, spearman_r)
    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )

    ax[0].tick_params(axis="both", which="major", labelsize=14)
    ax[0].tick_params(axis="both", which="minor", labelsize=14)

    if logscale:
        ax[0].annotate(r"$r_p$:{:.2f}".format(r_value), (1.10, 6.1))
        ax[0].annotate("p-value:{:.2f}".format(p_value), (1.10, 5.95))
    else:
        ax[0].annotate(r"$r_p$:{:.2f}".format(r_value), (1.10, 1.25 * 1e6))
        ax[0].annotate("p-value:{:.2f}".format(p_value), (1.10, 1.15 * 1e6))

    ax[1].scatter(fdf["a_HLF"], fdf["cfu"])
    z2 = np.polyfit(fdf["a_HLF"], fdf["cfu"], 1)
    p2 = np.poly1d(z2)
    ax[1].plot(fdf["a_HLF"], p2(fdf["a_HLF"]), "r")
    ax[1].set_xlabel(r"$a_H$", fontsize=16)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["a_HLF"], fdf["cfu"]
    )
    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )

    ax[1].tick_params(axis="both", which="major", labelsize=14)
    ax[1].tick_params(axis="both", which="minor", labelsize=14)

    if logscale:
        ax[1].annotate(r"$r_p$: {:.2f}".format(r_value), (0.88, 6.1))
        ax[1].annotate("p-value: {:.2f}".format(p_value), (0.88, 5.95))
    else:
        ax[1].annotate(r"$r_p$: {:.2f}".format(r_value), (0.78, 1.25 * 1e6))
        ax[1].annotate("p-value: {:.2f}".format(p_value), (0.78, 1.15 * 1e6))

    plt.savefig(im_save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression_v2(fdf, im_save_path, x_log=False, y_log=False):
    x_label = "cfu"
    y_label = "amplitude"
    fdf = pd.DataFrame.copy(fdf)
    if x_log:
        fdf["cfu"] = fdf["cfu"].map(np.log10)
        x_label = "log(cfu)"
    if y_log:
        fdf["a_TLF"] = fdf["a_TLF"].map(np.log10)
        fdf["a_HLF"] = fdf["a_HLF"].map(np.log10)
        y_label = "log(amplitude)"

    title = ""
    if x_log and y_log:
        title = "log-log-plot"
    elif x_log and ~y_log:
        title = "log-linear-plot"
    elif ~x_log and y_log:
        title = "linear-log-plot"
    elif ~x_log and ~y_log:
        title = "linear-linear-plot"
    else:
        raise ValueError("need to be one of the above 4")

    fig, ax = plt.subplots(figsize=(5, 3.8))
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("font", weight="bold")
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["cfu"],
        fdf["a_TLF"],
    )
    pearson_r = stats.pearsonr(fdf["cfu"], fdf["a_TLF"])
    spearman_r = stats.spearmanr(fdf["cfu"], fdf["a_TLF"])
    ax.scatter(
        fdf["cfu"],
        fdf["a_TLF"],
        c="green",
        label="tlf, r-pea: {:0.2f}, r-spe: {:0.2f}".format(
            pearson_r.statistic, spearman_r.correlation
        ),
    )
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    z1 = np.polyfit(fdf["cfu"], fdf["a_TLF"], 1)
    p1 = np.poly1d(z1)
    ax.plot(fdf["cfu"], p1(fdf["cfu"]), "g")

    print("tlf", pearson_r, spearman_r)
    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )

    # ax = ax.twinx()
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        fdf["cfu"], fdf["a_HLF"]
    )
    pearson_r = stats.pearsonr(fdf["cfu"], fdf["a_HLF"])
    spearman_r = stats.spearmanr(fdf["cfu"], fdf["a_HLF"])
    ax.scatter(
        fdf["cfu"],
        fdf["a_HLF"],
        c="red",
        label="hlf, r-pea: {:0.2f}, r-spe: {:0.2f}".format(
            pearson_r.statistic, spearman_r.correlation
        ),
    )
    # ax.set_ylabel(r"$a_{HLF}$", fontsize=16)

    z2 = np.polyfit(fdf["cfu"], fdf["a_HLF"], 1)
    p2 = np.poly1d(z2)
    ax.plot(fdf["cfu"], p2(fdf["cfu"]), "r")
    print("hlf", pearson_r, spearman_r)

    print(
        "slope, intercept, r_value, p_value, std_err ",
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    )

    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(im_save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_and_save_roc_curves(score, fpr, tpr, title):
    lw = 2
    plt.figure(2)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % score)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    with open("detection_roc_auc.png", "wb") as file:
        plt.savefig(file)
    plt.close()


def get_auc_sensitivity_specificity(cfu, y_tlf, present_cutoff=PRESENCE_CUTOFF):
    def get_y_true_y_pred(cfu_y, tlf_y, cfu_cutoff, tlf_cutoff, for_auc=False):
        y_true, y_pred = [], []
        for cfu_i, y_tlf_i in zip(cfu, y_tlf):
            y_true.append(cfu_i > cfu_cutoff)
            if for_auc:
                y_pred.append(y_tlf_i)
            else:
                y_pred.append(y_tlf_i > tlf_cutoff)
        return y_true, y_pred

    y_true, y_pred = get_y_true_y_pred(cfu, y_tlf, present_cutoff, None, for_auc=True)
    auc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    print("AUC score at cutoff: {}, is: {:0.3f}".format(present_cutoff, auc_score))
    plot_and_save_roc_curves(
        auc_score,
        fpr,
        tpr,
        "ROC-AUC for detection at CFU cutoff of:  %d" % present_cutoff,
    )

    best_cutoff, best_f1 = None, -1
    for tlf_cutoff in np.linspace(np.min(y_tlf), np.max(y_tlf), 20):
        y_true, y_pred = get_y_true_y_pred(
            cfu, y_tlf, present_cutoff, tlf_cutoff, for_auc=False
        )
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_cutoff = tlf_cutoff
            best_f1 = f1

    y_true, y_pred = get_y_true_y_pred(
        cfu, y_tlf, present_cutoff, best_cutoff, for_auc=False
    )

    print(
        "cfu cutoff: {}, best tlf cutoff: {:0.3f}".format(present_cutoff, best_cutoff)
    )
    print("F1-score: {:0.3f}".format(f1_score(y_true, y_pred)))
    print("Sensitivity: {:0.3f}".format(recall_score(y_true, y_pred)))
    print("Specificity: {:0.3f}".format(f1_score(~np.array(y_true), ~np.array(y_pred))))
    print("Precision: {:0.3f}".format(precision_score(y_true, y_pred)))


if __name__ == "__main__":
    ############### REGRESSION
    # perform regression and add CFU to the final
    save_path_normed = "normed_dfs.pkl"
    save_path_interped = "interped_dfs.pkl"
    save_path_regressed = "regressed_df.pkl"
    final_dataframe = perform_regression(
        save_path_normed, save_path_interped, save_path_regressed, overwrite=True
    )
    plot_regression_v2(
        final_dataframe, "coeff_and_cfu_linear_linear", x_log=False, y_log=False
    )
    plot_regression_v2(final_dataframe, "coeff_and_cfu_log_linear", x_log=True, y_log=False)
    plot_regression_v2(final_dataframe, "coeff_and_cfu_linear_log", x_log=False, y_log=True)
    plot_regression_v2(final_dataframe, "coeff_and_cfu_log_log", x_log=True, y_log=True)


    ############### CLASSIFICATION / DETECTION
    # check AUC score for detection (although the cutoff is arbitary now (median))
    # get_auc_sensitivity_specificity(final_dataframe["cfu"], final_dataframe["TLF"])
