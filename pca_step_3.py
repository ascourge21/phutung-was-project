from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


# only available after running "smoothing_and_interpolation_step_1.py"
raw_data_path = "interped_dfs.pkl"
WHITEN = True

sns.set_theme()


def pca_and_plot_components():
    with open(raw_data_path, "rb") as file:
        normalized_raw_data = pickle.load(file)

    X = np.array(normalized_raw_data["y"])
    pca = PCA(n_components=10, whiten=WHITEN)
    pca.fit(X)

    print(
        "variance explained by 2 components: {:0.3f}".format(
            np.sum(pca.explained_variance_ratio_[:2])
        )
    )
    print(
        "variance explained by 5 components: {:0.3f}".format(
            np.sum(pca.explained_variance_ratio_[:5])
        )
    )

    # print correlations
    pc0 = -pca.components_[0]
    pc1 = pca.components_[1]
    r_comp_0_tlf = pearsonr(pc0, normalized_raw_data["y_tlf"])
    print("comp 0, tlf corr:{:0.2f}".format(r_comp_0_tlf.statistic))

    r_comp_0_hlf = pearsonr(pc0, normalized_raw_data["y_hlf"])
    print("comp 0, hlf corr:{:0.2f}".format(r_comp_0_hlf.statistic))

    r_comp_1_tlf = pearsonr(pc1, normalized_raw_data["y_tlf"])
    print("comp 1, tlf corr:{:0.2f}".format(r_comp_1_tlf.statistic))

    r_comp_1_hlf = pearsonr(pc1, normalized_raw_data["y_hlf"])
    print("comp 1, hlf corr:{:0.2f}".format(r_comp_1_hlf.statistic))

    # draw components
    fig, ax = plt.subplots(figsize=(4, 3))
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("font", weight="bold")  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    # plt.rc("font", weight="bold")  # controls default text sizes
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pca_comp_1_norm = (pc0 - np.min(pc0)) / (np.max(pc0) - np.min(pc0))
    pca_comp_2_norm = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1))
    draw_inds = np.arange(0, len(normalized_raw_data["x"]), 5)
    plot_x = normalized_raw_data["x"][draw_inds]
    plt.plot(plot_x, pca_comp_1_norm[draw_inds], "gx", label="PC 1")
    plt.plot(plot_x, pca_comp_2_norm[draw_inds], "g^", label="PC 2")
    plt.plot(
        plot_x,
        normalized_raw_data["y_hlf"][draw_inds],
        "r--",
        linewidth=2,
        label="HLF, r1={:0.2f}, r2={:0.2f}".format(
            r_comp_0_hlf.statistic, r_comp_1_hlf.statistic
        ),
    )
    plt.plot(
        plot_x,
        normalized_raw_data["y_tlf"][draw_inds],
        "b-.",
        linewidth=2,
        label="TLF, r1={:0.2f}, r2={:0.2f}".format(
            r_comp_0_tlf.statistic, r_comp_1_tlf.statistic
        ),
    )
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig("pca_comps_and_hlf_tlf.png", bbox_inches="tight", dpi=150)
    plt.close()


def pca_and_plot_components_v2():
    with open(raw_data_path, "rb") as file:
        normalized_raw_data = pickle.load(file)

    X = np.array(normalized_raw_data["y"])
    pca = PCA(n_components=10, whiten=WHITEN)
    pca.fit(X)

    print(
        "variance explained by 2 components: {:0.3f}".format(
            np.sum(pca.explained_variance_ratio_[:2])
        )
    )
    print(
        "variance explained by 5 components: {:0.3f}".format(
            np.sum(pca.explained_variance_ratio_[:5])
        )
    )

    # get the factor term
    # y_tlf_hlf = np.multiply(normalized_raw_data["y_tlf"], normalized_raw_data["y_hlf"])
    # y_tlf_hlf = (y_tlf_hlf - y_tlf_hlf.min()) / (y_tlf_hlf.max() - y_tlf_hlf.min())

    y_tlf_hlf = normalized_raw_data["y_ecoli"]

    # print correlations
    pc0 = -pca.components_[0]
    pc1 = pca.components_[1]
    r_comp_0_tlf = pearsonr(pc0, normalized_raw_data["y_tlf"])
    print(
        "comp 0, tlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_0_tlf.statistic, r_comp_0_tlf.pvalue
        )
    )

    r_comp_0_hlf = pearsonr(pc0, normalized_raw_data["y_hlf"])
    print(
        "comp 0, hlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_0_hlf.statistic, r_comp_0_hlf.pvalue
        )
    )

    r_comp_0_tlf_hlf = pearsonr(pc0, y_tlf_hlf)
    print(
        "comp 0, tlf-hlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_0_tlf_hlf.statistic, r_comp_0_tlf_hlf.pvalue
        )
    )

    r_comp_1_tlf = pearsonr(pc1, normalized_raw_data["y_tlf"])
    print(
        "comp 1, tlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_1_tlf.statistic, r_comp_1_tlf.pvalue
        )
    )

    r_comp_1_hlf = pearsonr(pc1, normalized_raw_data["y_hlf"])
    print(
        "comp 1, hlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_1_hlf.statistic, r_comp_1_hlf.pvalue
        )
    )

    r_comp_1_tlf_hlf = pearsonr(pc1, y_tlf_hlf)
    print(
        "comp 1, tlf-hlf corr:{:0.2f}, p-val:{:0.2f}".format(
            r_comp_1_tlf_hlf.statistic, r_comp_1_tlf_hlf.pvalue
        )
    )

    # draw components
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("font", weight="bold")  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    # plt.rc("font", weight="bold")  # controls default text sizes
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pca_comp_1_norm = (pc0 - np.min(pc0)) / (np.max(pc0) - np.min(pc0))
    pca_comp_2_norm = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1))
    draw_inds = np.arange(0, len(normalized_raw_data["x"]), 5)
    plot_x = normalized_raw_data["x"][draw_inds]

    ax[0].plot(plot_x, pca_comp_1_norm[draw_inds], "kx", linewidth=2, label="PC 1")
    ax[0].plot(
        plot_x,
        normalized_raw_data["y_hlf"][draw_inds],
        "r--",
        linewidth=1.5,
        label="HLF, r={:0.2f}".format(r_comp_0_hlf.statistic),
    )
    ax[0].plot(
        plot_x,
        normalized_raw_data["y_tlf"][draw_inds],
        "b-.",
        linewidth=1.5,
        label="TLF, r={:0.2f}".format(r_comp_0_tlf.statistic),
    )
    ax[0].plot(
        plot_x,
        y_tlf_hlf[draw_inds],
        "g",
        linewidth=1.5,
        label="ecoli, r={:0.2f}".format(r_comp_0_tlf_hlf.statistic),
    )
    ax[0].set_xlabel("Wavelength (nm)", fontsize=14)
    ax[0].set_ylabel("Normalized Amplitude", fontsize=14)
    # ax[0].set_xticks(fontsize=14)
    # ax[0].set_yticks(fontsize=14)
    ax[0].legend(loc="lower right")

    ax[1].plot(plot_x, pca_comp_2_norm[draw_inds], "kx", linewidth=2, label="PC 2")
    ax[1].plot(
        plot_x,
        normalized_raw_data["y_hlf"][draw_inds],
        "r--",
        linewidth=1.5,
        label="HLF, r={:0.2f}".format(r_comp_1_hlf.statistic),
    )
    ax[1].plot(
        plot_x,
        normalized_raw_data["y_tlf"][draw_inds],
        "b-.",
        linewidth=1.5,
        label="TLF, r={:0.2f}".format(r_comp_1_tlf.statistic),
    )
    ax[1].plot(
        plot_x,
        y_tlf_hlf[draw_inds],
        "g",
        linewidth=1.5,
        label="ecoli, r={:0.2f}".format(r_comp_1_tlf_hlf.statistic),
    )
    ax[1].set_xlabel("Wavelength (nm)", fontsize=14)
    # ax[1].set_xticks(fontsize=14)
    # ax[1].set_yticks(fontsize=14)
    ax[1].legend(loc="lower right")
    plt.savefig("pca_comps_and_hlf_tlf_and_combo.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    pca_and_plot_components_v2()
