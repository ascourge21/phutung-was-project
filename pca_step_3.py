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
    r_comp_0_tlf = pearsonr(pca.components_[0], normalized_raw_data["y_tlf"])
    print("comp 0, tlf corr:{:0.2f}".format(r_comp_0_tlf.statistic))

    r_comp_0_hlf = pearsonr(pca.components_[0], normalized_raw_data["y_hlf"])
    print("comp 0, hlf corr:{:0.2f}".format(r_comp_0_hlf.statistic))

    r_comp_1_tlf = pearsonr(pca.components_[1], normalized_raw_data["y_tlf"])
    print("comp 1, tlf corr:{:0.2f}".format(r_comp_1_tlf.statistic))

    r_comp_1_hlf = pearsonr(pca.components_[1], normalized_raw_data["y_hlf"])
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

    pca_comp_1_norm = (pca.components_[0] - np.min(pca.components_[0])) / (
        np.max(pca.components_[0]) - np.min(pca.components_[0])
    )
    pca_comp_2_norm = (pca.components_[1] - np.min(pca.components_[1])) / (
        np.max(pca.components_[1]) - np.min(pca.components_[0])
    )
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


if __name__ == "__main__":
    pca_and_plot_components()
