from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


# only available after running "smoothing_and_interpolation_step_1.py"
raw_data_path = "interped_dfs.pkl"
WHITEN = True


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
    sns.set_theme()
    sns.set_theme(style="darkgrid")

    # fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plt.figure(1, figsize=(7, 5))

    pca_comp_1_norm = pca.components_[0] / np.max(pca.components_[0])
    pca_comp_2_norm = pca.components_[1] / np.max(pca.components_[1])
    draw_inds = np.arange(0, len(normalized_raw_data["x"]), 4)
    plot_x = normalized_raw_data["x"][draw_inds]
    plt.plot(plot_x, pca_comp_1_norm[draw_inds], "gx", label="PC 1")
    plt.plot(plot_x, pca_comp_2_norm[draw_inds], "g^", label="PC 2")
    plt.plot(
        plot_x,
        normalized_raw_data["y_hlf"][draw_inds],
        "r--",
        linewidth=2,
        label="HLF, r1={:0.2f}, r2={:0.2f}".format(r_comp_0_hlf.statistic, r_comp_1_hlf.statistic),
    )
    plt.plot(
        plot_x,
        normalized_raw_data["y_tlf"][draw_inds],
        "b-.",
        linewidth=2,
        label="TLF, r1={:0.2f}, r2={:0.2f}".format(r_comp_0_tlf.statistic, r_comp_1_tlf.statistic),
    )
    # plt.set_title('hlf component, principal component, 1')
    # plt.plot(
    #     plot_x,
    #     normalized_raw_data["y_tlf"],
    #     "b-.",
    #     label="TLF, r={:0.2f}".format(r_comp_1_tlf.statistic),
    # )
    # plt.plot(
    #     plot_x,
    #     normalized_raw_data["y_hlf"],
    #     "r--",
    #     label="HLF, r={:0.2f}".format(r_comp_1_hlf.statistic),
    # )
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.savefig("pca_comps_and_hlf_tlf.png")
    plt.close()


pca_and_plot_components()
