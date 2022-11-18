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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    pca_comp_1_norm = pca.components_[0] / np.max(pca.components_[0])
    axes[0].plot(normalized_raw_data["x"], pca_comp_1_norm, "g-", label="PC 1")
    axes[0].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_hlf"],
        "r--",
        label="HLF, r={:0.2f}".format(r_comp_0_hlf.statistic),
    )
    axes[0].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_tlf"],
        "b-.",
        label="TLF, r={:0.2f}".format(r_comp_0_tlf.statistic),
    )
    # axes[0].set_title('hlf component, principal component, 1')
    axes[0].legend()

    pca_comp_2_norm = pca.components_[1] / np.max(pca.components_[1])
    axes[1].plot(normalized_raw_data["x"], pca_comp_2_norm, "g-", label="PC 2")
    axes[1].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_tlf"],
        "b-.",
        label="TLF, r={:0.2f}".format(r_comp_1_tlf.statistic),
    )
    axes[1].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_hlf"],
        "r--",
        label="HLF, r={:0.2f}".format(r_comp_1_hlf.statistic),
    )
    axes[1].legend()
    plt.savefig("pca_comps_and_hlf_tlf.png")
    plt.close()


pca_and_plot_components()
