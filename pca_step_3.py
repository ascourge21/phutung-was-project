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

    print(["{:0.1f}".format(var * 100) for var in pca.explained_variance_ratio_])
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
        label="HLF, r=0.98",
    )
    axes[0].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_tlf"],
        "b-.",
        label="TLF, r=-0.55",
    )
    # axes[0].set_title('hlf component, principal component, 1')
    axes[0].legend()

    pca_comp_2_norm = pca.components_[1] / np.max(pca.components_[1])
    axes[1].plot(normalized_raw_data["x"], pca_comp_2_norm, "g-", label="PC 2")
    axes[1].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_tlf"],
        "b-.",
        label="TLF, r=-0.065",
    )
    axes[1].plot(
        normalized_raw_data["x"],
        normalized_raw_data["y_hlf"],
        "r--",
        label="HLF, r=-0.32",
    )
    axes[1].legend()
    plt.savefig("pca_comps_and_hlf_tlf.png")


pca_and_plot_components()
