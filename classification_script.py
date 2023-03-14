# system imports
import os
import pickle

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import seaborn as sns


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# try different model types - choose the best one
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    confusion_matrix,
)

TOTAL_WV_POINTS = 1520

CURVES_INTERP_MIN = 310
CURVES_INTERP_MAX = 550

N_GRID_SEARCHES = 1


# load cfu - gold standard / ground truth
def load_cfu_by_location():
    cfu_table = pd.read_csv("Compilation_All_RM_NormAUC_ScaleTLF.txt", delimiter="\t")
    location_to_cfu_dict = {}
    for _, row in cfu_table.iterrows():
        location = os.path.join(
            "Fluorescence Spectra NormAUC_ScaleTLF",
            row["DirL1"],
            row["DirL2"],
            row["Fname"].replace(".txt", ""),
        )
        location_to_cfu_dict[location] = row["CFU/100ml"]
    return location_to_cfu_dict


# load data


def get_list_of_files_to_use_for_signals():
    df = pd.read_csv("Compilation_All_RM_NormAUC_ScaleTLF.txt", delimiter="\t")
    df = df[df["CFU/100ml"].notna()]
    df = df[df["Eqv TLF (PMT)"].notna()]

    file_paths = []
    for _, row in df.iterrows():
        file_path = os.path.join(
            "Fluorescence Spectra NormAUC_ScaleTLF",
            row["DirL1"],
            row["DirL2"],
            row["Fname"],
        )
        assert os.path.exists(file_path)
        file_paths.append(file_path)
    return file_paths


# function definitions
def load_sites_data(filelist):
    """
    Return a dataframe given a list of files containing data in the form of x y_i
    fileist (list): List of files
    """
    all_sites_df = []
    for filename in filelist:
        column_1_name = "wv_" + os.path.basename(filename).replace(".txt", "")
        column_2_name = filename.replace(".txt", "")
        # integrationTime = float(filename.rsplit('_')[-1].rstrip('s.txt'))
        dfx = pd.read_csv(
            filename,
            sep="\t",
            names=[column_1_name, column_2_name],
        )
        assert len(dfx) == TOTAL_WV_POINTS, len(dfx)
        all_sites_df.append(dfx)

    wavelengths_df_1 = dfx[column_1_name]
    wavelengths_df_1 = wavelengths_df_1.rename("wv")
    combined_df_columns = []
    combined_df_columns.append(wavelengths_df_1)
    for df_i in all_sites_df:
        assert list(df_i[df_i.columns[0]]) == list(wavelengths_df_1), (
            list(df_i[df_i.columns[0]]),
            list(wavelengths_df_1),
        )
        combined_df_columns.append(df_i[df_i.columns[1]])
    combined_df = pd.concat(combined_df_columns, axis=1)
    combined_df = combined_df.set_index(combined_df["wv"], drop=True)
    combined_df = combined_df.drop("wv", axis=1)
    return combined_df


def load_canonical_signals(tlfFile, hlfFile):
    tlfIntTime = float(tlfFile.rsplit("_")[-1].rstrip("s.txt"))
    hlfIntTime = float(hlfFile.rsplit("_")[-1].rstrip("s.txt"))

    df1 = pd.read_csv(tlfFile, delimiter="\t", header=None, names=("wv", "amplitude"))
    df2 = pd.read_csv(hlfFile, delimiter="\t", header=None, names=("wv", "amplitude"))

    df1["amplitude"] /= tlfIntTime
    df2["amplitude"] /= hlfIntTime

    df1 = df1.set_index("wv")
    df2 = df2.set_index("wv")
    return df1, df2


def get_interped_curves(other_signals, tlf_signal, hlf_signal):
    # declare interp functions and apply to x
    x = np.arange(CURVES_INTERP_MIN, CURVES_INTERP_MAX, 0.5)
    y = []
    locations = []
    for i, location in enumerate(other_signals.columns):
        interp_f = interpolate.interp1d(other_signals.index, other_signals[location])
        y.append(interp_f(x))
        locations.append(location)

    y_tlf = interpolate.interp1d(tlf_signal.index, tlf_signal["amplitude"])(x)
    y_hlf = interpolate.interp1d(hlf_signal.index, hlf_signal["amplitude"])(x)

    interped_signals = {
        "x": x,
        "y": y,
        "y_tlf": y_tlf,
        "y_hlf": y_hlf,
        "locations": locations,
    }
    return interped_signals


AUC_CUTOFFS = {
    "tlf": {
        "min": 341,
        "max": 371,
    },
    "hlf": {
        "min": 400,
        "max": 500,
    },
}


def get_decomp_coeffs(y_signal, y_tlf, y_hlf):
    y_ones = np.ones(y_signal.shape)

    # Desired coefficient matrix for LSF. First columns is of 1
    y_tlf_hlf = np.multiply(y_tlf, y_hlf)
    y_tlf_hlf = (y_tlf_hlf - y_tlf_hlf.min()) / (y_tlf_hlf.max() - y_tlf_hlf.min())

    coeff_matrix = np.array([y_ones, y_tlf, y_hlf, y_tlf_hlf]).transpose()
    return np.linalg.lstsq(coeff_matrix, y_signal, rcond=None)[0]


def get_corr_coeffs(y_signal, y_tlf, y_hlf):
    tlf_c = np.corrcoef(y_signal, y_tlf)[0][1]
    hlf_c = np.corrcoef(y_signal, y_hlf)[0][1]

    # Desired coefficient matrix for LSF. First columns is of 1
    y_tlf_hlf = np.multiply(y_tlf, y_hlf)
    y_tlf_hlf = (y_tlf_hlf - y_tlf_hlf.min()) / (y_tlf_hlf.max() - y_tlf_hlf.min())

    tlf_hlf_c = np.corrcoef(y_signal, y_tlf_hlf)[0][1]
    return tlf_c, hlf_c, tlf_hlf_c


def area_under_curve(y_signal, x_wavelength, x_min, x_max):
    y_sum = 0
    for x, y in zip(x_wavelength, y_signal):
        if x >= x_min and x < x_max:
            y_sum += y
    return y_sum


# compute features for classification

CFU_ZERO_VAL = 0.1
TLF_SIGNAL_ONLY = False


def get_features(y_signal, y_tlf, y_hlf, x_wavelength, tlf_signal=False):
    tlf_c, hlf_c, tlf_hlf_c = get_corr_coeffs(y_signal, y_tlf, y_hlf)
    decomp_coeffs = get_decomp_coeffs(y_signal, y_tlf, y_hlf)
    auc_tlf = area_under_curve(
        y_signal,
        x_wavelength,
        AUC_CUTOFFS["tlf"]["min"],
        AUC_CUTOFFS["tlf"]["max"],
    )
    auc_hlf = area_under_curve(
        y_signal,
        x_wavelength,
        AUC_CUTOFFS["hlf"]["min"],
        AUC_CUTOFFS["hlf"]["max"],
    )
    if tlf_signal:
        feature = [
            auc_tlf,
        ]
    else:
        feature = [
            decomp_coeffs[1],
            decomp_coeffs[2],
            decomp_coeffs[3],
            tlf_c,
            hlf_c,
            tlf_hlf_c,
            auc_tlf,
            auc_hlf,
            np.max(y_signal),
        ]
    return feature


def get_classification_data(interped_curves):
    y_cfu = []
    X = []
    for i, loc in enumerate(interped_curves["locations"]):
        y_signal = interped_curves["y"][i]
        y_tlf = interped_curves["y_tlf"]
        y_hlf = interped_curves["y_hlf"]

        signal_feature = get_features(
            y_signal, y_tlf, y_hlf, interped_curves["x"], tlf_signal=TLF_SIGNAL_ONLY
        )
        X.append(signal_feature)
        y_cfu.append(location_to_cfu_dict[loc] != CFU_ZERO_VAL)

    y_cfu = np.array(y_cfu).astype("uint8")
    X = np.array(X)
    return X, y_cfu


def my_custom_loss_func(y_t, y_p):
    recall = recall_score(y_t, y_p)
    specificity = recall_score(
        ~np.array(y_t, dtype="bool"),
        ~np.array(y_p, dtype="bool"),
    )
    return recall * specificity


def get_best_param_for_fit(X_train, X_test, y_train, y_test):
    score_function = make_scorer(my_custom_loss_func)

    best_model = None
    best_score = None
    best_params = None

    for i in range(N_GRID_SEARCHES):
        param_grid = {
            "bootstrap": [True, False],
            "max_depth": range(2, 8, 2),
            "max_features": ["sqrt", "log2"],
            "n_estimators": range(5, 20, 3),
        }

        rfbase = RandomForestClassifier()

        rf_gridsearch = GridSearchCV(
            estimator=rfbase,
            param_grid=param_grid,
            scoring=score_function,
            cv=4,
            n_jobs=4,
            verbose=1,
            return_train_score=True,
        )

        rf_gridsearch.fit(X_train, y_train)

        if best_model is None:
            best_model = rf_gridsearch.best_estimator_
            best_score = rf_gridsearch.best_score_
            best_params = rf_gridsearch.best_params_
        elif rf_gridsearch.best_score_ > best_score:
            print("updating model", rf_gridsearch.best_score_, best_score)
            best_model = rf_gridsearch.best_estimator_
            best_score = rf_gridsearch.best_score_
            best_params = rf_gridsearch.best_params_
    return best_params


def get_best_fit_model(best_params, X_train, X_test, y_train, y_test):
    # fit on the entire trianing set
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("f1", f1_score(y_test, y_pred))
    print("recall", recall_score(y_test, y_pred))
    print("precision", precision_score(y_test, y_pred))
    print(
        "specificity",
        recall_score(
            ~np.array(y_test, dtype="bool"),
            ~np.array(y_pred, dtype="bool"),
        ),
    )
    return model


def print_and_plot_auc(y_tr, y_pr, title):
    score = roc_auc_score(y_tr, y_pr)
    print(score)
    fpr, tpr, _ = roc_curve(y_tr, y_pr)

    lw = 2
    plt.cla()
    plt.figure(1)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % score)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC for %s" % title)
    plt.legend(loc="lower right")
    plt.savefig("cfu_prediction_AUC.png")
    plt.close()
    return fpr, tpr


if __name__ == "__main__":
    location_to_cfu_dict = load_cfu_by_location()

    signal_file_paths = get_list_of_files_to_use_for_signals()
    df_signals = load_sites_data(signal_file_paths)

    # canonical signals
    tlf_signal_path = "tryptophan200ppb_25s.txt"
    hlf_signal_path = "humicacid0.1perc_25s.txt"
    df_tlf, df_hlf = load_canonical_signals(tlf_signal_path, hlf_signal_path)

    interped_curves = get_interped_curves(
        df_signals,
        df_tlf,
        df_hlf,
    )

    X, y_cfu = get_classification_data(interped_curves)

    # apply normalization
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cfu, test_size=0.33, random_state=0
    )
    best_params = get_best_param_for_fit(X_train, X_test, y_train, y_test)
    print(best_params)

    best_model = get_best_fit_model(best_params, X_train, X_test, y_train, y_test)

    probs = best_model.predict_proba(X_test)
    fpr, tpr = print_and_plot_auc(y_test, probs[:, 1], "cfu prediction")
