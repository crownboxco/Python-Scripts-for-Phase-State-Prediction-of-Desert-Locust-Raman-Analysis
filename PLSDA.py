import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

DATA_PATH = "CvI_pp.csv"
SAMPLE_COL = "Sample"
CLASS_COL = "Env"
OUTER_N_SPLITS = 10
OUTER_TEST_SIZE = 0.50
OUTER_RANDOM_STATE = 1818
LV_MIN = 2
LV_MAX = 20
BOOT_N_SPLITS = 1000
BOOT_RANDOM_STATE = 1818
USE_OVERSAMPLING = True
ROS_RANDOM_STATE = 1818

OUTPUT_CSV = "CvI_pp_Summary-results_CvI.csv"

LOADINGS_COMBINED_PNG = "CvI_pp_loadings_CvI.png"
CONFUSION_MATRIX_PNG = "CvI_pp_CM_CvI.png"
SPECTRA_MEAN_SE_PNG = "CvI_pp_Spectra_CvI.png"

X_AXIS_LABEL = "Raman Shift (cm$^{-1}$)"
BAND_LINES = [963, 1009, 1162, 1194, 1216, 1275, 1450, 1529]

def compute_class_metrics_from_cm(cm: np.ndarray, class_names):
    results = []
    total = cm.sum()

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
        ba = (tpr + tnr) / 2.0

        results.append({
            "Class": cls,
            "TPR": tpr,
            "TNR": tnr,
            "Precision": precision,
            "F1": f1,
            "Balanced_Accuracy": ba
        })

    return pd.DataFrame(results)


def summarize_metric(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def predict_plsda(pls, X):
    y_hat = pls.predict(X)
    return np.argmax(y_hat, axis=1)


def fit_plsda(X_train, y_train_enc, n_components, onehot_encoder, oversample=True, ros_random_state=42):
    if oversample:
        ros = RandomOverSampler(random_state=ros_random_state)
        X_fit, y_fit = ros.fit_resample(X_train, y_train_enc)
    else:
        X_fit, y_fit = X_train, y_train_enc

    y_fit_oh = onehot_encoder.transform(y_fit.reshape(-1, 1))

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_fit, y_fit_oh)
    return pls


def stratified_bootstrap_inbag_oob_indices(y, rng):
    y = np.asarray(y)
    classes = np.unique(y)

    inbag_parts = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        draw = rng.choice(cls_idx, size=len(cls_idx), replace=True)
        inbag_parts.append(draw)

    inbag_idx = np.concatenate(inbag_parts)
    rng.shuffle(inbag_idx)

    unique_inbag = np.unique(inbag_idx)
    oob_mask = np.ones(len(y), dtype=bool)
    oob_mask[unique_inbag] = False
    oob_idx = np.where(oob_mask)[0]

    return inbag_idx, oob_idx


def bootstrap_select_split_lv(
    X_train,
    y_train,
    class_names,
    lv_min,
    lv_max,
    n_boot=1000,
    boot_random_state=1818,
    oversample=True,
    ros_random_state=1818,
    split_idx=None
):
    rng = np.random.default_rng(boot_random_state)
    n_train = X_train.shape[0]

    max_allowed = min(lv_max, X_train.shape[1], n_train - 1)
    if max_allowed < lv_min:
        raise ValueError(
            f"Not enough training samples/features for LV search. "
            f"Allowed maximum LV is {max_allowed}, but lv_min={lv_min}."
        )

    lv_grid = list(range(lv_min, max_allowed + 1))

    ohe_inner = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_inner.fit(y_train.reshape(-1, 1))

    global_store = {
        lv: {
            "Balanced_Accuracy": [],
            "Macro_F1": [],
            "Global_TPR": [],
            "Global_TNR": []
        }
        for lv in lv_grid
    }

    class_store = {
        lv: {
            cls: {
                "TPR": [],
                "TNR": [],
                "Precision": [],
                "F1": [],
                "Balanced_Accuracy": []
            }
            for cls in class_names
        }
        for lv in lv_grid
    }

    bootstrap_best_lvs = []

    for b in tqdm(
        range(1, n_boot + 1),
        desc=f"Split {split_idx} | Bootstrap",
        leave=False
    ):
        inbag_idx, oob_idx = stratified_bootstrap_inbag_oob_indices(y_train, rng)

        if len(oob_idx) == 0:
            continue

        X_inbag = X_train[inbag_idx]
        y_inbag = y_train[inbag_idx]
        X_oob = X_train[oob_idx]
        y_oob = y_train[oob_idx]

        if len(np.unique(y_oob)) < 2:
            continue

        this_boot_scores = []

        for lv in lv_grid:
            try:
                pls = fit_plsda(
                    X_train=X_inbag,
                    y_train_enc=y_inbag,
                    n_components=lv,
                    onehot_encoder=ohe_inner,
                    oversample=oversample,
                    ros_random_state=ros_random_state + split_idx * 10000 + b * 100 + lv
                )

                y_pred_oob = predict_plsda(pls, X_oob)
                cm = confusion_matrix(y_oob, y_pred_oob, labels=np.arange(len(class_names)))
                class_df = compute_class_metrics_from_cm(cm, class_names)

                macro_f1 = f1_score(y_oob, y_pred_oob, average="macro", zero_division=0)
                global_tpr = float(np.mean(class_df["TPR"].values))
                global_tnr = float(np.mean(class_df["TNR"].values))
                global_ba = global_tpr

                global_store[lv]["Balanced_Accuracy"].append(global_ba)
                global_store[lv]["Macro_F1"].append(macro_f1)
                global_store[lv]["Global_TPR"].append(global_tpr)
                global_store[lv]["Global_TNR"].append(global_tnr)

                for _, row in class_df.iterrows():
                    cls = row["Class"]
                    class_store[lv][cls]["TPR"].append(float(row["TPR"]))
                    class_store[lv][cls]["TNR"].append(float(row["TNR"]))
                    class_store[lv][cls]["Precision"].append(float(row["Precision"]))
                    class_store[lv][cls]["F1"].append(float(row["F1"]))
                    class_store[lv][cls]["Balanced_Accuracy"].append(float(row["Balanced_Accuracy"]))

                this_boot_scores.append((lv, macro_f1))

            except Exception:
                continue

        if len(this_boot_scores) == 0:
            continue

        this_boot_scores = sorted(this_boot_scores, key=lambda x: (-x[1], x[0]))
        best_lv_this_boot = int(this_boot_scores[0][0])
        bootstrap_best_lvs.append(best_lv_this_boot)

    if len(bootstrap_best_lvs) == 0:
        raise RuntimeError("No valid bootstrap/OOB evaluations were produced.")

    split_best_lv = int(np.median(bootstrap_best_lvs))

    cv_global_rows = []
    for metric in ["Balanced_Accuracy", "Macro_F1", "Global_TPR", "Global_TNR"]:
        mean_val, sd_val = summarize_metric(global_store[split_best_lv][metric])
        cv_global_rows.append({
            "Split": split_idx,
            "Split_Best_LV": split_best_lv,
            "Metric": metric,
            "Mean": mean_val,
            "SD": sd_val
        })
    cv_global_df = pd.DataFrame(cv_global_rows)

    cv_class_rows = []
    for cls in class_names:
        for metric in ["TPR", "TNR", "Precision", "F1", "Balanced_Accuracy"]:
            mean_val, sd_val = summarize_metric(class_store[split_best_lv][cls][metric])
            cv_class_rows.append({
                "Split": split_idx,
                "Split_Best_LV": split_best_lv,
                "Class": cls,
                "Metric": metric,
                "Mean": mean_val,
                "SD": sd_val
            })
    cv_class_df = pd.DataFrame(cv_class_rows)

    lv_dist_df = pd.DataFrame({
        "Split": split_idx,
        "Bootstrap_Best_LV": bootstrap_best_lvs
    })

    return split_best_lv, cv_global_df, cv_class_df, lv_dist_df

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

required_cols = {SAMPLE_COL, CLASS_COL}
feature_cols = [c for c in df.columns if c not in [SAMPLE_COL, CLASS_COL]]
numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

#sample_env_counts = df.groupby(SAMPLE_COL)[CLASS_COL].nunique()

sample_df = (
    df.groupby(SAMPLE_COL)
      .agg({CLASS_COL: "first", **{c: "mean" for c in numeric_feature_cols}})
      .reset_index()
)

X_all = sample_df[numeric_feature_cols].to_numpy(dtype=float)
y_all = sample_df[CLASS_COL].astype(str).to_numpy()

le = LabelEncoder()
y_all_enc = le.fit_transform(y_all)
class_names = le.classes_
n_classes = len(class_names)

ohe_outer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_outer.fit(y_all_enc.reshape(-1, 1))

x_axis = np.array(numeric_feature_cols, dtype=float)

spectra_rows = []
for cls in class_names:
    cls_mask = sample_df[CLASS_COL].astype(str).values == cls
    cls_X = sample_df.loc[cls_mask, numeric_feature_cols].to_numpy(dtype=float)

    cls_mean = np.mean(cls_X, axis=0)
    if cls_X.shape[0] > 1:
        cls_sd = np.std(cls_X, axis=0, ddof=1)
        cls_se = cls_sd / np.sqrt(cls_X.shape[0])
    else:
        cls_sd = np.zeros(cls_X.shape[1], dtype=float)
        cls_se = np.zeros(cls_X.shape[1], dtype=float)

    for i, feat in enumerate(numeric_feature_cols):
        spectra_rows.append({
            "Class": cls,
            "Variable": feat,
            "X_Value": x_axis[i],
            "Mean_Spectrum": cls_mean[i],
            "SE_Spectrum": cls_se[i],   # use SE if that is what you want
            "SD_Spectrum": cls_sd[i],
            "N_Samples": cls_X.shape[0]
        })

spectra_summary_df = pd.DataFrame(spectra_rows)

fig, ax = plt.subplots(figsize=(8, 5))

CLASS_COLORS = {
    class_names[0]: "#FC6A03", #"#FC6A03"
    class_names[1]: "green", #green
    # class_names[2]: "green",
    # class_names[3]: "purple"
}

for cls in class_names:
    cls_df = spectra_summary_df[spectra_summary_df["Class"] == cls].sort_values("X_Value")
    color = CLASS_COLORS.get(cls, "black")

    ax.plot(
        cls_df["X_Value"],
        cls_df["Mean_Spectrum"],
        label=f"{cls} (n={int(cls_df['N_Samples'].iloc[0])})",
        color=color,
        linewidth=2)

    ax.fill_between(
        cls_df["X_Value"].to_numpy(dtype=float),
        (cls_df["Mean_Spectrum"] - cls_df["SD_Spectrum"]).to_numpy(dtype=float),
        (cls_df["Mean_Spectrum"] + cls_df["SD_Spectrum"]).to_numpy(dtype=float),
        alpha=0.20,
        color=color)

# draw first so limits are known
fig.canvas.draw()

# place labels near the top of the plotting region
ymin, ymax = ax.get_ylim()
text_y = ymax - 0.03 * (ymax - ymin)

for band in BAND_LINES:
    if x_axis.min() <= band <= x_axis.max():
        ax.axvline(band, linestyle="--", color="dimgray", linewidth=1.2)
        ax.text(
            band,
            text_y,
            f"{band}",
            rotation=90,
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6)
        )

ax.set_xlabel(X_AXIS_LABEL)
ax.set_ylabel("Relative Intensity")
ax.legend()
fig.tight_layout()
plt.show()
#fig.savefig(SPECTRA_MEAN_SE_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

sss = StratifiedShuffleSplit(
    n_splits=OUTER_N_SPLITS,
    test_size=OUTER_TEST_SIZE,
    random_state=OUTER_RANDOM_STATE
)

outer_split_data = []
cv_global_all = []
cv_class_all = []
lv_dist_all = []
split_best_lv_rows = []

for split_idx, (train_idx, test_idx) in enumerate(
    tqdm(sss.split(X_all, y_all_enc), total=OUTER_N_SPLITS, desc="Outer Splits | CV Selection"),
    start=1
):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all_enc[train_idx], y_all_enc[test_idx]

    split_best_lv, cv_global_df, cv_class_df, lv_dist_df = bootstrap_select_split_lv(
        X_train=X_train,
        y_train=y_train,
        class_names=class_names,
        lv_min=LV_MIN,
        lv_max=LV_MAX,
        n_boot=BOOT_N_SPLITS,
        boot_random_state=BOOT_RANDOM_STATE + split_idx,
        oversample=USE_OVERSAMPLING,
        ros_random_state=ROS_RANDOM_STATE,
        split_idx=split_idx
    )

    outer_split_data.append({
        "Split": split_idx,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "Split_Best_LV": split_best_lv
    })

    cv_global_all.append(cv_global_df)
    cv_class_all.append(cv_class_df)
    lv_dist_all.append(lv_dist_df)
    split_best_lv_rows.append({"Split": split_idx, "Split_Best_LV": split_best_lv})

split_best_lv_df = pd.DataFrame(split_best_lv_rows)
global_best_lv = int(np.median(split_best_lv_df["Split_Best_LV"].values))

print("\n" + "=" * 80)
print("GLOBAL CHOSEN LV")
print("=" * 80)
print(split_best_lv_df.to_string(index=False))
print(f"\nGlobal Best LV (median of split-level best LVs): {global_best_lv}")

test_global_all = []
test_class_all = []

all_loadings = []
all_confusion_matrices = []
all_true_class_counts = []

for split_info in tqdm(outer_split_data, total=len(outer_split_data), desc="Outer Splits | Final Test Fit"):
    split_idx = split_info["Split"]
    X_train = split_info["X_train"]
    X_test = split_info["X_test"]
    y_train = split_info["y_train"]
    y_test = split_info["y_test"]

    pls_final = fit_plsda(
        X_train=X_train,
        y_train_enc=y_train,
        n_components=global_best_lv,
        onehot_encoder=ohe_outer,
        oversample=USE_OVERSAMPLING,
        ros_random_state=ROS_RANDOM_STATE + split_idx
    )

    all_loadings.append(pls_final.x_loadings_.copy())

    y_pred_test = predict_plsda(pls_final, X_test)

    cm_test = confusion_matrix(y_test, y_pred_test, labels=np.arange(n_classes))
    class_df_test = compute_class_metrics_from_cm(cm_test, class_names)

    all_confusion_matrices.append(cm_test)
    all_true_class_counts.append(np.bincount(y_test, minlength=n_classes))

    test_global_all.append(pd.DataFrame([{
        "Split": split_idx,
        "Global_Best_LV": global_best_lv,
        "Balanced_Accuracy": float(np.mean(class_df_test["TPR"].values)),
        "Macro_F1": f1_score(y_test, y_pred_test, average="macro", zero_division=0),
        "Global_TPR": float(np.mean(class_df_test["TPR"].values)),
        "Global_TNR": float(np.mean(class_df_test["TNR"].values))
    }]))

    tmp_class = class_df_test.copy()
    tmp_class["Split"] = split_idx
    tmp_class["Global_Best_LV"] = global_best_lv
    test_class_all.append(tmp_class)

cv_global_all_df = pd.concat(cv_global_all, ignore_index=True)
cv_class_all_df = pd.concat(cv_class_all, ignore_index=True)
lv_dist_all_df = pd.concat(lv_dist_all, ignore_index=True)
test_global_all_df = pd.concat(test_global_all, ignore_index=True)
test_class_all_df = pd.concat(test_class_all, ignore_index=True)

all_loadings = np.stack(all_loadings, axis=0)
all_confusion_matrices = np.stack(all_confusion_matrices, axis=0)
all_true_class_counts = np.stack(all_true_class_counts, axis=0)

cv_global_summary_rows = []
for metric in ["Balanced_Accuracy", "Macro_F1", "Global_TPR", "Global_TNR"]:
    vals = cv_global_all_df.loc[cv_global_all_df["Metric"] == metric, "Mean"].values
    mean_val, sd_val = summarize_metric(vals)
    cv_global_summary_rows.append({
        "Section": "CV_Global",
        "Metric": metric,
        "Mean": mean_val,
        "SD": sd_val
    })
cv_global_summary_df = pd.DataFrame(cv_global_summary_rows)

cv_class_summary_rows = []
for cls in class_names:
    for metric in ["TPR", "TNR", "Precision", "F1", "Balanced_Accuracy"]:
        vals = cv_class_all_df.loc[
            (cv_class_all_df["Class"] == cls) & (cv_class_all_df["Metric"] == metric),
            "Mean"
        ].values
        mean_val, sd_val = summarize_metric(vals)
        cv_class_summary_rows.append({
            "Section": "CV_Class",
            "Class": cls,
            "Metric": metric,
            "Mean": mean_val,
            "SD": sd_val
        })
cv_class_summary_df = pd.DataFrame(cv_class_summary_rows)

test_global_summary_rows = []
for metric in ["Balanced_Accuracy", "Macro_F1", "Global_TPR", "Global_TNR"]:
    vals = test_global_all_df[metric].values
    mean_val, sd_val = summarize_metric(vals)
    test_global_summary_rows.append({
        "Section": "Test_Global",
        "Metric": metric,
        "Mean": mean_val,
        "SD": sd_val
    })
test_global_summary_df = pd.DataFrame(test_global_summary_rows)

test_class_summary_rows = []
for cls in class_names:
    cls_df = test_class_all_df[test_class_all_df["Class"] == cls]
    for metric in ["TPR", "TNR", "Precision", "F1", "Balanced_Accuracy"]:
        vals = cls_df[metric].values
        mean_val, sd_val = summarize_metric(vals)
        test_class_summary_rows.append({
            "Section": "Test_Class",
            "Class": cls,
            "Metric": metric,
            "Mean": mean_val,
            "SD": sd_val
        })
test_class_summary_df = pd.DataFrame(test_class_summary_rows)

mean_loadings = np.mean(all_loadings, axis=0)
sd_loadings = np.std(all_loadings, axis=0, ddof=1)

loadings_rows = []
for lv_idx in range(global_best_lv):
    for feat_idx, feat_name in enumerate(numeric_feature_cols):
        loadings_rows.append({
            "Global_Best_LV_Model_Component": lv_idx + 1,
            "Variable": feat_name,
            "X_Value": x_axis[feat_idx],
            "Mean_Loading": mean_loadings[feat_idx, lv_idx],
            "SD_Loading": sd_loadings[feat_idx, lv_idx]
        })
loadings_summary_df = pd.DataFrame(loadings_rows)

ncols = 1 if global_best_lv <= 3 else 2
nrows = int(np.ceil(global_best_lv / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.8 * nrows), squeeze=False)
axes_flat = axes.flatten()

for lv_idx in range(global_best_lv):
    ax = axes_flat[lv_idx]
    y_mean = mean_loadings[:, lv_idx]
    y_sd = sd_loadings[:, lv_idx]

    ax.plot(x_axis, y_mean)
    ax.fill_between(x_axis, y_mean - y_sd, y_mean + y_sd, alpha=0.25)

    ymin = np.min(y_mean - y_sd)
    ymax = np.max(y_mean + y_sd)
    yrange = ymax - ymin if ymax > ymin else 1.0
    text_y = ymax - 0.04 * yrange

    for band in BAND_LINES:
        if x_axis.min() <= band <= x_axis.max():
            ax.axvline(band, linestyle="--", color="dimgray", linewidth=1.2)
            ax.text(
                band,
                text_y,
                f"{band}",
                rotation=90,
                ha="center",
                va="top",
                fontsize=9,
                color="dimgray",
                clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6)
            )

    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("Loading")
    ax.set_title(f"LV {lv_idx + 1}")

for k in range(global_best_lv, len(axes_flat)):
    fig.delaxes(axes_flat[k])

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(LOADINGS_COMBINED_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

cm_mean = np.mean(all_confusion_matrices, axis=0)
true_n_mean = np.mean(all_true_class_counts, axis=0)
# Convert to row-wise percentages (per true class)
cm_percent = cm_mean / cm_mean.sum(axis=1, keepdims=True)
cm_percent = np.nan_to_num(cm_percent)  # handle any divide-by-zero
cm_rows = []
for i, true_cls in enumerate(class_names):
    for j, pred_cls in enumerate(class_names):
        cm_rows.append({
            "True_Class": true_cls,
            "Predicted_Class": pred_cls,
            "Mean_Count": cm_mean[i, j]
        })
cm_summary_df = pd.DataFrame(cm_rows)

x_display_labels = [
    f"{cls} (n={int(true_n_mean[i])})"
    for i, cls in enumerate(class_names)
]
y_display_labels = [str(cls) for cls in class_names]

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
)

ax.set(
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=x_display_labels,
    yticklabels=y_display_labels,
    ylabel="Predicted Class",
    xlabel="True Class"
)

plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

for i in range(n_classes):
    for j in range(n_classes):
        ax.text(
            j,
            i,
            f"{cm_mean[i, j]:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10
        )

plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "=" * 80)
print("CV GLOBAL METRICS")
print("=" * 80)
print(cv_global_summary_df[["Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("CV CLASS-SPECIFIC METRICS")
print("=" * 80)
print(cv_class_summary_df[["Class", "Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("TEST GLOBAL METRICS")
print("=" * 80)
print(test_global_summary_df[["Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("TEST CLASS-SPECIFIC METRICS")
print("=" * 80)
print(test_class_summary_df[["Class", "Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("GLOBAL FINAL LV PLOT OUTPUTS")
print("=" * 80)
print(f"Global Best LV: {global_best_lv}")
print(f"Combined loadings plot: {LOADINGS_COMBINED_PNG}")
print(f"Confusion matrix plot: {CONFUSION_MATRIX_PNG}")
print(f"Class mean ± SE spectra plot: {SPECTRA_MEAN_SE_PNG}")

csv_parts = []

csv_parts.append(
    cv_global_summary_df.assign(Split="", Split_Best_LV="", Global_Best_LV=global_best_lv, Class="")
)
csv_parts.append(
    cv_class_summary_df.assign(Split="", Split_Best_LV="", Global_Best_LV=global_best_lv)
)
csv_parts.append(
    test_global_summary_df.assign(Split="", Split_Best_LV="", Global_Best_LV=global_best_lv, Class="")
)
csv_parts.append(
    test_class_summary_df.assign(Split="", Split_Best_LV="", Global_Best_LV=global_best_lv)
)

split_lv_export_df = split_best_lv_df.copy()
split_lv_export_df["Section"] = "Split_LV_Selection"
split_lv_export_df["Global_Best_LV"] = global_best_lv
split_lv_export_df["Class"] = ""
split_lv_export_df["Metric"] = ""
split_lv_export_df["Mean"] = ""
split_lv_export_df["SD"] = ""
csv_parts.append(split_lv_export_df)

results_out_df = pd.concat(csv_parts, ignore_index=True, sort=False)

desired_cols = ["Section", "Split", "Split_Best_LV", "Global_Best_LV", "Class", "Metric", "Mean", "SD"]
for col in desired_cols:
    if col not in results_out_df.columns:
        results_out_df[col] = ""
results_out_df = results_out_df[desired_cols]

results_out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nResults saved to: {OUTPUT_CSV}")

import pickle

final_model = fit_plsda(
    X_train=X_all,
    y_train_enc=y_all_enc,
    n_components=global_best_lv,
    onehot_encoder=ohe_outer,
    oversample=USE_OVERSAMPLING,
    ros_random_state=ROS_RANDOM_STATE
)

model_bundle = {
    "model": final_model,
    "global_best_lv": global_best_lv,
    "label_encoder": le,
    "onehot_encoder": ohe_outer,
    "feature_columns": numeric_feature_cols,
    "class_names": class_names,
    "sample_col": SAMPLE_COL,
    "class_col": CLASS_COL
}

with open("best_plsda_model_CvI.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Saved model to best_plsda_model.pkl")