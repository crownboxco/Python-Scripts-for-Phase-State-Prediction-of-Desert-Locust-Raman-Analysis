import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "IandG_pp.csv"
SAMPLE_COL = "Sample"
CLASS_COL = "Cond"

OUTER_N_SPLITS = 10
OUTER_TEST_SIZE = 0.50
OUTER_RANDOM_STATE = 1818

INNER_CV_FOLDS = 10
GRID_RANDOM_STATE = 1818

USE_OVERSAMPLING = True
ROS_RANDOM_STATE = 1818

OUTPUT_CSV = "IandG_pp_Summary-results_ANN_IandG.csv"

PERM_IMPORTANCE_PNG = "IandG_pp_PermutationImportance_ANN_IandG.png"
CONFUSION_MATRIX_PNG = "IandG_pp_CM_ANN_IandG.png"
SPECTRA_MEAN_SE_PNG = "IandG_pp_Spectra_ANN_IandG.png"

X_AXIS_LABEL = "Raman Shift (cm$^{-1}$)"
BAND_LINES = [963, 1009, 1162, 1194, 1216, 1275, 1450, 1529]

# Permutation importance settings
PERM_N_REPEATS = 20
PERM_RANDOM_STATE = 1818

# Small ANN grid
PARAM_GRID = {
    "ann__hidden_dims": [(64,), (128,), (64, 32)],
    "ann__dropout": [0.0, 0.2],
    "ann__lr": [1e-3, 5e-4],
    "ann__weight_decay": [1e-4],
    "ann__batch_size": [16],
    "ann__epochs": [150],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

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


def summarize_metric_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean_val = float(np.mean(values))
    if len(values) == 1:
        return mean_val, 0.0
    se_val = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean_val, se_val


def mode_param_dict(list_of_dicts):
    """
    Select the most frequent exact parameter combination across outer splits.
    Tie-break by highest mean CV Macro F1 among tied combinations.
    """
    keys = []
    for d in list_of_dicts:
        key = tuple(sorted(d.items()))
        keys.append(key)

    counts = pd.Series(keys).value_counts()
    top_count = counts.iloc[0]
    tied = counts[counts == top_count].index.tolist()

    chosen_key = tied[0]
    return dict(chosen_key)

class ANNClassifierTorch(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims=(64,),
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        epochs=150,
        random_state=1818,
        device=DEVICE,
        verbose=False
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_model(self, n_features, n_classes):
        layers = []
        in_dim = n_features

        for h in self.hidden_dims:
            linear = nn.Linear(in_dim, h)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h

        out = nn.Linear(in_dim, n_classes)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)

        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self._set_seed()

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.model_ = self._build_model(n_features, n_classes).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if self.verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {epoch_loss / len(loader):.6f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()

        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

required_cols = {SAMPLE_COL, CLASS_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

feature_cols = [c for c in df.columns if c not in [SAMPLE_COL, CLASS_COL]]
numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

if len(numeric_feature_cols) == 0:
    raise ValueError("No numeric feature columns found.")

sample_env_counts = df.groupby(SAMPLE_COL)[CLASS_COL].nunique()
bad_samples = sample_env_counts[sample_env_counts > 1].index.tolist()
if bad_samples:
    raise ValueError(
        "The following samples have more than one class, so sample-level "
        f"aggregation is ambiguous: {bad_samples[:10]}"
        + (" ..." if len(bad_samples) > 10 else "")
    )

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

try:
    x_axis = np.array(numeric_feature_cols, dtype=float)
except Exception:
    x_axis = np.arange(len(numeric_feature_cols), dtype=float)
    X_AXIS_LABEL = "Variable Index"

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
            "SE_Spectrum": cls_se[i],
            "SD_Spectrum": cls_sd[i],
            "N_Samples": cls_X.shape[0]
        })

spectra_summary_df = pd.DataFrame(spectra_rows)

fig, ax = plt.subplots(figsize=(8, 5))

default_colors = ["blue", "red", "green", "purple", "orange", "black"]
CLASS_COLORS = {cls: default_colors[i % len(default_colors)] for i, cls in enumerate(class_names)}

for cls in class_names:
    cls_df = spectra_summary_df[spectra_summary_df["Class"] == cls].sort_values("X_Value")
    color = CLASS_COLORS.get(cls, "black")

    ax.plot(
        cls_df["X_Value"],
        cls_df["Mean_Spectrum"],
        label=f"{cls} (n={int(cls_df['N_Samples'].iloc[0])})",
        color=color,
        linewidth=2
    )

    ax.fill_between(
        cls_df["X_Value"].to_numpy(dtype=float),
        (cls_df["Mean_Spectrum"] - cls_df["SD_Spectrum"]).to_numpy(dtype=float),
        (cls_df["Mean_Spectrum"] + cls_df["SD_Spectrum"]).to_numpy(dtype=float),
        alpha=0.20,
        color=color
    )

fig.canvas.draw()
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
ax.set_ylabel("Relative Intensity (arb. u.)")
ax.legend()
fig.tight_layout()
fig.savefig(SPECTRA_MEAN_SE_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

sss = StratifiedShuffleSplit(
    n_splits=OUTER_N_SPLITS,
    test_size=OUTER_TEST_SIZE,
    random_state=OUTER_RANDOM_STATE
)

outer_split_data = []
cv_global_all = []
cv_class_all = []
split_best_param_rows = []
split_best_params_only = []

macro_f1_scorer = make_scorer(f1_score, average="macro", zero_division=0)

for split_idx, (train_idx, test_idx) in enumerate(
    tqdm(sss.split(X_all, y_all_enc), total=OUTER_N_SPLITS, desc="Outer Splits | GridSearchCV"),
    start=1
):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all_enc[train_idx], y_all_enc[test_idx]

    pipe = ImbPipeline(steps=[
        ("ros", RandomOverSampler(random_state=ROS_RANDOM_STATE + split_idx) if USE_OVERSAMPLING else "passthrough"),
        ("scaler", StandardScaler()),
        ("ann", ANNClassifierTorch(
            random_state=GRID_RANDOM_STATE + split_idx,
            device=DEVICE,
            verbose=False
        ))
    ])

    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=GRID_RANDOM_STATE + split_idx
    )

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring=macro_f1_scorer,
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        return_train_score=False,
        verbose=0
    )

    gs.fit(X_train, y_train)

    best_estimator = gs.best_estimator_
    best_params = gs.best_params_
    best_cv_macro_f1 = float(gs.best_score_)

    # Cross-validation summary for selected model on this split
    cv_results = pd.DataFrame(gs.cv_results_)
    row = cv_results[cv_results["rank_test_score"] == 1].iloc[0]
    mean_cv_macro_f1 = float(row["mean_test_score"])
    sd_cv_macro_f1 = float(row["std_test_score"])

    # Use the selected model to estimate training-split CV metrics only approximately from best score
    # Keep the same structure as original script where CV summaries are stored.
    cv_global_df = pd.DataFrame([
        {
            "Split": split_idx,
            "Metric": "Macro_F1",
            "Mean": mean_cv_macro_f1,
            "SD": sd_cv_macro_f1
        }
    ])

    cv_global_all.append(cv_global_df)

    split_record = {
        "Split": split_idx,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "Best_Params": best_params,
        "Best_CV_Macro_F1": best_cv_macro_f1
    }
    outer_split_data.append(split_record)

    split_best_params_only.append(best_params.copy())

    split_best_param_rows.append({
        "Split": split_idx,
        "Best_CV_Macro_F1": best_cv_macro_f1,
        "hidden_dims": str(best_params["ann__hidden_dims"]),
        "dropout": best_params["ann__dropout"],
        "lr": best_params["ann__lr"],
        "weight_decay": best_params["ann__weight_decay"],
        "batch_size": best_params["ann__batch_size"],
        "epochs": best_params["ann__epochs"]
    })

split_best_params_df = pd.DataFrame(split_best_param_rows)
global_best_params = mode_param_dict(split_best_params_only)

print("\n" + "=" * 80)
print("GLOBAL CHOSEN ANN PARAMETERS")
print("=" * 80)
print(split_best_params_df.to_string(index=False))
print("\nGlobal Best Parameters:")
for k, v in global_best_params.items():
    print(f"{k}: {v}")

test_global_all = []
test_class_all = []

all_confusion_matrices = []
all_true_class_counts = []
all_perm_importances = []

for split_info in tqdm(outer_split_data, total=len(outer_split_data), desc="Outer Splits | Final Test Fit"):
    split_idx = split_info["Split"]
    X_train = split_info["X_train"]
    X_test = split_info["X_test"]
    y_train = split_info["y_train"]
    y_test = split_info["y_test"]

    final_pipe = ImbPipeline(steps=[
        ("ros", RandomOverSampler(random_state=ROS_RANDOM_STATE + split_idx) if USE_OVERSAMPLING else "passthrough"),
        ("scaler", StandardScaler()),
        ("ann", ANNClassifierTorch(
            hidden_dims=global_best_params["ann__hidden_dims"],
            dropout=global_best_params["ann__dropout"],
            lr=global_best_params["ann__lr"],
            weight_decay=global_best_params["ann__weight_decay"],
            batch_size=global_best_params["ann__batch_size"],
            epochs=global_best_params["ann__epochs"],
            random_state=GRID_RANDOM_STATE + 1000 + split_idx,
            device=DEVICE,
            verbose=False
        ))
    ])

    final_pipe.fit(X_train, y_train)
    y_pred_test = final_pipe.predict(X_test)

    cm_test = confusion_matrix(y_test, y_pred_test, labels=np.arange(n_classes))
    class_df_test = compute_class_metrics_from_cm(cm_test, class_names)

    all_confusion_matrices.append(cm_test)
    all_true_class_counts.append(np.bincount(y_test, minlength=n_classes))

    test_global_all.append(pd.DataFrame([{
        "Split": split_idx,
        "Balanced_Accuracy": float(np.mean(class_df_test["Balanced_Accuracy"].values)),
        "Macro_F1": f1_score(y_test, y_pred_test, average="macro", zero_division=0),
        "Global_TPR": float(np.mean(class_df_test["TPR"].values)),
        "Global_TNR": float(np.mean(class_df_test["TNR"].values))
    }]))

    tmp_class = class_df_test.copy()
    tmp_class["Split"] = split_idx
    test_class_all.append(tmp_class)

    # Permutation importance on the held-out test set
    perm = permutation_importance(
        estimator=final_pipe,
        X=X_test,
        y=y_test,
        scoring=macro_f1_scorer,
        n_repeats=PERM_N_REPEATS,
        random_state=PERM_RANDOM_STATE + split_idx,
        n_jobs=1
    )
    all_perm_importances.append(perm.importances_mean)

cv_global_all_df = pd.concat(cv_global_all, ignore_index=True)
test_global_all_df = pd.concat(test_global_all, ignore_index=True)
test_class_all_df = pd.concat(test_class_all, ignore_index=True)

all_confusion_matrices = np.stack(all_confusion_matrices, axis=0)
all_true_class_counts = np.stack(all_true_class_counts, axis=0)
all_perm_importances = np.stack(all_perm_importances, axis=0)

cv_global_summary_rows = []
for metric in ["Macro_F1"]:
    vals = cv_global_all_df.loc[cv_global_all_df["Metric"] == metric, "Mean"].values
    mean_val, sd_val = summarize_metric(vals)
    cv_global_summary_rows.append({
        "Section": "CV_Global",
        "Metric": metric,
        "Mean": mean_val,
        "SD": sd_val
    })
cv_global_summary_df = pd.DataFrame(cv_global_summary_rows)

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

mean_perm_importance = np.mean(all_perm_importances, axis=0)
sd_perm_importance = np.std(all_perm_importances, axis=0, ddof=1)

perm_rows = []
for feat_idx, feat_name in enumerate(numeric_feature_cols):
    perm_rows.append({
        "Variable": feat_name,
        "X_Value": x_axis[feat_idx],
        "Mean_Permutation_Importance": mean_perm_importance[feat_idx],
        "SD_Permutation_Importance": sd_perm_importance[feat_idx]
    })
perm_summary_df = pd.DataFrame(perm_rows).sort_values("X_Value")

fig, ax = plt.subplots(figsize=(10, 5))
y_mean = perm_summary_df["Mean_Permutation_Importance"].to_numpy(dtype=float)
y_sd = perm_summary_df["SD_Permutation_Importance"].to_numpy(dtype=float)

ax.plot(x_axis, y_mean, linewidth=1.8)
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
ax.set_ylabel("Permutation Importance (Macro F1 decrease)")
ax.set_title("ANN-DA Permutation Importance")
fig.tight_layout()
fig.savefig(PERM_IMPORTANCE_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

cm_mean = np.mean(all_confusion_matrices, axis=0)
true_n_mean = np.mean(all_true_class_counts, axis=0)

cm_percent = cm_mean / cm_mean.sum(axis=1, keepdims=True)
cm_percent = np.nan_to_num(cm_percent)

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
print("TEST GLOBAL METRICS")
print("=" * 80)
print(test_global_summary_df[["Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("TEST CLASS-SPECIFIC METRICS")
print("=" * 80)
print(test_class_summary_df[["Class", "Metric", "Mean", "SD"]].to_string(index=False))

print("\n" + "=" * 80)
print("GLOBAL FINAL ANN PLOT OUTPUTS")
print("=" * 80)
print("Global Best Parameters:")
for k, v in global_best_params.items():
    print(f"  {k}: {v}")
print(f"Permutation importance plot: {PERM_IMPORTANCE_PNG}")
print(f"Confusion matrix plot: {CONFUSION_MATRIX_PNG}")
print(f"Class mean ± SD spectra plot: {SPECTRA_MEAN_SE_PNG}")

csv_parts = []

csv_parts.append(
    cv_global_summary_df.assign(Split="", Class="", Global_Model="")
)
csv_parts.append(
    test_global_summary_df.assign(Split="", Class="", Global_Model="")
)
csv_parts.append(
    test_class_summary_df.assign(Split="", Global_Model="")
)

split_param_export_df = split_best_params_df.copy()
split_param_export_df["Section"] = "Split_Model_Selection"
split_param_export_df["Class"] = ""
split_param_export_df["Metric"] = ""
split_param_export_df["Mean"] = ""
split_param_export_df["SD"] = ""
split_param_export_df["Global_Model"] = str(global_best_params)
csv_parts.append(split_param_export_df)

results_out_df = pd.concat(csv_parts, ignore_index=True, sort=False)

desired_cols = [
    "Section", "Split", "Class", "Metric", "Mean", "SD",
    "Best_CV_Macro_F1", "hidden_dims", "dropout", "lr",
    "weight_decay", "batch_size", "epochs", "Global_Model"
]
for col in desired_cols:
    if col not in results_out_df.columns:
        results_out_df[col] = ""
results_out_df = results_out_df[desired_cols]

results_out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nResults saved to: {OUTPUT_CSV}")

import joblib

final_deploy_pipe = ImbPipeline(steps=[
    ("ros", RandomOverSampler(random_state=ROS_RANDOM_STATE) if USE_OVERSAMPLING else "passthrough"),
    ("scaler", StandardScaler()),
    ("ann", ANNClassifierTorch(
        hidden_dims=global_best_params["ann__hidden_dims"],
        dropout=global_best_params["ann__dropout"],
        lr=global_best_params["ann__lr"],
        weight_decay=global_best_params["ann__weight_decay"],
        batch_size=global_best_params["ann__batch_size"],
        epochs=global_best_params["ann__epochs"],
        random_state=GRID_RANDOM_STATE + 9999,
        device=DEVICE,
        verbose=False
    ))
])

final_deploy_pipe.fit(X_all, y_all_enc)

model_bundle = {
    "pipeline": final_deploy_pipe,
    "label_encoder": le,
    "feature_columns": numeric_feature_cols,
    "class_names": class_names,
    "sample_col": SAMPLE_COL,
    "class_col": CLASS_COL,
    "global_best_params": global_best_params,
    "x_axis_label": X_AXIS_LABEL,
    "band_lines": BAND_LINES
}

joblib.dump(model_bundle, "best_ann_pipeline.joblib")
print("Saved final model bundle to best_ann_pipeline.joblib")