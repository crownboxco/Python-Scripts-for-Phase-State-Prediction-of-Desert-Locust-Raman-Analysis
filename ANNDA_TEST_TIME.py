import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_FILE = "best_ann_pipeline.joblib"
INPUT_CSV = "C2I_pp.csv"          # change if needed

SAMPLE_COL_FALLBACK = "Sample"
TRUE_ENV_COL = "Env"
TRUE_ED_COL = "ED"
ED_HOURS = [0.5, 1, 2, 4, 8, 12, 24, 48, 72]

FIG_W = 8
FIG_H = 5.5
CMAP = "Blues"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def natural_key(text):
    text = str(text)
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', text)]

def build_ed_hour_mapping(ed_values, ed_hours):
    unique_ed = sorted(pd.Series(ed_values).dropna().astype(str).unique(), key=natural_key)
    if len(unique_ed) != len(ed_hours):
        raise ValueError(
            f"Number of unique ED labels ({len(unique_ed)}) does not match "
            f"number of supplied hour labels ({len(ed_hours)}).\n"
            f"Unique ED labels found: {unique_ed}"
        )
    return {ed_label: f"{hour} h" for ed_label, hour in zip(unique_ed, ed_hours)}

def compute_binary_metrics_ovr(y_true, y_pred, positive_class):
    y_true_bin = (np.asarray(y_true) == positive_class).astype(int)
    y_pred_bin = (np.asarray(y_pred) == positive_class).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ba = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else np.nan
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "F1": f1,
        "Balanced_Accuracy": ba,
        "TPR": tpr,
        "TNR": tnr
    }

def summarize_metrics_by_env_class(y_true, y_pred, env_classes, group_name):
    per_class_rows = []

    for cls in env_classes:
        m = compute_binary_metrics_ovr(y_true, y_pred, cls)
        per_class_rows.append({
            "Group": group_name,
            "Env_Class": cls,
            **m
        })

    per_class_df = pd.DataFrame(per_class_rows)

    summary_row = {
        "Group": group_name,
        "N": len(y_true),
        "Mean_F1": per_class_df["F1"].mean(skipna=True),
        "SD_F1": per_class_df["F1"].std(ddof=1, skipna=True),
        "Mean_Balanced_Accuracy": per_class_df["Balanced_Accuracy"].mean(skipna=True),
        "SD_Balanced_Accuracy": per_class_df["Balanced_Accuracy"].std(ddof=1, skipna=True),
        "Mean_TPR": per_class_df["TPR"].mean(skipna=True),
        "SD_TPR": per_class_df["TPR"].std(ddof=1, skipna=True),
        "Mean_TNR": per_class_df["TNR"].mean(skipna=True),
        "SD_TNR": per_class_df["TNR"].std(ddof=1, skipna=True),
        "Macro_F1_multiclass": f1_score(y_true, y_pred, labels=env_classes, average="macro", zero_division=0),
        "Macro_Balanced_Accuracy_multiclass": balanced_accuracy_score(y_true, y_pred)
    }

    return per_class_df, pd.DataFrame([summary_row])

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

saved = joblib.load(MODEL_FILE)

pipe = saved["pipeline"]
label_encoder = saved["label_encoder"]
feature_columns = saved["feature_columns"]
saved_class_names = saved.get("class_names", None)
sample_col = saved.get("sample_col", SAMPLE_COL_FALLBACK)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.astype(str).str.strip()

required_cols = set(feature_columns) | {sample_col, TRUE_ENV_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

# If ED exists, use it for the contingency plot
has_ed = TRUE_ED_COL in df.columns

sample_env_counts = df.groupby(sample_col)[TRUE_ENV_COL].nunique()
bad_env_samples = sample_env_counts[sample_env_counts > 1].index.tolist()
if bad_env_samples:
    raise ValueError(
        f"The following samples have more than one '{TRUE_ENV_COL}' label, "
        f"so sample-level aggregation is ambiguous: {bad_env_samples[:10]}"
        + (" ..." if len(bad_env_samples) > 10 else "")
    )

if has_ed:
    sample_ed_counts = df.groupby(sample_col)[TRUE_ED_COL].nunique()
    bad_ed_samples = sample_ed_counts[sample_ed_counts > 1].index.tolist()
    if bad_ed_samples:
        raise ValueError(
            f"The following samples have more than one '{TRUE_ED_COL}' label, "
            f"so sample-level aggregation is ambiguous: {bad_ed_samples[:10]}"
            + (" ..." if len(bad_ed_samples) > 10 else "")
        )

agg_dict = {TRUE_ENV_COL: "first", **{c: "mean" for c in feature_columns}}
if has_ed:
    agg_dict[TRUE_ED_COL] = "first"

sample_df_new = (
    df.groupby(sample_col)
      .agg(agg_dict)
      .reset_index()
)

# Keep column order tidy
ordered_cols = [sample_col, TRUE_ENV_COL] + ([TRUE_ED_COL] if has_ed else []) + feature_columns
sample_df_new = sample_df_new[ordered_cols]
X_new = sample_df_new[feature_columns].to_numpy(dtype=float)
y_pred_enc = pipe.predict(X_new)
y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

sample_df_new["Predicted_Env"] = y_pred_labels

y_true_env = sample_df_new[TRUE_ENV_COL].astype(str).values
y_pred_env = sample_df_new["Predicted_Env"].astype(str).values

if saved_class_names is not None:
    env_classes = [str(x) for x in saved_class_names]
else:
    env_classes = sorted(pd.Series(np.concatenate([y_true_env, y_pred_env])).unique(), key=natural_key)

misclassified = int(np.sum(y_true_env != y_pred_env))
n_total = len(y_true_env)
n_correct = n_total - misclassified

print("\n" + "=" * 80)
print("ANN MODEL TEST RESULTS")
print("=" * 80)
print(f"Model file          : {MODEL_FILE}")
print(f"Input file          : {INPUT_CSV}")
print(f"Samples evaluated   : {n_total}")
print(f"Correct             : {n_correct}")
print(f"Misclassifications  : {misclassified}")
print(f"Accuracy            : {n_correct / n_total:.4f}")
print(f"Macro F1            : {f1_score(y_true_env, y_pred_env, labels=env_classes, average='macro', zero_division=0):.4f}")
print(f"Balanced Accuracy   : {balanced_accuracy_score(y_true_env, y_pred_env):.4f}")

global_per_class_df, global_summary_df = summarize_metrics_by_env_class(
    y_true=y_true_env,
    y_pred=y_pred_env,
    env_classes=env_classes,
    group_name="Global"
)

print("\nGlobal one-vs-rest per-class metrics:")
print(global_per_class_df.to_string(index=False))

print("\nGlobal summary (mean ± SD across Env one-vs-rest metrics):")
print(global_summary_df.to_string(index=False))

print("\nMisclassified samples:")
mis_df = sample_df_new.loc[y_true_env != y_pred_env, [sample_col, TRUE_ENV_COL, "Predicted_Env"]]
if has_ed:
    mis_df = sample_df_new.loc[y_true_env != y_pred_env, [sample_col, TRUE_ENV_COL, TRUE_ED_COL, "Predicted_Env"]]

if mis_df.empty:
    print("None")
else:
    print(mis_df.to_string(index=False))

if has_ed:
    y_true_ed = sample_df_new[TRUE_ED_COL].astype(str).values
    ed_to_hour = build_ed_hour_mapping(y_true_ed, ED_HOURS)
    sample_df_new["ED_Hour_Label"] = sample_df_new[TRUE_ED_COL].astype(str).map(ed_to_hour)
    ordered_ed_hour_labels = [f"{h} h" for h in ED_HOURS]

    contingency = pd.crosstab(
        index=sample_df_new["Predicted_Env"].astype(str),
        columns=sample_df_new["ED_Hour_Label"].astype(str),
        dropna=False
    )

    contingency = contingency.reindex(index=env_classes, fill_value=0)
    contingency = contingency.reindex(columns=ordered_ed_hour_labels, fill_value=0)

    cm_counts = contingency.to_numpy(dtype=float)
    col_sums = cm_counts.sum(axis=0, keepdims=True)
    cm_percent = np.divide(cm_counts, col_sums, where=(col_sums != 0))
    cm_percent = np.nan_to_num(cm_percent)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap=CMAP, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    )

    ax.set(
        xticks=np.arange(len(ordered_ed_hour_labels)),
        yticks=np.arange(len(env_classes)),
        xticklabels=ordered_ed_hour_labels,
        yticklabels=env_classes,
        xlabel="Exposure Time",
        ylabel="Predicted Class"
    )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            ax.text(
                j, i,
                f"{cm_counts[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10
            )

    plt.tight_layout()
    plt.show()

else:
    cm = confusion_matrix(y_true_env, y_pred_env, labels=env_classes)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap=CMAP, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    )

    ax.set(
        xticks=np.arange(len(env_classes)),
        yticks=np.arange(len(env_classes)),
        xticklabels=env_classes,
        yticklabels=env_classes,
        xlabel="True Env",
        ylabel="Predicted Env"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=10
            )

    plt.tight_layout()
    plt.show()