import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

MODEL_FILE = "best_plsda_model.pkl"
INPUT_CSV = "I2C_pp.csv"

SAMPLE_COL = "Sample"
TRUE_ENV_COL = "Env"
TRUE_ED_COL = "ED"

ED_HOURS = [0.5, 1, 2, 4, 8, 12, 24, 48, 72]

FIG_W = 8
FIG_H = 5.5
CMAP = "Blues"

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

def safe_mode(series):
    s = pd.Series(series).dropna().astype(str)
    if s.empty:
        return np.nan
    modes = s.mode()
    return modes.iloc[0] if len(modes) > 0 else s.iloc[0]

def compute_binary_metrics_ovr(y_true, y_pred, positive_class):
    y_true_bin = (np.asarray(y_true) == positive_class).astype(int)
    y_pred_bin = (np.asarray(y_pred) == positive_class).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ba  = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else np.nan
    f1  = f1_score(y_true_bin, y_pred_bin, zero_division=0)

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

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)

pls_model = saved["model"]
label_encoder = saved["label_encoder"]
feature_columns = saved["feature_columns"]
saved_class_names = saved.get("class_names", None)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.astype(str).str.strip()

required_cols = set(feature_columns) | {SAMPLE_COL, TRUE_ENV_COL, TRUE_ED_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

sample_env_counts = df.groupby(SAMPLE_COL)[TRUE_ENV_COL].nunique()
bad_env_samples = sample_env_counts[sample_env_counts > 1].index.tolist()
if bad_env_samples:
    raise ValueError(
        f"These samples have more than one {TRUE_ENV_COL}: {bad_env_samples[:10]}"
        + (" ..." if len(bad_env_samples) > 10 else "")
    )

sample_ed_counts = df.groupby(SAMPLE_COL)[TRUE_ED_COL].nunique()
bad_ed_samples = sample_ed_counts[sample_ed_counts > 1].index.tolist()
if bad_ed_samples:
    raise ValueError(
        f"These samples have more than one {TRUE_ED_COL}: {bad_ed_samples[:10]}"
        + (" ..." if len(bad_ed_samples) > 10 else "")
    )

X_new = df[feature_columns].to_numpy(dtype=float)

y_pred_numeric = np.argmax(pls_model.predict(X_new), axis=1)
y_pred_env = label_encoder.inverse_transform(y_pred_numeric)

df["Predicted_Env"] = y_pred_env

sample_df = (
    df.groupby(SAMPLE_COL, as_index=False)
      .agg({
          TRUE_ENV_COL: "first",
          TRUE_ED_COL: "first",
          "Predicted_Env": safe_mode
      })
)

y_true_env = sample_df[TRUE_ENV_COL].astype(str).values
y_pred_env = sample_df["Predicted_Env"].astype(str).values
y_true_ed = sample_df[TRUE_ED_COL].astype(str).values

if saved_class_names is not None:
    env_classes = [str(x) for x in saved_class_names]
else:
    env_classes = sorted(pd.Series(np.concatenate([y_true_env, y_pred_env])).unique(), key=natural_key)

ed_to_hour = build_ed_hour_mapping(y_true_ed, ED_HOURS)
sample_df["ED_Hour_Label"] = sample_df[TRUE_ED_COL].astype(str).map(ed_to_hour)
ordered_ed_hour_labels = [f"{h} h" for h in ED_HOURS]

contingency = pd.crosstab(
    index=sample_df["Predicted_Env"].astype(str),
    columns=sample_df["ED_Hour_Label"].astype(str),
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

global_per_class_df, global_summary_df = summarize_metrics_by_env_class(
    y_true=y_true_env,
    y_pred=y_pred_env,
    env_classes=env_classes,
    group_name="Global"
)

all_per_class_dfs = [global_per_class_df]
all_summary_dfs = [global_summary_df]

for ed_label in ordered_ed_hour_labels:
    sub = sample_df[sample_df["ED_Hour_Label"] == ed_label].copy()
    if sub.shape[0] == 0:
        continue

    ed_per_class_df, ed_summary_df = summarize_metrics_by_env_class(
        y_true=sub[TRUE_ENV_COL].astype(str).values,
        y_pred=sub["Predicted_Env"].astype(str).values,
        env_classes=env_classes,
        group_name=str(ed_label)
    )

    all_per_class_dfs.append(ed_per_class_df)
    all_summary_dfs.append(ed_summary_df)

per_class_metrics_df = pd.concat(all_per_class_dfs, axis=0, ignore_index=True)
summary_metrics_df = pd.concat(all_summary_dfs, axis=0, ignore_index=True)

misclassified = int(np.sum(y_true_env != y_pred_env))

print("\n" + "=" * 80)
print("SAMPLE-LEVEL RESULTS")
print("=" * 80)
print(f"Samples evaluated  : {len(sample_df)}")
print(f"Misclassifications : {misclassified}")
print(f"Accuracy           : {(len(sample_df) - misclassified) / len(sample_df):.4f}")

print("\nSample-level contingency table (Predicted Env vs True ED):")
print(contingency)

print("\nSample-level summary (mean ± SD across Env one-vs-rest metrics):")
print(summary_metrics_df.to_string(index=False))