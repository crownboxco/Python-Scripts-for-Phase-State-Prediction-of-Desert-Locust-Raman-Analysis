import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums, rankdata
from statsmodels.stats.multitest import multipletests

DATA_PATH = "CvI_pp.csv"
SAMPLE_COL = "Sample"
CLASS_COL = "Env"

X_AXIS_LABEL = "Environment"
BAND_LINES = [963, 1009, 1162, 1194, 1216, 1275, 1450, 1529]

# Optional desired class order for Env if you want control over plotting order.
# Leave as [] to use observed order.
ENV_ORDER = []

# Multiple-comparison p-value adjustment for pairwise Wilcoxon rank-sum tests
POSTHOC_P_ADJUST = "bonferroni"

# Bootstrap settings for 95% CI of median ranks
RANK_BOOT_N = 1000
RANK_BOOT_RANDOM_STATE = 1818

# Output files
BAND_STATS_XLSX = "BandLine_WilcoxonRankSum_Results.xlsx"
BAND_STATS_CSV = "BandLine_WilcoxonRankSum_Summary.csv"
BAND_RANKS_PNG = "BandLine_MedianRank_95CI_Env.png"

def nearest_x_index(x_axis, target):
    x_axis = np.asarray(x_axis, dtype=float)
    return int(np.argmin(np.abs(x_axis - float(target))))


def bootstrap_median_ci(values, n_boot=5000, alpha=0.05, random_state=1818):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    median_val = float(np.median(values))

    if len(values) == 1:
        return median_val, median_val, median_val

    rng = np.random.default_rng(random_state)
    boot_stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = np.median(sample)

    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return median_val, lower, upper


def clean_label(value):
    """
    Generic label cleaner. Keeps labels as strings and removes a leading 'Class'
    if present, since some datasets use labels like 'Class A' or 'Class 1'.
    """
    s = str(value).strip()
    s = re.sub(r'(?i)^class\s*', '', s).strip()
    return s


def adjust_pvalues(pvals, method="bonferroni"):
    """
    Adjust a list of p-values using statsmodels.multipletests.
    Supported common methods:
      - bonferroni
      - holm
      - fdr_bh
    """
    pvals = np.asarray(pvals, dtype=float)
    if len(pvals) == 0:
        return np.array([], dtype=float)

    _, pvals_adj, _, _ = multipletests(pvals, method=method)
    return pvals_adj

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

# Check that each sample belongs to only one class
sample_class_counts = df.groupby(SAMPLE_COL)[CLASS_COL].nunique()
bad_samples = sample_class_counts[sample_class_counts > 1].index.tolist()
if bad_samples:
    raise ValueError(
        f"These samples have more than one class and cannot be averaged safely: {bad_samples[:10]}"
        + (" ..." if len(bad_samples) > 10 else "")
    )

# Sample-level averaging
sample_df = (
    df.groupby(SAMPLE_COL)
      .agg({CLASS_COL: "first", **{c: "mean" for c in numeric_feature_cols}})
      .reset_index()
)

# Clean class labels
sample_df[CLASS_COL] = sample_df[CLASS_COL].apply(clean_label)

# Determine class order
observed_classes = sample_df[CLASS_COL].astype(str).unique().tolist()

if ENV_ORDER:
    class_names = [x for x in ENV_ORDER if x in observed_classes]
    unexpected_classes = [x for x in observed_classes if x not in ENV_ORDER]
    class_names += unexpected_classes
else:
    class_names = sorted(observed_classes)

# X-axis from column names if possible
try:
    x_axis = np.array(numeric_feature_cols, dtype=float)
except Exception:
    x_axis = np.arange(len(numeric_feature_cols), dtype=float)

band_summary_rows = []
wilcoxon_long_rows = []
rank_plot_rows = []

for band in BAND_LINES:
    peak_idx = nearest_x_index(x_axis, band)
    peak_x = float(x_axis[peak_idx])
    peak_var = numeric_feature_cols[peak_idx]

    peak_df = sample_df[[SAMPLE_COL, CLASS_COL, peak_var]].copy()
    peak_df = peak_df.rename(columns={peak_var: "Intensity", CLASS_COL: "Group"})
    peak_df["Group"] = peak_df["Group"].astype(str)

    valid_group_names = []
    group_values = {}

    for cls in class_names:
        vals = peak_df.loc[peak_df["Group"] == cls, "Intensity"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            valid_group_names.append(cls)
            group_values[cls] = vals

    if len(valid_group_names) < 2:
        print(f"\nSkipping band {band}: fewer than 2 non-empty groups.")
        continue

    pair_rows = []
    raw_pvals = []

    group_pairs = list(itertools.combinations(valid_group_names, 2))

    for g1, g2 in group_pairs:
        vals1 = group_values[g1]
        vals2 = group_values[g2]

        if len(vals1) == 0 or len(vals2) == 0:
            stat = np.nan
            pval = np.nan
        else:
            stat, pval = ranksums(vals1, vals2)

        pair_rows.append({
            "Band_Line_Requested": band,
            "Band_Line_Used": peak_x,
            "Variable": peak_var,
            "Group_1": g1,
            "Group_2": g2,
            "N_1": len(vals1),
            "N_2": len(vals2),
            "Wilcoxon_RankSum_Z": stat,
            "Wilcoxon_RankSum_p_raw": pval
        })
        raw_pvals.append(pval)

    # Adjust p-values within each band
    adj_pvals = adjust_pvalues(raw_pvals, method=POSTHOC_P_ADJUST)

    for row, p_adj in zip(pair_rows, adj_pvals):
        row["Wilcoxon_RankSum_p_adjusted"] = float(p_adj)
        row["P_Adjust_Method"] = POSTHOC_P_ADJUST
        wilcoxon_long_rows.append(row)

    # Summary row for each band
    min_raw_p = float(np.nanmin(raw_pvals)) if len(raw_pvals) else np.nan
    min_adj_p = float(np.nanmin(adj_pvals)) if len(adj_pvals) else np.nan
    n_sig = int(np.sum(np.asarray(adj_pvals) < 0.05)) if len(adj_pvals) else 0

    band_summary_rows.append({
        "Band_Line_Requested": band,
        "Band_Line_Used": peak_x,
        "Variable": peak_var,
        "N_Groups": len(valid_group_names),
        "N_Pairwise_Comparisons": len(group_pairs),
        "Min_Wilcoxon_p_raw": min_raw_p,
        "Min_Wilcoxon_p_adjusted": min_adj_p,
        "N_Significant_Pairs_Adjusted_p_lt_0.05": n_sig,
        "P_Adjust_Method": POSTHOC_P_ADJUST
    })

    print("\n" + "=" * 90)
    print(f"BAND LINE: requested={band} | used={peak_x:.2f} | variable={peak_var}")
    print("=" * 90)
    print(f"Pairwise Wilcoxon rank-sum tests ({POSTHOC_P_ADJUST}-adjusted)")
    print(pd.DataFrame(pair_rows).assign(
        Wilcoxon_RankSum_p_adjusted=adj_pvals
    ).to_string(index=False))

    peak_df = peak_df[peak_df["Group"].isin(valid_group_names)].copy()
    peak_df["Rank"] = rankdata(peak_df["Intensity"].to_numpy(dtype=float), method="average")

    for cls in valid_group_names:
        cls_ranks = peak_df.loc[peak_df["Group"] == cls, "Rank"].to_numpy(dtype=float)

        med_rank, ci_low, ci_high = bootstrap_median_ci(
            cls_ranks,
            n_boot=RANK_BOOT_N,
            alpha=0.05,
            random_state=RANK_BOOT_RANDOM_STATE + int(round(peak_x))
        )

        rank_plot_rows.append({
            "Band_Line_Requested": band,
            "Band_Line_Used": peak_x,
            "Variable": peak_var,
            "Group": cls,
            "N": len(cls_ranks),
            "Median_Rank": med_rank,
            "Rank_CI_Lower_95": ci_low,
            "Rank_CI_Upper_95": ci_high,
            "Min_Wilcoxon_p_adjusted": min_adj_p
        })

band_summary_df = pd.DataFrame(band_summary_rows)
wilcoxon_long_df = pd.DataFrame(wilcoxon_long_rows)
rank_plot_df = pd.DataFrame(rank_plot_rows)

if not rank_plot_df.empty:
    unique_bands_used = rank_plot_df["Band_Line_Used"].drop_duplicates().tolist()
    n_panels = len(unique_bands_used)

    # Force 2 rows x 4 columns
    nrows = 2
    ncols = 4

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10, 4.5),
        squeeze=False,
        constrained_layout=True
    )
    axes_flat = axes.flatten()

    for ax_idx, peak_x in enumerate(unique_bands_used):
        ax = axes_flat[ax_idx]

        sub = rank_plot_df[rank_plot_df["Band_Line_Used"] == peak_x].copy()
        sub["Group"] = pd.Categorical(sub["Group"], categories=class_names, ordered=True)
        sub = sub.sort_values("Group")

        y_pos = np.arange(len(sub))
        x = sub["Median_Rank"].to_numpy(dtype=float)
        xerr_lower = x - sub["Rank_CI_Lower_95"].to_numpy(dtype=float)
        xerr_upper = sub["Rank_CI_Upper_95"].to_numpy(dtype=float) - x

        ax.errorbar(
            x,
            y_pos,
            xerr=[xerr_lower, xerr_upper],
            fmt="o",
            capsize=4,
            linewidth=1.2,
            markersize=5
        )

        # Determine row/column position
        row_idx = ax_idx // ncols
        col_idx = ax_idx % ncols

        is_first_column = (col_idx == 0)
        is_bottom_row = (row_idx == nrows - 1)

        tick_labels = [str(val) for val in sub["Group"].astype(str)]

        # Y-axis ticks/labels: only first column
        ax.set_yticks(y_pos)
        if is_first_column:
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", length=0)
            ax.set_ylabel("")

        # X-axis ticks/labels: only bottom row
        if not is_bottom_row:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)
            ax.set_xlabel("")

        min_adj_p = sub["Min_Wilcoxon_p_adjusted"].iloc[0]
        ax.set_title(
            f"{peak_x:.0f} cm$^{{-1}}$ | Adj. p = {min_adj_p:.3g}",
            fontsize=9
        )
        ax.grid(alpha=0.25)
        ax.margins(x=0.08)

    # Remove unused panels if fewer than 8
    for extra_ax in axes_flat[n_panels:]:
        fig.delaxes(extra_ax)

    fig.supxlabel("Pooled Rank", fontsize=12)
    fig.supylabel(X_AXIS_LABEL, fontsize=12)
    # fig.savefig(BAND_RANKS_PNG, dpi=300, bbox_inches="tight")
    plt.show()

print(f"\nSaved Wilcoxon summary: {BAND_STATS_CSV}")
print(f"Saved full results workbook: {BAND_STATS_XLSX}")
print(f"Saved rank plot: {BAND_RANKS_PNG}")