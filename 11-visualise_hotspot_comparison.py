"""
GROMACS Hotspot vs Non-Hotspot Comparison Visualiser
=============================================================
Same underlying metrics as visualise_gromacs.py (RMSD, Rg, SASA time series
and RMSF per-residue), but processes ALL sequences in one run and groups
them by hotspot vs non-hotspot status (from the "_hot" / "_non" suffix on
each sequence folder name), plus adds direct group-comparison plots: mean
of each sequence's average value, hotspot vs non-hotspot, with a paired
t-test (your 12 sequences form 6 matched hotspot/non-hotspot pairs, so a
paired test is the statistically appropriate one - edit STAT_TEST below if
you want an unpaired/independent t-test instead).

USAGE
-----
    python visualise_hotspot_comparison.py

Paths are soft-coded like curves-seq.sh: this script is expected to live in
a "scripts" folder that sits next to "output" (scripts/../output).

Expected folder layout:

    output/
      <pair_prefix>_hot/
        <replicate>/analysis/<metric>_<component>.xvg
      <pair_prefix>_non/
        <replicate>/analysis/<metric>_<component>.xvg

    e.g. output/APC_637_hot/1/analysis/rmsd_backbone.xvg
         output/APC_637_non/1/analysis/rmsd_backbone.xvg

    -> these two are treated as a matched pair ("APC_637") for the paired
       t-test, since they share everything before the _hot/_non suffix.

Components (backbone/dna/protein) are auto-discovered per metric, same as
visualise_gromacs.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------- CONFIG ---------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "output"                                  # <-- change if named differently
OUT_DIR = BASE_DIR / "figures" / "hotspot_vs_nonhotspot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_METRICS = {
    "rmsd": {"ylabel": "RMSD (nm)"},
    "rg":   {"ylabel": "Radius of gyration (nm)"},
    "sasa": {"ylabel": "SASA (nm$^2$)"},
    # rmsf handled separately - it's per-residue, not a time series
}

TIME_UNIT_DIVISOR = 1000   # ps -> ns; set to 1 to keep ps as-is
EQUIL_FRACTION = 0.5       # fraction of trajectory (from the end) considered "equilibrated"

GROUP_ORDER = ["hotspot", "non-hotspot"]
GROUP_COLOURS = {"hotspot": "#d62728", "non-hotspot": "#1f77b4"}  # red / blue

STAT_TEST = "paired"       # "paired" (scipy ttest_rel) or "unpaired" (scipy ttest_ind)

sns.set_theme(style="whitegrid", context="talk")


# ------------------------- CLASSIFICATION ------------------------------

def classify_sequence(sequence: str):
    """
    Split a sequence folder name into (group, pair_id) from its _hot/_non
    suffix. EDIT HERE if your naming convention changes.
    Returns (None, None) for anything that doesn't match, so it can be
    skipped and reported rather than silently misclassified.
    """
    if sequence.endswith("_hot"):
        return "hotspot", sequence[: -len("_hot")]
    elif sequence.endswith("_non"):
        return "non-hotspot", sequence[: -len("_non")]
    return None, None


# ------------------------- FILE DISCOVERY ------------------------------

def parse_metadata(filepath: Path):
    """
    Extract (sequence, replicate, metric, component) from a file path.
    Assumes: output/<sequence>/<replicate>/analysis/<metric>_<component>.xvg
    """
    replicate = filepath.parent.parent.name
    sequence = filepath.parent.parent.parent.name
    stem = filepath.stem.lower()
    metric, _, component = stem.partition("_")
    return sequence, replicate, metric, component


def read_xvg(filepath: Path) -> np.ndarray:
    """Read a GROMACS .xvg file, ignoring comment (#) and grace (@) lines."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows) if rows else np.empty((0, 0))


# ------------------------- LOAD ALL DATA --------------------------------

def load_timeseries() -> pd.DataFrame:
    records = []
    skipped = set()
    for fp in DATA_DIR.rglob("*.xvg"):
        sequence, replicate, metric, component = parse_metadata(fp)
        if metric not in TIMESERIES_METRICS:
            continue
        group, pair_id = classify_sequence(sequence)
        if group is None:
            skipped.add(sequence)
            continue
        arr = read_xvg(fp)
        if arr.size == 0:
            print(f"WARNING: no data parsed from {fp}")
            continue
        time = arr[:, 0] / TIME_UNIT_DIVISOR
        value = arr[:, 1]     # first data column after time; adjust index
                               # if you need a different column
        records.extend(zip(
            [sequence]*len(time), [group]*len(time), [pair_id]*len(time),
            [replicate]*len(time), [metric]*len(time), [component]*len(time),
            time, value
        ))
    if skipped:
        print(f"NOTE: skipped sequences not matching _hot/_non naming: {sorted(skipped)}")
    return pd.DataFrame(records, columns=[
        "sequence", "group", "pair_id", "replicate", "metric", "component", "time", "value"
    ])


def load_rmsf() -> pd.DataFrame:
    records = []
    for fp in DATA_DIR.rglob("rmsf_*.xvg"):
        sequence, replicate, _, component = parse_metadata(fp)
        group, pair_id = classify_sequence(sequence)
        if group is None:
            continue
        arr = read_xvg(fp)
        if arr.size == 0:
            continue
        residue, value = arr[:, 0], arr[:, 1]
        records.extend(zip(
            [sequence]*len(residue), [group]*len(residue), [pair_id]*len(residue),
            [replicate]*len(residue), [component]*len(residue), residue, value
        ))
    return pd.DataFrame(records, columns=[
        "sequence", "group", "pair_id", "replicate", "component", "residue", "value"
    ])


# --------------------------- GROUP TIME SERIES ----------------------------

def plot_group_timeseries(df: pd.DataFrame, metric: str, component: str, info: dict):
    """Group-level time series: bold mean +/- std across the 6 sequences in
       each group (hotspot / non-hotspot), with faint per-sequence mean
       lines (averaged over that sequence's 3 replicates) underneath."""
    sub = df[(df.metric == metric) & (df.component == component)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for group in GROUP_ORDER:
        g = sub[sub.group == group]
        if g.empty:
            continue
        colour = GROUP_COLOURS[group]

        seq_means = g.groupby(["sequence", "time"], as_index=False).value.mean()
        for _, seq_data in seq_means.groupby("sequence"):
            ax.plot(seq_data.time, seq_data.value, color=colour, alpha=0.15, lw=0.8)

        pivot = seq_means.pivot_table(index="time", columns="sequence", values="value").sort_index()
        mean, std = pivot.mean(axis=1), pivot.std(axis=1)
        ax.plot(mean.index, mean.values, color=colour, lw=2.5, label=f"{group} (n={pivot.shape[1]} seq)")
        ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Time (ns)" if TIME_UNIT_DIVISOR == 1000 else "Time (ps)")
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{metric.upper()} ({component}) \u2014 hotspot vs non-hotspot\n"
                 f"(band = std across the 6 sequences per group)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{metric}_{component}_group_timeseries.png", dpi=300)
    fig.savefig(OUT_DIR / f"{metric}_{component}_group_timeseries.pdf")
    plt.close(fig)


def plot_group_rmsf(rmsf_df: pd.DataFrame, component: str):
    sub = rmsf_df[rmsf_df.component == component]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for group in GROUP_ORDER:
        g = sub[sub.group == group]
        if g.empty:
            continue
        colour = GROUP_COLOURS[group]

        seq_means = g.groupby(["sequence", "residue"], as_index=False).value.mean()
        for _, seq_data in seq_means.groupby("sequence"):
            ax.plot(seq_data.residue, seq_data.value, color=colour, alpha=0.15, lw=0.8)

        pivot = seq_means.pivot_table(index="residue", columns="sequence", values="value").sort_index()
        mean, std = pivot.mean(axis=1), pivot.std(axis=1)
        ax.plot(mean.index, mean.values, color=colour, lw=2.5, label=f"{group} (n={pivot.shape[1]} seq)")
        ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (nm)")
    ax.set_title(f"RMSF ({component}) \u2014 hotspot vs non-hotspot")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"rmsf_{component}_group_perresidue.png", dpi=300)
    fig.savefig(OUT_DIR / f"rmsf_{component}_group_perresidue.pdf")
    plt.close(fig)


# --------------------- PER-SEQUENCE AVERAGES + GROUP COMPARISON -------------

def compute_sequence_averages(ts_df: pd.DataFrame, rmsf_df: pd.DataFrame) -> pd.DataFrame:
    """One scalar per sequence per metric+component:
       - time-series metrics: mean over the equilibrated tail, across replicates
       - rmsf: mean across all residues, across replicates
    """
    pieces = []

    for (metric, component), sub in ts_df.groupby(["metric", "component"]):
        cutoff = sub.groupby("sequence").time.transform(lambda t: t.max() * (1 - EQUIL_FRACTION))
        equil = sub[sub.time >= cutoff]
        seq_avg = equil.groupby(["sequence", "group", "pair_id"], as_index=False).value.mean()
        seq_avg["metric"] = metric
        seq_avg["component"] = component
        pieces.append(seq_avg)

    for component, sub in rmsf_df.groupby("component"):
        seq_avg = sub.groupby(["sequence", "group", "pair_id"], as_index=False).value.mean()
        seq_avg["metric"] = "rmsf"
        seq_avg["component"] = component
        pieces.append(seq_avg)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_stat_test(hot_vals, non_vals):
    if STAT_TEST == "paired":
        return stats.ttest_rel(hot_vals, non_vals)
    return stats.ttest_ind(hot_vals, non_vals)


def plot_group_comparison(seq_avg_df: pd.DataFrame, metric: str, component: str, ylabel: str):
    """Mean of each group's per-sequence averages (hotspot vs non-hotspot),
       with individual sequence points, lines connecting matched pairs, and
       a paired t-test across the 6 pairs."""
    sub = seq_avg_df[(seq_avg_df.metric == metric) & (seq_avg_df.component == component)]
    hot = sub[sub.group == "hotspot"].set_index("pair_id").value
    non = sub[sub.group == "non-hotspot"].set_index("pair_id").value
    common_pairs = sorted(set(hot.index) & set(non.index))

    if len(common_pairs) < 2:
        print(f"WARNING: fewer than 2 matched pairs for {metric} ({component}) - skipping stats")
        p_val = np.nan
    else:
        hot_vals = hot.loc[common_pairs].values
        non_vals = non.loc[common_pairs].values
        _, p_val = run_stat_test(hot_vals, non_vals)

    means = sub.groupby("group").value.mean().reindex(GROUP_ORDER)
    sems = sub.groupby("group").value.sem().reindex(GROUP_ORDER)
    x_pos = np.arange(len(GROUP_ORDER))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.bar(x_pos, means.values, yerr=sems.values, capsize=6,
           color=[GROUP_COLOURS[g] for g in GROUP_ORDER], alpha=0.55, width=0.5, zorder=2)

    for pid in common_pairs:
        ax.plot([0, 1], [hot.loc[pid], non.loc[pid]], color="grey", lw=1, alpha=0.6, zorder=3)
    ax.scatter([0]*len(hot), hot.values, color=GROUP_COLOURS["hotspot"],
               edgecolor="black", zorder=4, label="hotspot seq.")
    ax.scatter([1]*len(non), non.values, color=GROUP_COLOURS["non-hotspot"],
               edgecolor="black", zorder=4, label="non-hotspot seq.")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel(ylabel)

    title = f"{metric.upper()} ({component}) \u2014 mean of per-sequence averages"
    if not np.isnan(p_val):
        y_max = max(hot.max(), non.max())
        y_bracket = y_max * 1.08
        ax.plot([0, 0, 1, 1], [y_bracket, y_bracket*1.02, y_bracket*1.02, y_bracket], color="black", lw=1)
        stars = "n.s."
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"
        test_label = "paired t-test" if STAT_TEST == "paired" else "unpaired t-test"
        ax.text(0.5, y_bracket*1.03, f"{test_label} p={p_val:.4f} ({stars})", ha="center", fontsize=10)
        ax.set_ylim(top=y_bracket*1.18)

    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{metric}_{component}_group_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / f"{metric}_{component}_group_comparison.pdf")
    plt.close(fig)


# ------------------------------ MAIN --------------------------------------

if __name__ == "__main__":
    print(f"Looking for data in: {DATA_DIR.resolve()}")
    ts_df = load_timeseries()
    if ts_df.empty:
        raise SystemExit(
            f"No time-series data loaded from {DATA_DIR.resolve()}. "
            f"Check DATA_DIR and parse_metadata()/classify_sequence() match your layout."
        )
    n_hot = ts_df[ts_df.group == "hotspot"].sequence.nunique()
    n_non = ts_df[ts_df.group == "non-hotspot"].sequence.nunique()
    print(f"  {len(ts_df)} rows | {n_hot} hotspot sequences, {n_non} non-hotspot sequences | "
          f"metrics found: {sorted(ts_df.metric.unique())}")

    rmsf_df = load_rmsf()
    print(f"RMSF rows: {len(rmsf_df)}")

    print("\n--- Group time series (all sequences, split hotspot vs non-hotspot) ---")
    for metric, info in TIMESERIES_METRICS.items():
        sub = ts_df[ts_df.metric == metric]
        if sub.empty:
            print(f"NOTE: no files found for metric '{metric}' - skipped.")
            continue
        for component in sorted(sub.component.unique()):
            print(f"Plotting {metric} ({component}) group time series...")
            plot_group_timeseries(ts_df, metric, component, info)

    if not rmsf_df.empty:
        for component in sorted(rmsf_df.component.unique()):
            print(f"Plotting RMSF ({component}) group plot...")
            plot_group_rmsf(rmsf_df, component)
    else:
        print("NOTE: no rmsf_*.xvg files found - skipped.")

    print("\n--- Group comparison (mean of per-sequence averages + stat test) ---")
    seq_avg_df = compute_sequence_averages(ts_df, rmsf_df)
    for metric, info in TIMESERIES_METRICS.items():
        sub = seq_avg_df[seq_avg_df.metric == metric]
        for component in sorted(sub.component.unique()):
            print(f"Plotting {metric} ({component}) group comparison...")
            plot_group_comparison(seq_avg_df, metric, component, info["ylabel"])

    rmsf_avg = seq_avg_df[seq_avg_df.metric == "rmsf"]
    for component in sorted(rmsf_avg.component.unique()):
        print(f"Plotting RMSF ({component}) group comparison...")
        plot_group_comparison(seq_avg_df, "rmsf", component, "RMSF (nm)")

    print(f"\nDone. Figures saved to: {OUT_DIR.resolve()}")
