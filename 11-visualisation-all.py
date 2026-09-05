"""
GROMACS Multi-Sequence, Multi-Replicate Analysis Visualiser
=============================================================
Visualises RMSD, Rg, SASA (time series) and RMSF (per-residue)
across N DNA sequences x M replicates, with separate handling for
each component (e.g. backbone, dna, protein) within each metric.

HOW TO USE
----------
1. Edit the CONFIG section below (paths) and, if needed, `parse_metadata()`
   to match your folder / file naming convention.
2. Run: python visualise_gromacs.py
3. Plots are written to ./figures/

Expected folder layout (edit `parse_metadata()` if yours differs):

    data/
      <sequence>/
        <replicate>/
          analysis/
            rmsd_backbone.xvg   rmsd_dna.xvg
            rg_backbone.xvg     rg_dna.xvg
            rmsf_backbone.xvg   rmsf_dna.xvg
            sasa_backbone.xvg   sasa_dna.xvg   sasa_protein.xvg

    e.g. data/APC_637_hot/1/analysis/rmsd_backbone.xvg

Components are auto-discovered per metric from whatever "<metric>_<component>.xvg"
files are present, so it doesn't matter that SASA has 3 components (backbone/dna/
protein) while RMSD/Rg/RMSF have 2 (backbone/dna) - each gets its own figure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- CONFIG ---------------------------------
# Soft-coded paths, mirroring curves-seq.sh: this script is expected to live
# in a "scripts" folder that sits next to "output" (scripts/../output), so
# paths are derived from the script's own location - this works no matter
# what directory you run it from, and needs no editing per-machine/per-run.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent                     # the level containing both scripts/ and output/
DATA_DIR = BASE_DIR / "output"
OUT_DIR = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_METRICS = {
    "rmsd": {"ylabel": "RMSD (nm)"},
    "rg":   {"ylabel": "Radius of gyration (nm)"},
    "sasa": {"ylabel": "SASA (nm$^2$)"},
    # rmsf is handled separately below - it's per-residue, not a time series
}

TIME_UNIT_DIVISOR = {"rmsd": 1, "rg": 1000, "sasa": 1000}
# ^ metric-specific, NOT a single shared constant. 10-analysis.sh runs
# `gmx rms` with -tu ns, so RMSD's raw .xvg time column is already in ns
# (divisor 1 = no further conversion needed). `gmx gyrate` and `gmx sasa`
# get no -tu flag, so GROMACS defaults those to ps (divisor 1000 to get ns).
# Using one uniform divisor for every metric silently re-divided RMSD's
# already-in-ns values by another 1000, compressing a 300 ns axis to 0.3.

# fraction of the trajectory (from the end) considered "equilibrated" -
# used only for the summary box/violin plots
EQUIL_FRACTION = 0.5

sns.set_theme(style="whitegrid", context="talk")


# ------------------------- FILE DISCOVERY ------------------------------

def parse_metadata(filepath: Path):
    """
    Extract (sequence, replicate, metric, component) from a file path.
    EDIT THIS if your naming convention differs.

    Assumes: data/<sequence>/<replicate>/analysis/<metric>_<component>.xvg
    e.g.     data/APC_637_hot/1/analysis/rmsd_backbone.xvg
    """
    replicate = filepath.parent.parent.name          # e.g. "1"
    sequence = filepath.parent.parent.parent.name     # e.g. "APC_637_hot"

    stem = filepath.stem.lower()                      # e.g. "rmsd_backbone"
    metric, _, component = stem.partition("_")        # "rmsd", "backbone"

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
    for fp in DATA_DIR.rglob("*.xvg"):
        sequence, replicate, metric, component = parse_metadata(fp)
        if metric not in TIMESERIES_METRICS:
            continue
        arr = read_xvg(fp)
        if arr.size == 0:
            print(f"WARNING: no data parsed from {fp}")
            continue
        time = arr[:, 0] / TIME_UNIT_DIVISOR[metric]
        value = arr[:, 1]     # first data column after time; change index
                               # if your xvg has the quantity you want in a
                               # different column (e.g. gmx gyrate writes
                               # total Rg, Rg_x, Rg_y, Rg_z as cols 1-4)
        records.extend(zip([sequence]*len(time), [replicate]*len(time),
                            [metric]*len(time), [component]*len(time), time, value))
    return pd.DataFrame(records, columns=["sequence", "replicate", "metric", "component", "time", "value"])


def load_rmsf() -> pd.DataFrame:
    records = []
    for fp in DATA_DIR.rglob("rmsf_*.xvg"):
        sequence, replicate, _, component = parse_metadata(fp)
        arr = read_xvg(fp)
        if arr.size == 0:
            continue
        residue, value = arr[:, 0], arr[:, 1]
        records.extend(zip([sequence]*len(residue), [replicate]*len(residue),
                            [component]*len(residue), residue, value))
    return pd.DataFrame(records, columns=["sequence", "replicate", "component", "residue", "value"])


# --------------------------- PLOTTING ------------------------------------

def _palette(n):
    return sns.color_palette("tab20", n) if n <= 20 else sns.color_palette("husl", n)


def plot_timeseries(df: pd.DataFrame, metric: str, component: str, info: dict):
    """One figure per metric+component: mean +/- std across replicates per
       sequence, with faint individual replicate traces underneath so
       outliers or convergence issues are still visible."""
    sub = df[(df.metric == metric) & (df.component == component)]
    sequences = sorted(sub.sequence.unique())
    palette = _palette(len(sequences))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for colour, seq in zip(palette, sequences):
        seq_data = sub[sub.sequence == seq]

        for _, rep_data in seq_data.groupby("replicate"):
            ax.plot(rep_data.time, rep_data.value, color=colour, alpha=0.15, lw=0.8)

        pivot = seq_data.pivot_table(index="time", columns="replicate", values="value").sort_index()
        mean, std = pivot.mean(axis=1), pivot.std(axis=1)
        ax.plot(mean.index, mean.values, color=colour, lw=2, label=seq)
        ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Time (ns)")  # every metric is now correctly in ns after the per-metric divisor above
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{metric.upper()} ({component}) \u2014 mean \u00b1 std across replicates")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Sequence")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{metric}_{component}_timeseries.png", dpi=300)
    fig.savefig(OUT_DIR / f"{metric}_{component}_timeseries.pdf")
    plt.close(fig)


def plot_rmsf(rmsf_df: pd.DataFrame, component: str):
    sub = rmsf_df[rmsf_df.component == component]
    sequences = sorted(sub.sequence.unique())
    palette = _palette(len(sequences))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for colour, seq in zip(palette, sequences):
        seq_data = sub[sub.sequence == seq]
        mean = seq_data.groupby("residue").value.mean()
        std = seq_data.groupby("residue").value.std()
        ax.plot(mean.index, mean.values, color=colour, lw=1.5, label=seq)
        ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (nm)")
    ax.set_title(f"RMSF ({component}) \u2014 mean \u00b1 std across replicates")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Sequence")
    
    if component == "dna":
        LESION_RESIDUE = 8
        ax.axvline(LESION_RESIDUE, color="black", linestyle="--", lw=1.5, alpha=0.8, zorder=5)
        ax.annotate("lesion", xy=(LESION_RESIDUE, ax.get_ylim()[1] * 0.98),
            xytext=(5, 0), textcoords="offset points",
            color="black", fontsize=12, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"rmsf_{component}_perresidue.png", dpi=300)
    fig.savefig(OUT_DIR / f"rmsf_{component}_perresidue.pdf")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, metric: str, component: str, info: dict):
    """Box+strip plot comparing sequences using only the 'equilibrated' tail
       of each trajectory -> one point per replicate per sequence. Good for
       spotting which sequence(s) behave differently."""
    sub = df[(df.metric == metric) & (df.component == component)]
    cutoff = sub.groupby(["sequence", "replicate"]).time.transform(
        lambda t: t.max() * (1 - EQUIL_FRACTION)
    )
    equil = sub[sub.time >= cutoff]
    per_rep_mean = equil.groupby(["sequence", "replicate"], as_index=False).value.mean()
    sequences = sorted(per_rep_mean.sequence.unique())
    palette = _palette(len(sequences))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.boxplot(data=per_rep_mean, x="sequence", y="value", order=sequences,
                hue="sequence", legend=False,
                showfliers=False, palette=palette, ax=ax)
    sns.stripplot(data=per_rep_mean, x="sequence", y="value", order=sequences,
                  color="black", size=5, alpha=0.7, ax=ax)
    ax.set_xlabel("Sequence")
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{metric.upper()} ({component}) \u2014 equilibrated average per replicate "
                 f"(last {int(EQUIL_FRACTION*100)}% of trajectory)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{metric}_{component}_summary_boxplot.png", dpi=300)
    plt.close(fig)


# ------------------------------ MAIN --------------------------------------

if __name__ == "__main__":
    print(f"Looking for data in: {DATA_DIR.resolve()}")
    all_xvg = list(DATA_DIR.rglob("*.xvg"))
    print(f"Found {len(all_xvg)} .xvg files total")

    print("Loading time-series data (RMSD, Rg, SASA)...")
    ts_df = load_timeseries()
    if ts_df.empty:
        msg = [f"No time-series data loaded from {DATA_DIR.resolve()}."]
        if not DATA_DIR.exists():
            msg.append(f"That directory does not exist - check BASE_DIR/DATA_DIR above.")
        elif len(all_xvg) == 0:
            msg.append("The directory exists but contains no .xvg files anywhere under it.")
        else:
            msg.append(
                f"{len(all_xvg)} .xvg files were found, but none parsed into a known metric. "
                f"First few paths found:\n  " + "\n  ".join(str(p) for p in all_xvg[:5]) +
                "\nCheck these match the folder depth parse_metadata() expects "
                "(sequence/replicate/analysis/metric_component.xvg)."
            )
        raise SystemExit("\n".join(msg))
    print(f"  {len(ts_df)} rows | sequences: {ts_df.sequence.nunique()} | "
          f"replicates: {ts_df.replicate.nunique()} | metrics found: {sorted(ts_df.metric.unique())}")

    print("Loading RMSF data...")
    rmsf_df = load_rmsf()
    print(f"  {len(rmsf_df)} rows")

    for metric, info in TIMESERIES_METRICS.items():
        sub = ts_df[ts_df.metric == metric]
        if sub.empty:
            print(f"NOTE: no files found for metric '{metric}' - skipped.")
            continue
        for component in sorted(sub.component.unique()):
            print(f"Plotting {metric} ({component}) time series...")
            plot_timeseries(ts_df, metric, component, info)
            print(f"Plotting {metric} ({component}) summary boxplot...")
            plot_summary(ts_df, metric, component, info)

    if not rmsf_df.empty:
        for component in sorted(rmsf_df.component.unique()):
            print(f"Plotting RMSF ({component})...")
            plot_rmsf(rmsf_df, component)
    else:
        print("NOTE: no rmsf_*.xvg files found - skipped.")

    print(f"\nDone. Figures saved to: {OUT_DIR.resolve()}")