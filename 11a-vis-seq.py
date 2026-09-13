"""
GROMACS Single-Sequence Replicate Visualiser
=============================================================
Visualises RMSD, Rg, SASA (time series) and RMSF (per-residue) for the
3 replicates of ONE sequence, passed as a command-line argument -
mirrors the usage pattern of curves-seq.sh.

USAGE
-----
    python visualise_gromacs.py <sequence_name>

    Example:
    python visualise_gromacs.py APC_637_hot

Paths are soft-coded like curves-seq.sh: this script is expected to live in
a "scripts" folder that sits next to "output" (scripts/../output), so it
works from any working directory without editing.

Expected folder layout (edit parse_metadata() if yours differs):

    output/
      <sequence>/
        <replicate>/
          analysis/
            rmsd_backbone.xvg   rmsd_dna.xvg
            rg_backbone.xvg     rg_dna.xvg
            rmsf_backbone.xvg   rmsf_dna.xvg
            sasa_backbone.xvg   sasa_dna.xvg   sasa_protein.xvg

    e.g. output/APC_637_hot/1/analysis/rmsd_backbone.xvg

Components (backbone/dna/protein) are auto-discovered per metric from
whatever "<metric>_<component>.xvg" files exist, and each gets its own
figure. Replicate folders are also auto-discovered under the sequence
folder (not hardcoded to 1/2/3), matching however many you actually have.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- CONFIG ---------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent                      # level containing both scripts/ and output/
DATA_DIR = BASE_DIR / "output"
FIGURES_ROOT = DATA_DIR / "figures"

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
EQUIL_FRACTION = 0.5       # fraction of trajectory (from the end) considered "equilibrated"

sns.set_theme(style="whitegrid", context="talk")
REP_PALETTE = sns.color_palette("Set1", 8)  # supports up to 8 replicates distinctly


# ------------------------- ARGUMENT HANDLING ----------------------------

def get_sequence_arg() -> str:
    if len(sys.argv) < 2:
        print("ERROR: No sequence name provided!", file=sys.stderr)
        print(f"Usage: python {Path(__file__).name} <sequence_name>", file=sys.stderr)
        print("Example: python visualise_gromacs.py APC_637_hot", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1]


# ------------------------- FILE DISCOVERY ------------------------------

def parse_metadata(filepath: Path):
    """
    Extract (replicate, metric, component) from a file path within a single
    sequence's folder. EDIT THIS if your naming convention differs.

    Assumes: output/<sequence>/<replicate>/analysis/<metric>_<component>.xvg
    """
    replicate = filepath.parent.parent.name           # e.g. "1"
    stem = filepath.stem.lower()                       # e.g. "rmsd_backbone"
    metric, _, component = stem.partition("_")          # "rmsd", "backbone"
    return replicate, metric, component


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


# ------------------------- LOAD ONE SEQUENCE'S DATA ------------------------

def load_timeseries(seq_dir: Path) -> pd.DataFrame:
    records = []
    for fp in seq_dir.rglob("*.xvg"):
        replicate, metric, component = parse_metadata(fp)
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
        records.extend(zip([replicate]*len(time), [metric]*len(time),
                            [component]*len(time), time, value))
    return pd.DataFrame(records, columns=["replicate", "metric", "component", "time", "value"])


def load_rmsf(seq_dir: Path) -> pd.DataFrame:
    records = []
    for fp in seq_dir.rglob("rmsf_*.xvg"):
        replicate, _, component = parse_metadata(fp)
        arr = read_xvg(fp)
        if arr.size == 0:
            continue
        residue, value = arr[:, 0], arr[:, 1]
        records.extend(zip([replicate]*len(residue), [component]*len(residue), residue, value))
    return pd.DataFrame(records, columns=["replicate", "component", "residue", "value"])


# --------------------------- PLOTTING ------------------------------------

def plot_timeseries(df: pd.DataFrame, metric: str, component: str, seq: str, info: dict, out_dir: Path):
    """One figure per metric+component: each replicate as its own coloured
       line (so a diverging replicate is immediately visible), plus a bold
       black dashed mean line across replicates."""
    sub = df[(df.metric == metric) & (df.component == component)]
    replicates = sorted(sub.replicate.unique())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for colour, rep in zip(REP_PALETTE, replicates):
        rep_data = sub[sub.replicate == rep].sort_values("time")
        ax.plot(rep_data.time, rep_data.value, color=colour, lw=1.2, alpha=0.85, label=f"rep {rep}")

    pivot = sub.pivot_table(index="time", columns="replicate", values="value").sort_index()

    ax.set_xlabel("Time (ns)")  # every metric is now correctly in ns after the per-metric divisor above
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{metric.upper()} ({component}) \u2014 {seq}")
    ax.legend(fontsize=9, title="Replicate")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_{component}_timeseries.png", dpi=300)
    fig.savefig(out_dir / f"{metric}_{component}_timeseries.pdf")
    plt.close(fig)


def plot_rmsf(rmsf_df: pd.DataFrame, component: str, seq: str, out_dir: Path):
    sub = rmsf_df[rmsf_df.component == component]
    replicates = sorted(sub.replicate.unique())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for colour, rep in zip(REP_PALETTE, replicates):
        rep_data = sub[sub.replicate == rep].sort_values("residue")
        ax.plot(rep_data.residue, rep_data.value, color=colour, lw=1.5, label=f"rep {rep}")

    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (nm)")
    ax.set_title(f"RMSF ({component}) \u2014 {seq}")
    ax.legend(fontsize=9, title="Replicate")
    fig.tight_layout()
    fig.savefig(out_dir / f"rmsf_{component}_perresidue.png", dpi=300)
    fig.savefig(out_dir / f"rmsf_{component}_perresidue.pdf")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, metric: str, component: str, seq: str, info: dict, out_dir: Path):
    """Box+strip plot of the equilibrated tail, one box per replicate -
       shows how much each replicate's distribution differs, and whether
       any replicate hasn't converged to the same range as the others."""
    sub = df[(df.metric == metric) & (df.component == component)]
    cutoff = sub.groupby("replicate").time.transform(lambda t: t.max() * (1 - EQUIL_FRACTION))
    equil = sub[sub.time >= cutoff]
    replicates = sorted(equil.replicate.unique())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.boxplot(data=equil, x="replicate", y="value", order=replicates,
                hue="replicate", legend=False, showfliers=False,
                palette=REP_PALETTE[:len(replicates)], ax=ax)
    ax.set_xlabel("Replicate")
    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{metric.upper()} ({component}) \u2014 {seq}\n"
                 f"equilibrated distribution (last {int(EQUIL_FRACTION*100)}% of trajectory)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_{component}_summary_boxplot.png", dpi=300)
    plt.close(fig)


# ------------------------------ MAIN --------------------------------------

if __name__ == "__main__":
    SEQ = get_sequence_arg()
    seq_dir = DATA_DIR / SEQ

    if not seq_dir.exists():
        print(f"ERROR: {seq_dir.resolve()} does not exist.", file=sys.stderr)
        available = [p.name for p in DATA_DIR.iterdir() if p.is_dir()] if DATA_DIR.exists() else []
        if available:
            print(f"Sequences found under {DATA_DIR.resolve()}: {sorted(available)}", file=sys.stderr)
        else:
            print(f"{DATA_DIR.resolve()} itself does not exist or has no subfolders - check DATA_DIR.", file=sys.stderr)
        sys.exit(1)

    # warn (don't fail) on missing replicate/analysis folders, like curves-seq.sh does
    for rep_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
        if not (rep_dir / "analysis").exists():
            print(f"Warning: {rep_dir / 'analysis'} does not exist. Skipping replicate {rep_dir.name}...")

    out_dir = FIGURES_ROOT / SEQ
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sequence: {SEQ}")
    print(f"Looking for data in: {seq_dir.resolve()}")

    ts_df = load_timeseries(seq_dir)
    if ts_df.empty:
        raise SystemExit(
            f"No time-series data loaded from {seq_dir.resolve()}. "
            f"Check parse_metadata() matches your folder layout."
        )
    print(f"  {len(ts_df)} rows | replicates: {sorted(ts_df.replicate.unique())} | "
          f"metrics found: {sorted(ts_df.metric.unique())}")

    rmsf_df = load_rmsf(seq_dir)
    print(f"RMSF rows: {len(rmsf_df)}")

    for metric, info in TIMESERIES_METRICS.items():
        sub = ts_df[ts_df.metric == metric]
        if sub.empty:
            print(f"NOTE: no files found for metric '{metric}' - skipped.")
            continue
        for component in sorted(sub.component.unique()):
            print(f"Plotting {metric} ({component}) time series...")
            plot_timeseries(ts_df, metric, component, SEQ, info, out_dir)
            print(f"Plotting {metric} ({component}) summary boxplot...")
            plot_summary(ts_df, metric, component, SEQ, info, out_dir)

    if not rmsf_df.empty:
        for component in sorted(rmsf_df.component.unique()):
            print(f"Plotting RMSF ({component})...")
            plot_rmsf(rmsf_df, component, SEQ, out_dir)
    else:
        print("NOTE: no rmsf_*.xvg files found - skipped.")

    print(f"\nDone. Figures saved to: {out_dir.resolve()}")