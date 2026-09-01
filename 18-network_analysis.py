#!/usr/bin/env python3
"""
Whole-network connectivity (hub residue) analysis for NTHL1-DNA RING contact data.

Unlike analyse_ring_contacts.py (which only looks at contacts touching the damaged
base), this script looks at EVERY residue's contact network: how many distinct
partners does each residue have, across the whole protein-DNA complex? This finds
"hub" residues that may not touch the lesion at all but are still structurally or
functionally important - and checks whether any of them differ between hotspot and
non-hotspot sequence contexts.

Input layout (same as analyse_ring_contacts.py):
    RING/
      APC_637_hot/
        APC_637_1_contacts.tsv
        APC_637_2_contacts.tsv
        APC_637_3_contacts.tsv
      APC_641_non/
        ...

Usage:
    python analyse_ring_network.py                       (defaults to ../RING next to this script)
    python analyse_ring_network.py /path/to/RING          (explicit override, if ever needed)

Outputs:
    Written into EACH sequence subfolder (e.g. RING/APC_637_hot/):
        APC_637_residue_degree.csv     one row per residue, averaged across replicates

    Written into the RING folder itself (or --out if given):
        all_residue_degree_long.csv       every residue's degree, every sequence, every replicate
        universal_hub_residues.csv        residues appearing with substantial degree in ALL sequences
        hub_hotspot_vs_nonhotspot_stats.csv   paired Wilcoxon test per universal hub residue
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Configuration - shared with analyse_ring_contacts.py, kept in sync manually
# ---------------------------------------------------------------------------

HOTSPOT_MAP = {
    'APC_637': 1, 'APC_641': 0, 'APC_3335': 0, 'APC_3340': 1,
    'APC_4099': 1, 'APC_4103': 0, 'APC_4343': 0, 'APC_4348': 1,
    'TP53_632': 0, 'TP53_637': 1, 'TP53_844': 1, 'TP53_849': 0,
}

SEQUENCE_PAIRS = [
    ('APC_637', 'APC_641'),
    ('APC_3340', 'APC_3335'),
    ('APC_4099', 'APC_4103'),
    ('APC_4348', 'APC_4343'),
    ('TP53_637', 'TP53_632'),
    ('TP53_844', 'TP53_849'),
]

# Same noise cutoff used throughout this project. Edit directly if needed.
MIN_PROBABILITY = 0.10

# A residue counts as a "universal hub" if it appears in every sequence with at
# least this many distinct contact partners (averaged across replicates).
MIN_DEGREE_FOR_HUB = 5

FILENAME_RE = re.compile(r'^(?P<gene>[A-Za-z0-9]+)_(?P<position>\d+)_(?P<replicate>\d+)_contacts\.tsv$')
FOLDER_RE = re.compile(r'^(?P<gene>[A-Za-z0-9]+)_(?P<position>\d+)_(?P<label>hot|non)$')


def folder_hotspot_label(dirname: str):
    m = FOLDER_RE.match(dirname)
    if not m:
        return None
    return 1 if m.group('label') == 'hot' else 0


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {path.name}")
    return m.group('gene'), int(m.group('position')), int(m.group('replicate'))


def load_contacts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    n_raw = len(df)

    bad_mask = df['Probability'].isna()
    for col in ('Source', 'Target'):
        parts = df[col].astype(str).str.split('/', expand=True)
        if parts.shape[1] < 3:
            bad_mask = bad_mask | True
        else:
            resnum_str = parts[1].astype(str).str.strip()
            resname_str = parts[2].astype(str).str.strip()
            bad_mask = (
                bad_mask
                | parts[1].isna() | parts[2].isna()
                | (resnum_str == '') | (resname_str == '')
                | ~resnum_str.str.fullmatch(r'-?\d+')
            )

    n_bad = int(bad_mask.sum())
    if n_bad:
        print(f"  ! {path.name}: dropping {n_bad} of {n_raw} row(s) with missing/malformed data", file=sys.stderr)

    df = df[~bad_mask].copy()
    for col in ('Source', 'Target'):
        parts = df[col].str.split('/', expand=True)
        df[f'{col}_resnum'] = parts[1].astype(int)
        df[f'{col}_resname'] = parts[2]
    return df, n_raw, n_bad


def compute_degree(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct contact partners per residue, across BOTH Source and Target roles."""
    edges = pd.concat([
        df[['Source_resnum', 'Source_resname', 'Target_resnum']]
            .rename(columns={'Source_resnum': 'resnum', 'Source_resname': 'resname', 'Target_resnum': 'partner'}),
        df[['Target_resnum', 'Target_resname', 'Source_resnum']]
            .rename(columns={'Target_resnum': 'resnum', 'Target_resname': 'resname', 'Source_resnum': 'partner'}),
    ])
    return edges.groupby(['resnum', 'resname'])['partner'].nunique().reset_index(name='degree')


def process_file(path: Path):
    gene, position, replicate = parse_filename(path)
    sequence = f'{gene}_{position}'
    df_clean, n_raw, n_bad = load_contacts(path)

    n_below = int((df_clean['Probability'] < MIN_PROBABILITY).sum())
    df = df_clean[df_clean['Probability'] >= MIN_PROBABILITY].copy()

    degree_df = compute_degree(df)
    degree_df.insert(0, 'gene', gene)
    degree_df.insert(1, 'position', position)
    degree_df.insert(2, 'replicate', replicate)
    degree_df.insert(3, 'sequence', sequence)
    degree_df.insert(4, 'hotspot', HOTSPOT_MAP.get(sequence))

    return degree_df, n_raw, n_bad, n_below


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ring_dir', type=Path, nargs='?', default=None,
                     help='Path to the RING folder. If omitted, defaults to a "RING" folder next to this script.')
    ap.add_argument('--out', type=Path, default=None,
                     help='Where to write the cross-sequence summary files (default: the RING folder itself)')
    args = ap.parse_args()

    if args.ring_dir is None:
        args.ring_dir = Path(__file__).resolve().parent.parent / 'RING'
        if not args.ring_dir.is_dir():
            sys.exit(f"No RING folder found at {args.ring_dir}. "
                      f"Pass the path explicitly instead: python3 {Path(__file__).name} /path/to/RING")

    out_dir = args.out if args.out is not None else args.ring_dir

    seq_dirs = sorted(d for d in args.ring_dir.iterdir() if d.is_dir())
    if not seq_dirs:
        sys.exit(f"No subfolders found in {args.ring_dir}")

    all_degree_frames = []
    total_raw = total_bad = total_below = 0

    for seq_dir in seq_dirs:
        files = sorted(seq_dir.glob('*_contacts.tsv'))
        if not files:
            continue

        folder_label = folder_hotspot_label(seq_dir.name)
        seq_frames = []
        for f in files:
            try:
                degree_df, n_raw, n_bad, n_below = process_file(f)
            except Exception as e:
                print(f"  ! Skipping {f.name}: {e}", file=sys.stderr)
                continue
            total_raw += n_raw
            total_bad += n_bad
            total_below += n_below

            hotspot_val = HOTSPOT_MAP.get(f'{degree_df["gene"].iloc[0]}_{degree_df["position"].iloc[0]}')
            if folder_label is not None and hotspot_val is not None and folder_label != hotspot_val:
                print(f"  ! WARNING: {seq_dir.name} folder suggests hotspot={folder_label} "
                      f"but HOTSPOT_MAP disagrees", file=sys.stderr)

            seq_frames.append(degree_df)

        if not seq_frames:
            continue

        seq_all = pd.concat(seq_frames, ignore_index=True)
        seq_name = seq_all['sequence'].iloc[0]

        # per-sequence output: average degree per residue across replicates
        seq_avg = (seq_all.groupby(['resnum', 'resname'])['degree']
                   .agg(mean_degree='mean', n_replicates='count').reset_index()
                   .sort_values('mean_degree', ascending=False))
        seq_avg.to_csv(seq_dir / f'{seq_name}_residue_degree.csv', index=False)

        all_degree_frames.append(seq_all)

    if not all_degree_frames:
        sys.exit("No files processed successfully.")

    long_df = pd.concat(all_degree_frames, ignore_index=True)

    print(f"\nQC summary across {long_df['sequence'].nunique()} sequences, "
          f"{long_df.groupby(['sequence'])['replicate'].nunique().sum()} files, {total_raw} total contact rows:")
    print(f"  {total_bad} row(s) dropped for missing/malformed data")
    print(f"  {total_below} row(s) dropped for Probability < {MIN_PROBABILITY}")

    rep_counts = long_df.groupby('sequence')['replicate'].nunique()
    short = rep_counts[rep_counts < 3]
    if len(short):
        print("NOTE: these sequences have fewer than 3 replicates in this run "
              "(fine if you haven't uploaded them all yet):",
              {k: int(v) for k, v in short.items()}, file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / 'all_residue_degree_long.csv', index=False)

    # per-sequence average degree, wide format: rows = residue, columns = sequence
    seq_avg_all = (long_df.groupby(['resnum', 'resname', 'sequence', 'hotspot'])['degree']
                   .mean().reset_index())

    # a residue is a "universal hub" if EVERY sequence present in this run has it
    # at >= MIN_DEGREE_FOR_HUB
    n_sequences = seq_avg_all['sequence'].nunique()
    per_residue = seq_avg_all.groupby(['resnum', 'resname'])
    qualifies = per_residue.apply(
        lambda g: (g['sequence'].nunique() == n_sequences) and (g['degree'].min() >= MIN_DEGREE_FOR_HUB)
    )
    hub_residues = qualifies[qualifies].index.tolist()

    hub_summary = (seq_avg_all[seq_avg_all.set_index(['resnum', 'resname']).index.isin(hub_residues)]
                   .groupby(['resnum', 'resname'])['degree'].agg(['mean', 'min', 'max']).reset_index()
                   .sort_values('mean', ascending=False))
    hub_summary.to_csv(out_dir / 'universal_hub_residues.csv', index=False)

    print(f"\n{len(hub_residues)} universal hub residue(s) found across {n_sequences} sequence(s) "
          f"(present in all, degree >= {MIN_DEGREE_FOR_HUB}):")
    print(hub_summary.to_string(index=False))

    # paired Wilcoxon test per universal hub residue
    stat_rows = []
    for resnum, resname in hub_residues:
        grp = seq_avg_all[(seq_avg_all['resnum'] == resnum) & (seq_avg_all['resname'] == resname)]
        hs_vals, nh_vals = [], []
        for hs_seq, nh_seq in SEQUENCE_PAIRS:
            hs_row = grp[grp['sequence'] == hs_seq]
            nh_row = grp[grp['sequence'] == nh_seq]
            if hs_row.empty or nh_row.empty:
                continue
            hs_vals.append(hs_row['degree'].values[0])
            nh_vals.append(nh_row['degree'].values[0])
        if len(hs_vals) < 2:
            stat_rows.append({'resnum': resnum, 'resname': resname, 'n_pairs': len(hs_vals),
                               'mean_hotspot': np.nan, 'mean_nonhotspot': np.nan,
                               'p_value': np.nan, 'note': 'not enough pairs yet'})
            continue
        try:
            stat, p = wilcoxon(hs_vals, nh_vals)
        except ValueError:
            stat, p = np.nan, np.nan
        stat_rows.append({'resnum': resnum, 'resname': resname, 'n_pairs': len(hs_vals),
                           'mean_hotspot': np.mean(hs_vals), 'mean_nonhotspot': np.mean(nh_vals),
                           'p_value': p, 'note': ''})
    stats_df = pd.DataFrame(stat_rows).sort_values('p_value')
    stats_df.to_csv(out_dir / 'hub_hotspot_vs_nonhotspot_stats.csv', index=False)

    print(f"\nPaired hotspot vs non-hotspot test per hub residue "
          f"({len(SEQUENCE_PAIRS)} pairs defined, some may not be available yet):")
    print(stats_df.to_string(index=False))
    print(f"\nCross-sequence results written to {out_dir}/")
    print(f"Per-sequence results written into each RING/<sequence>_hot|non/ folder.")


if __name__ == '__main__':
    main()