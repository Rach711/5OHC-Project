#!/usr/bin/env python3
"""
Extract and aggregate RING contact-probability data for NTHL1-DNA MD simulations,
focused on the damaged base (DO) and the Lys212/Asp231 catalytic dyad, and compare
hotspot vs non-hotspot sequence contexts.

Input layout (as produced by your RING folder structure):
    RING/
      APC_637_hot/
        APC_637_1_contacts.tsv
        APC_637_2_contacts.tsv
        APC_637_3_contacts.tsv
        (plus .pdb files, ignored)
      APC_641_non/
        APC_641_1_contacts.tsv
        ...
      ...

Each *_contacts.tsv has columns: Source, Interaction, Target, Probability
where Source/Target look like "_/212/LYS" (chain/resnum/resname), and Probability
is the fraction of frames (0-1) in which that contact was present.

Usage:
    python analyse_ring_contacts.py                        (defaults to ../RING next to this script)
    python analyse_ring_contacts.py /path/to/RING           (explicit override, if ever needed)

Outputs:
    Written into EACH sequence subfolder (e.g. RING/APC_637_hot/):
        APC_637_DO_summary.csv        one row per replicate (3 rows) for this sequence
        APC_637_DO_contacts_long.csv  every contact touching the damaged base, this sequence

    Written into the RING folder itself (or --out if given) - cross-sequence results:
        per_replicate_summary.csv       one row per simulation (36 rows), all sequences
        all_DO_contacts_long.csv        every DO contact, all sequences, long format
        per_sequence_averaged.csv       replicate-averaged, one row per sequence (12 rows)
        hotspot_vs_nonhotspot_stats.csv paired Wilcoxon signed-rank test per metric
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Configuration - check these before trusting the output
# ---------------------------------------------------------------------------

# Hotspot (1) / non-hotspot (0) labels, taken directly from RandomForest.py
HOTSPOT_MAP = {
    'APC_637': 1, 'APC_641': 0, 'APC_3335': 0, 'APC_3340': 1,
    'APC_4099': 1, 'APC_4103': 0, 'APC_4343': 0, 'APC_4348': 1,
    'TP53_632': 0, 'TP53_637': 1, 'TP53_844': 1, 'TP53_849': 0,
}

# Matched hotspot/non-hotspot pairs used for the paired Wilcoxon test.
# >>> Inferred from nearby genomic positions in HOTSPOT_MAP - PLEASE CONFIRM <<<
# Each tuple is (hotspot_sequence, non_hotspot_sequence).
SEQUENCE_PAIRS = [
    ('APC_637', 'APC_641'),
    ('APC_3340', 'APC_3335'),
    ('APC_4099', 'APC_4103'),
    ('APC_4348', 'APC_4343'),
    ('TP53_637', 'TP53_632'),
    ('TP53_844', 'TP53_849'),
]

# Specific (non-promiscuous) interaction types to keep; VDW excluded
KEEP_INTERACTIONS = {'HBOND', 'IONIC', 'PICATION', 'PIHBOND', 'PIPISTACK'}

# Catalytic dyad residue numbers, confirmed against this PDB's own numbering
# (Lys212 = nucleophile, Asp231 = general base; see Ikeda et al. 1998 JBC and
# Carroll et al. 2021 NAR - note Carroll's paper uses a +8 offset, Lys220/Asp239)
CAT_LYS = 212
CAT_ASP = 231

# Minimum contact Probability to count at all (contacts below this are treated as
# noise/transient and dropped before any aggregation). Change this value directly if needed.
MIN_PROBABILITY = 0.10

FILENAME_RE = re.compile(r'^(?P<gene>[A-Za-z0-9]+)_(?P<position>\d+)_(?P<replicate>\d+)_contacts\.tsv$')
FOLDER_RE = re.compile(r'^(?P<gene>[A-Za-z0-9]+)_(?P<position>\d+)_(?P<label>hot|non)$')


def folder_hotspot_label(dirname: str):
    """Return 1/0/None from a folder name like 'APC_637_hot' or 'APC_641_non'."""
    m = FOLDER_RE.match(dirname)
    if not m:
        return None
    return 1 if m.group('label') == 'hot' else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {path.name}")
    return m.group('gene'), int(m.group('position')), int(m.group('replicate'))


def load_contacts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    n_raw = len(df)

    # A row is "bad" if Probability is missing, or Source/Target don't split into
    # exactly chain/resnum/resname (e.g. a truncated or malformed line).
    bad_mask = df['Probability'].isna()
    split_parts = {}
    for col in ('Source', 'Target'):
        parts = df[col].astype(str).str.split('/', expand=True)
        split_parts[col] = parts
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
        print(f"  ! {path.name}: dropping {n_bad} of {n_raw} row(s) with missing/malformed "
              f"Source, Target, or Probability", file=sys.stderr)

    df = df[~bad_mask].copy()
    for col in ('Source', 'Target'):
        parts = df[col].str.split('/', expand=True)
        df[f'{col}_resnum'] = parts[1].astype(int)
        df[f'{col}_resname'] = parts[2]
    return df, n_raw, n_bad


def find_damaged_base_resnum(df: pd.DataFrame) -> int:
    do_nums = pd.unique(pd.concat([
        df.loc[df['Source_resname'] == 'DO', 'Source_resnum'],
        df.loc[df['Target_resname'] == 'DO', 'Target_resnum'],
    ]))
    if len(do_nums) == 0:
        raise ValueError("No DO (damaged base) residue found in this file")
    if len(do_nums) > 1:
        raise ValueError(f"Multiple DO residue numbers found: {list(do_nums)} - unexpected")
    return int(do_nums[0])


def _filter_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """mode: 'specific' (HBOND/IONIC/PICATION/PIHBOND/PIPISTACK), 'vdw', or 'all'."""
    if mode == 'specific':
        return df[df['Interaction'].isin(KEEP_INTERACTIONS)]
    elif mode == 'vdw':
        return df[df['Interaction'] == 'VDW']
    elif mode == 'all':
        return df
    raise ValueError(f"Unknown mode: {mode}")


def contacts_touching_residue(df: pd.DataFrame, resnum: int, mode='specific') -> pd.DataFrame:
    mask = (df['Source_resnum'] == resnum) | (df['Target_resnum'] == resnum)
    out = df[mask].copy()
    out = _filter_mode(out, mode)
    out['partner_resnum'] = np.where(out['Source_resnum'] == resnum, out['Target_resnum'], out['Source_resnum'])
    out['partner_resname'] = np.where(out['Source_resnum'] == resnum, out['Target_resname'], out['Source_resname'])
    return out


def contact_between(df: pd.DataFrame, resnum_a: int, resnum_b: int, mode='specific') -> pd.DataFrame:
    mask = (
        ((df['Source_resnum'] == resnum_a) & (df['Target_resnum'] == resnum_b)) |
        ((df['Source_resnum'] == resnum_b) & (df['Target_resnum'] == resnum_a))
    )
    out = df[mask].copy()
    out = _filter_mode(out, mode)
    return out


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: Path, min_probability: float = MIN_PROBABILITY):
    gene, position, replicate = parse_filename(path)
    sequence = f'{gene}_{position}'
    df_clean, n_raw, n_bad_rows = load_contacts(path)

    # Find the damaged base from the full cleaned data BEFORE thresholding, so its
    # identity doesn't depend on whether its contacts happen to be persistent.
    do_resnum = find_damaged_base_resnum(df_clean)

    # Now drop low-persistence contacts (below min_probability) for everything else.
    n_below_threshold = int((df_clean['Probability'] < min_probability).sum())
    df = df_clean[df_clean['Probability'] >= min_probability].copy()

    # Damaged-base contacts: keep ALL interaction types here (specific + VDW),
    # tagged with a category column, so nothing is silently discarded.
    do_contacts_all = contacts_touching_residue(df, do_resnum, mode='all')
    do_contacts_all['category'] = np.where(do_contacts_all['Interaction'] == 'VDW', 'vdw', 'specific')
    do_contacts_specific = do_contacts_all[do_contacts_all['category'] == 'specific']
    do_contacts_vdw = do_contacts_all[do_contacts_all['category'] == 'vdw']

    long_df = do_contacts_all[['Interaction', 'category', 'Probability', 'partner_resnum', 'partner_resname']].copy()
    long_df.insert(0, 'gene', gene)
    long_df.insert(1, 'position', position)
    long_df.insert(2, 'replicate', replicate)
    long_df.insert(3, 'sequence', sequence)
    long_df.insert(4, 'hotspot', HOTSPOT_MAP.get(sequence))
    long_df.insert(5, 'do_resnum', do_resnum)

    # Catalytic dyad itself stays specific-interactions-only (this is a genuine
    # H-bond/ionic salt bridge, not a lesion-pocket-insertion signal)
    dyad = contact_between(df, CAT_LYS, CAT_ASP, mode='specific')

    lys_do_specific = contact_between(df, CAT_LYS, do_resnum, mode='specific')
    lys_do_vdw = contact_between(df, CAT_LYS, do_resnum, mode='vdw')
    asp_do_specific = contact_between(df, CAT_ASP, do_resnum, mode='specific')
    asp_do_vdw = contact_between(df, CAT_ASP, do_resnum, mode='vdw')

    summary = {
        'gene': gene, 'position': position, 'replicate': replicate,
        'sequence': sequence, 'hotspot': HOTSPOT_MAP.get(sequence),
        'do_resnum': do_resnum,
        'n_rows_raw': n_raw,
        'n_rows_dropped_missing': n_bad_rows,
        'n_rows_dropped_below_threshold': n_below_threshold,
        'n_residues_contacting_DO_specific': int(do_contacts_specific['partner_resnum'].nunique()),
        'n_residues_contacting_DO_vdw': int(do_contacts_vdw['partner_resnum'].nunique()),
        'total_DO_contact_probability_specific': float(do_contacts_specific['Probability'].sum()),
        'total_DO_contact_probability_vdw': float(do_contacts_vdw['Probability'].sum()),
        'dyad_hbond_prob': float(dyad.loc[dyad['Interaction'] == 'HBOND', 'Probability'].sum()),
        'dyad_ionic_prob': float(dyad.loc[dyad['Interaction'] == 'IONIC', 'Probability'].sum()),
        f'Lys{CAT_LYS}_DO_prob_specific': float(lys_do_specific['Probability'].sum()),
        f'Lys{CAT_LYS}_DO_prob_vdw': float(lys_do_vdw['Probability'].sum()),
        f'Asp{CAT_ASP}_DO_prob_specific': float(asp_do_specific['Probability'].sum()),
        f'Asp{CAT_ASP}_DO_prob_vdw': float(asp_do_vdw['Probability'].sum()),
    }
    return summary, long_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ring_dir', type=Path, nargs='?', default=None,
                     help='Path to the RING folder (contains one subfolder per sequence, e.g. APC_637_hot). '
                          'If omitted, defaults to a "RING" folder next to this script (i.e. Scripts/../RING).')
    ap.add_argument('--out', type=Path, default=None,
                     help='Where to write the cross-sequence summary files (default: the RING folder itself)')
    args = ap.parse_args()

    if args.ring_dir is None:
        args.ring_dir = Path(__file__).resolve().parent.parent / 'RING'
        if not args.ring_dir.is_dir():
            sys.exit(f"No RING folder found at {args.ring_dir} (expected next to this script's parent folder). "
                      f"Pass the path explicitly instead: python3 {Path(__file__).name} /path/to/RING")

    out_dir = args.out if args.out is not None else args.ring_dir

    seq_dirs = sorted(d for d in args.ring_dir.iterdir() if d.is_dir())
    if not seq_dirs:
        sys.exit(f"No subfolders found in {args.ring_dir}")

    summaries, long_frames = [], []

    for seq_dir in seq_dirs:
        files = sorted(seq_dir.glob('*_contacts.tsv'))
        if not files:
            continue  # not a sequence folder (e.g. unrelated folder alongside RING/)

        folder_label = folder_hotspot_label(seq_dir.name)

        seq_summaries, seq_long_frames = [], []
        for f in files:
            try:
                summary, long_df = process_file(f)
            except Exception as e:
                print(f"  ! Skipping {f.name}: {e}", file=sys.stderr)
                continue

            # cross-check the folder's _hot/_non suffix against HOTSPOT_MAP
            if folder_label is not None and summary['hotspot'] is not None and folder_label != summary['hotspot']:
                print(f"  ! WARNING: {seq_dir.name} folder suggests hotspot={folder_label} "
                      f"but HOTSPOT_MAP says {summary['hotspot']} for {summary['sequence']}", file=sys.stderr)

            seq_summaries.append(summary)
            seq_long_frames.append(long_df)

        if not seq_summaries:
            continue

        # --- write this sequence's own results back into its own subfolder ---
        seq_summary_df = pd.DataFrame(seq_summaries).sort_values('replicate')
        seq_long_df = pd.concat(seq_long_frames, ignore_index=True)
        seq_name = seq_summary_df['sequence'].iloc[0]
        seq_summary_df.to_csv(seq_dir / f'{seq_name}_DO_summary.csv', index=False)
        seq_long_df.to_csv(seq_dir / f'{seq_name}_DO_contacts_long.csv', index=False)

        summaries.extend(seq_summaries)
        long_frames.extend(seq_long_frames)

    if not summaries:
        sys.exit("No files processed successfully.")

    summary_df = pd.DataFrame(summaries).sort_values(['gene', 'position', 'replicate'])
    long_df_all = pd.concat(long_frames, ignore_index=True)

    missing = summary_df[summary_df['hotspot'].isnull()]
    if len(missing):
        print("WARNING: these sequences aren't in HOTSPOT_MAP:",
              sorted(missing['sequence'].unique()), file=sys.stderr)

    do_counts = summary_df.groupby('gene')['do_resnum'].nunique()
    inconsistent = do_counts[do_counts > 1]
    if len(inconsistent):
        print("WARNING: damaged-base residue number is NOT consistent within these genes:",
              list(inconsistent.index), file=sys.stderr)

    rep_counts = summary_df.groupby('sequence')['replicate'].nunique()
    short = rep_counts[rep_counts < 3]
    if len(short):
        print("WARNING: these sequences have fewer than 3 replicates in the output "
              "(a file may have failed to parse - see warnings above):",
              {k: int(v) for k, v in short.items()}, file=sys.stderr)

    total_dropped_missing = int(summary_df['n_rows_dropped_missing'].sum())
    total_dropped_threshold = int(summary_df['n_rows_dropped_below_threshold'].sum())
    total_raw = int(summary_df['n_rows_raw'].sum())
    print(f"\nQC summary across {len(summary_df)} files, {total_raw} total contact rows:")
    print(f"  {total_dropped_missing} row(s) dropped for missing/malformed data")
    print(f"  {total_dropped_threshold} row(s) dropped for Probability < {MIN_PROBABILITY}")

    # --- write the cross-sequence ("non specific") results into the RING folder ---
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / 'per_replicate_summary.csv', index=False)
    long_df_all.to_csv(out_dir / 'all_DO_contacts_long.csv', index=False)

    numeric_cols = ['n_residues_contacting_DO_specific', 'n_residues_contacting_DO_vdw',
                     'total_DO_contact_probability_specific', 'total_DO_contact_probability_vdw',
                     'dyad_hbond_prob', 'dyad_ionic_prob',
                     f'Lys{CAT_LYS}_DO_prob_specific', f'Lys{CAT_LYS}_DO_prob_vdw',
                     f'Asp{CAT_ASP}_DO_prob_specific', f'Asp{CAT_ASP}_DO_prob_vdw']
    per_seq = (summary_df.groupby(['gene', 'position', 'sequence', 'hotspot'])[numeric_cols]
               .mean().reset_index())
    per_seq.to_csv(out_dir / 'per_sequence_averaged.csv', index=False)

    stat_rows = []
    for metric in numeric_cols:
        hs_vals, nh_vals = [], []
        for hotspot_seq, nonhotspot_seq in SEQUENCE_PAIRS:
            hs_row = per_seq[per_seq['sequence'] == hotspot_seq]
            nh_row = per_seq[per_seq['sequence'] == nonhotspot_seq]
            if hs_row.empty or nh_row.empty:
                continue
            hs_vals.append(hs_row[metric].values[0])
            nh_vals.append(nh_row[metric].values[0])
        if len(hs_vals) < 2:
            stat_rows.append({'metric': metric, 'n_pairs': len(hs_vals),
                               'mean_hotspot': np.nan, 'mean_nonhotspot': np.nan,
                               'wilcoxon_stat': np.nan, 'p_value': np.nan,
                               'note': 'not enough pairs'})
            continue
        try:
            stat, p = wilcoxon(hs_vals, nh_vals)
        except ValueError:
            stat, p = np.nan, np.nan
        stat_rows.append({
            'metric': metric, 'n_pairs': len(hs_vals),
            'mean_hotspot': np.mean(hs_vals), 'mean_nonhotspot': np.mean(nh_vals),
            'wilcoxon_stat': stat, 'p_value': p, 'note': ''
        })
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(out_dir / 'hotspot_vs_nonhotspot_stats.csv', index=False)

    print(f"Processed {len(summaries)} files across {len(seq_dirs)} sequence folders.")
    print(f"Per-sequence results written into each RING/<sequence>_hot|non/ folder.")
    print(f"Cross-sequence results written to {out_dir}/\n")
    print(stats_df.to_string(index=False))


if __name__ == '__main__':
    main()