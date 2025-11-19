#!/usr/bin/env python
"""
Create a subset of DIA test data for pytest:
1. Load search results and filter to q-value < 0.01
2. Randomly select N precursors from filtered results
3. Subset DIA-NN library to selected precursors
4. Extract proteins needed for those precursors from FASTA
5. Save to tests/data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyteomics import mzml

# Set random seed for reproducibility
np.random.seed(42)

# Paths
SEARCH_RESULTS_PATH = Path("data/Ast-Neo-15min-2mz-4ms-200agc-10.dia.tsv")
LIBRARY_PATH = Path("data/report-lib.parquet")
FASTA_PATH = Path("data/uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta")
TEST_MZML_PATH = Path("tests/data/test_dia_5windows.mzML")
OUTPUT_DIR = Path("tests/data")

# Target sizes
N_PRECURSORS = 1000
Q_VALUE_THRESHOLD = 0.01


def get_isolation_windows_from_mzml(mzml_path):
    """Extract unique isolation windows from mzML file."""
    windows = set()
    
    with mzml.read(str(mzml_path)) as reader:
        for spectrum in reader:
            if spectrum.get('ms level') == 2:
                precursor_list = spectrum.get('precursorList', {})
                if not precursor_list:
                    continue
                
                precursors = precursor_list.get('precursor', [])
                if not precursors:
                    continue
                
                precursor = precursors[0]
                isolation_window = precursor.get('isolationWindow', {})
                
                if not isolation_window:
                    continue
                
                target_mz = isolation_window.get('isolation window target m/z')
                lower_offset = isolation_window.get('isolation window lower offset')
                upper_offset = isolation_window.get('isolation window upper offset')
                
                if target_mz is not None and lower_offset is not None and upper_offset is not None:
                    lower = target_mz - lower_offset
                    upper = target_mz + upper_offset
                    windows.add((round(lower, 4), round(upper, 4)))
    
    return sorted(windows)


def precursor_in_windows(mass, charge, windows):
    """Check if precursor m/z falls within any isolation window."""
    precursor_mz = mass / charge
    for lower, upper in windows:
        if lower <= precursor_mz <= upper:
            return True
    return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. Get isolation windows from test mzML file
    print(f"Reading isolation windows from {TEST_MZML_PATH}...")
    if not TEST_MZML_PATH.exists():
        print(f"  ✗ Error: {TEST_MZML_PATH} not found")
        print(f"  Please run: python scripts/subset_mzml_by_windows.py to create test mzML first")
        return
    
    isolation_windows = get_isolation_windows_from_mzml(TEST_MZML_PATH)
    print(f"  Found {len(isolation_windows)} isolation windows")
    if isolation_windows:
        print(f"  Window range: [{isolation_windows[0][0]:.2f}-{isolation_windows[0][1]:.2f}] to "
              f"[{isolation_windows[-1][0]:.2f}-{isolation_windows[-1][1]:.2f}] m/z")
    
    # 1. Load search results and filter by q-value
    print(f"\nLoading search results from {SEARCH_RESULTS_PATH}...")
    if not SEARCH_RESULTS_PATH.exists():
        print(f"  ✗ Error: {SEARCH_RESULTS_PATH} not found")
        return
    
    results_df = pd.read_csv(SEARCH_RESULTS_PATH, sep='\t', low_memory=False)
    print(f"  Original results: {len(results_df)} PSMs")
    
    # Filter by q-value (use mokapot precursor q-value)
    if 'mokapot_precursor_qvalue' not in results_df.columns:
        print(f"  ✗ Error: mokapot_precursor_qvalue column not found")
        print(f"  Available columns: {results_df.columns.tolist()}")
        return
    
    filtered_df = results_df[results_df['mokapot_precursor_qvalue'] <= Q_VALUE_THRESHOLD].copy()
    print(f"  Filtered (mokapot_precursor_qvalue <= {Q_VALUE_THRESHOLD}): {len(filtered_df)} PSMs")
    
    # Filter to precursors within test mzML isolation windows
    if 'Mass' in filtered_df.columns and 'Charge' in filtered_df.columns:
        filtered_df['in_window'] = filtered_df.apply(
            lambda row: precursor_in_windows(row['Mass'], row['Charge'], isolation_windows),
            axis=1
        )
        filtered_df = filtered_df[filtered_df['in_window']].copy()
        print(f"  Filtered to precursors in test mzML windows: {len(filtered_df)} PSMs")
    else:
        print(f"  ✗ Warning: Could not filter by isolation windows (missing Mass or Charge columns)")
    
    # Get unique precursors (peptide + charge)
    # Create precursor ID from peptide sequence and charge
    if 'Peptide' in filtered_df.columns and 'Charge' in filtered_df.columns:
        filtered_df['precursor_id'] = filtered_df['Peptide'] + '_' + filtered_df['Charge'].astype(str)
    else:
        print("  ✗ Error: Could not find Peptide and Charge columns")
        return
    
    unique_precursors = filtered_df['precursor_id'].unique()
    print(f"  Unique precursors: {len(unique_precursors)}")
    
    # Check if we have enough precursors
    if len(unique_precursors) < N_PRECURSORS:
        print(f"  ✗ Warning: Only found {len(unique_precursors)} precursors, need {N_PRECURSORS}")
        print(f"  Consider using more isolation windows in the test mzML file")
        print(f"  Proceeding with {len(unique_precursors)} precursors...")
        n_select = len(unique_precursors)
        selected_precursors = unique_precursors
    else:
        # Randomly select N precursors
        n_select = N_PRECURSORS
        selected_precursors = np.random.choice(unique_precursors, size=n_select, replace=False)
        print(f"  Selected {len(selected_precursors)} precursors for subset")
    
    # Keep peptide sequences for library matching
    selected_peptides = set()
    for precursor_id in selected_precursors:
        peptide = precursor_id.rsplit('_', 1)[0]  # Remove charge suffix
        selected_peptides.add(peptide)
    
    # 2. Subset DIA-NN library
    print(f"\nLoading library from {LIBRARY_PATH}...")
    if not LIBRARY_PATH.exists():
        print(f"  ✗ Error: {LIBRARY_PATH} not found")
        return
    
    lib_df = pd.read_parquet(LIBRARY_PATH)
    print(f"  Original library: {len(lib_df)} rows")
    
    # Filter library to selected peptides
    # DIA-NN library uses different column names
    if 'Stripped.Sequence' in lib_df.columns:
        peptide_col = 'Stripped.Sequence'
    elif 'StrippedPeptide' in lib_df.columns:
        peptide_col = 'StrippedPeptide'
    elif 'ModifiedPeptide' in lib_df.columns:
        peptide_col = 'ModifiedPeptide'
    elif 'Modified.Sequence' in lib_df.columns:
        peptide_col = 'Modified.Sequence'
    elif 'PeptideSequence' in lib_df.columns:
        peptide_col = 'PeptideSequence'
    else:
        print("  Available columns:", lib_df.columns.tolist())
        print("  ✗ Error: Could not find peptide sequence column in library")
        return
    
    lib_subset = lib_df[lib_df[peptide_col].isin(selected_peptides)].copy()
    print(f"  Subset library: {len(lib_subset)} rows (includes fragments for {len(lib_subset[peptide_col].unique())} peptides)")
    
    # Save subset library
    output_lib = OUTPUT_DIR / f"test_library_{n_select}.parquet"
    lib_subset.to_parquet(output_lib, index=False)
    print(f"  ✓ Saved subset library: {output_lib}")
    
    # 3. Extract proteins from FASTA
    print(f"\nExtracting proteins from {FASTA_PATH}...")
    if not FASTA_PATH.exists():
        print(f"  ✗ Error: {FASTA_PATH} not found")
        return
    
    # Get unique protein names from library
    protein_names = set()
    if 'Protein.Names' in lib_subset.columns:
        for names in lib_subset['Protein.Names'].dropna():
            if isinstance(names, str):
                protein_names.update(names.split(';'))
    elif 'Protein.Ids' in lib_subset.columns:
        for ids in lib_subset['Protein.Ids'].dropna():
            if isinstance(ids, str):
                protein_names.update(ids.split(';'))
    elif 'ProteinName' in lib_subset.columns:
        for names in lib_subset['ProteinName'].dropna():
            if isinstance(names, str):
                protein_names.update(names.split(';'))
    elif 'ProteinId' in lib_subset.columns:
        for ids in lib_subset['ProteinId'].dropna():
            if isinstance(ids, str):
                protein_names.update(ids.split(';'))
    elif 'UniprotId' in lib_subset.columns:
        for ids in lib_subset['UniprotId'].dropna():
            if isinstance(ids, str):
                protein_names.update(ids.split(';'))
    
    if not protein_names:
        print("  ✗ Warning: Could not extract protein names from library")
        print(f"    Available columns: {lib_subset.columns.tolist()}")
    
    print(f"  Found {len(protein_names)} unique protein identifiers")
    
    # Read FASTA and extract matching proteins
    output_fasta = OUTPUT_DIR / f"test_proteins_{n_select}.fasta"
    proteins_written = 0
    current_protein = None
    current_seq = []
    write_current = False
    
    with open(FASTA_PATH, 'r') as fin, open(output_fasta, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                # Write previous protein if needed
                if write_current and current_protein:
                    fout.write(current_protein)
                    fout.write(''.join(current_seq))
                    proteins_written += 1
                
                # Check if this protein is in our subset
                current_protein = line
                current_seq = []
                
                # Check if any protein name matches the header
                write_current = any(prot in line for prot in protein_names) if protein_names else False
            else:
                if write_current:
                    current_seq.append(line)
        
        # Write last protein
        if write_current and current_protein:
            fout.write(current_protein)
            fout.write(''.join(current_seq))
            proteins_written += 1
    
    print(f"  ✓ Saved {proteins_written} proteins to: {output_fasta}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test data created successfully!")
    print(f"{'='*60}")
    print(f"Library:  {output_lib}")
    print(f"  - {len(lib_subset)} rows ({len(lib_subset[peptide_col].unique())} unique peptides)")
    print(f"FASTA:    {output_fasta}")
    print(f"  - {proteins_written} proteins")
    print(f"\nSource data:")
    print(f"  - Search results: {SEARCH_RESULTS_PATH}")
    print(f"  - Filtered to mokapot_precursor_qvalue <= {Q_VALUE_THRESHOLD}")
    print(f"  - Filtered to precursors in {TEST_MZML_PATH} isolation windows")
    print(f"  - Selected {n_select} precursors from {len(unique_precursors)} available")
    print("\nUse these files for DIA integration tests in pytest.")

if __name__ == '__main__':
    main()
