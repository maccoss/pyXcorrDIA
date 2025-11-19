#!/usr/bin/env python
"""
Create a subset of DIA test data for pytest:
1. Randomly select 1000 precursors from report-lib.parquet
2. Extract proteins needed for those precursors from FASTA
3. Save to tests/data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Set random seed for reproducibility
np.random.seed(42)

# Paths
LIBRARY_PATH = Path("report-lib.parquet")
FASTA_PATH = Path("uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta")
MZML_PATH = Path("Ast-Neo-15min-2mz-4ms-200agc-10.mzML")  # Full 15-min gradient
OUTPUT_DIR = Path("tests/data")

# Target sizes
N_PRECURSORS = 1000

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and subset library
    print(f"Loading library from {LIBRARY_PATH}...")
    df = pd.read_parquet(LIBRARY_PATH)
    print(f"  Original library: {len(df)} rows")
    
    # Get unique precursors (by Precursor.Id)
    precursor_ids = df['Precursor.Id'].unique()
    print(f"  Unique precursor IDs: {len(precursor_ids)}")
    
    # Randomly select N precursors
    selected_precursors = np.random.choice(precursor_ids, size=min(N_PRECURSORS, len(precursor_ids)), replace=False)
    print(f"  Selected {len(selected_precursors)} precursors")
    
    # Filter library to selected precursors
    df_subset = df[df['Precursor.Id'].isin(selected_precursors)].copy()
    print(f"  Subset library: {len(df_subset)} rows (includes fragments for each precursor)")
    
    # Save subset library
    output_lib = OUTPUT_DIR / "test_library_1000.parquet"
    df_subset.to_parquet(output_lib, index=False)
    print(f"  ✓ Saved subset library: {output_lib}")
    
    # 2. Extract proteins from FASTA
    print(f"\nExtracting proteins from {FASTA_PATH}...")
    
    # Get unique protein names from library
    protein_names = set()
    if 'Protein.Names' in df_subset.columns:
        for names in df_subset['Protein.Names'].dropna():
            # Handle multiple proteins (semicolon-separated)
            protein_names.update(names.split(';'))
    elif 'Protein.Ids' in df_subset.columns:
        for ids in df_subset['Protein.Ids'].dropna():
            protein_names.update(ids.split(';'))
    
    print(f"  Found {len(protein_names)} unique proteins")
    
    # Read FASTA and extract matching proteins
    output_fasta = OUTPUT_DIR / "test_proteins_1000.fasta"
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
                write_current = any(prot in line for prot in protein_names)
            else:
                if write_current:
                    current_seq.append(line)
        
        # Write last protein
        if write_current and current_protein:
            fout.write(current_protein)
            fout.write(''.join(current_seq))
            proteins_written += 1
    
    print(f"  ✓ Saved {proteins_written} proteins to: {output_fasta}")
    
    # 3. Copy mzML file
    print("\nCopying DIA mzML file...")
    if MZML_PATH.exists():
        output_mzml = OUTPUT_DIR / "test_dia_full.mzML"
        shutil.copy2(MZML_PATH, output_mzml)
        print(f"  ✓ Copied {MZML_PATH} to: {output_mzml}")
        print(f"     Size: {output_mzml.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"     Note: This is the full 15-minute gradient DIA file")
        print(f"           Provides complete RT coverage for all 1000 precursors")
    else:
        print(f"  ✗ Warning: {MZML_PATH} not found")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test data created successfully!")
    print(f"{'='*60}")
    print(f"Library:  {output_lib}")
    print(f"  - {len(df_subset)} rows ({len(selected_precursors)} precursors)")
    print(f"FASTA:    {output_fasta}")
    print(f"  - {proteins_written} proteins")
    if MZML_PATH.exists():
        print(f"mzML:     {output_mzml}")
    print(f"\nUse these files for DIA integration tests in pytest.")

if __name__ == '__main__':
    main()
