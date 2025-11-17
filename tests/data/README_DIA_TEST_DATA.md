# DIA Test Data

This directory contains test datasets for DIA (Data-Independent Acquisition) peptide-centric search validation.

## Files

### `test_library_1000.parquet`
- **Source**: Randomly selected from `report-lib.parquet` (DIA-NN library format)
- **Precursors**: 1,000 randomly selected precursors (seed=42 for reproducibility)
- **Rows**: ~11,851 rows (includes all fragment ions for each precursor)
- **Purpose**: Realistic library for testing with manageable size

### `test_proteins_1000.fasta`
- **Source**: Subset of `uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta`
- **Proteins**: 943 proteins that contain the 1,000 library precursors
- **Purpose**: Matching protein database for the test library

### `test_dia_60000-70000.mzML`
- **Source**: Subset of `Ast-Neo-15min-2mz-4ms-200agc-10.mzML`
- **Scan range**: Scan IDs 60000-70000 (~1 minute RT window)
- **Size**: ~248 MB
- **Purpose**: Small DIA file for fast testing
- **Limitation**: Narrow RT range (~1 min) means RT correlation plots will show limited spread
  - For full RT correlation validation, use the complete 15-minute gradient file

## Regenerating Test Data

To regenerate these test files:

```bash
python create_test_data_subset.py
```

Requirements:
- `report-lib.parquet` (DIA-NN library)
- `uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta`
- `Ast-Neo-15min-2mz-4ms-200agc-60000-70000.mzML`

## Usage in Tests

Example pytest usage:

```python
def test_dia_search(dia_library_1000, dia_fasta_1000, dia_mzml_small):
    # Fixtures automatically provide paths to these files
    library = SpectrumLibrary(dia_library_1000)
    # ...
```

## QC Plot Expectations

When running DIA search with these test files:

### MS1 Mass Accuracy
- **Expected**: ~1,000 measurements (one per precursor)
- **Quality**: Representative of full dataset
- **Units**: PPM or m/z (depending on search parameters)

### MS2 Mass Accuracy
- **Expected**: ~12-15 measurements per precursor (~12,000 total)
- **Quality**: Representative of full dataset
- **Collection**: Only at spectrum with best LibCosine score (not all spectra)

### RT Correlation
- **Expected**: ~1,000 RT pairs (library RT vs measured RT)
- **Quality**: LIMITED due to narrow RT window (~1 min)
- **Limitation**: Most precursors will elute in similar RT range
- **For full validation**: Use `Ast-Neo-15min-2mz-4ms-200agc-10.mzML` (15 min gradient)

## Running DIA Search with Test Data

Quick test (100 precursors):
```bash
python pyXcorrDIA.py tests/data/test_proteins_1000.fasta \
                     tests/data/test_dia_60000-70000.mzML \
                     --dia_mode \
                     --speclib tests/data/test_library_1000.parquet \
                     --test_library_peptides 100 \
                     --verbose
```

Full test (1000 precursors):
```bash
python pyXcorrDIA.py tests/data/test_proteins_1000.fasta \
                     tests/data/test_dia_60000-70000.mzML \
                     --dia_mode \
                     --speclib tests/data/test_library_1000.parquet \
                     --verbose
```

## Validation Checklist

After running DIA search, verify:

- ✓ MS1 errors: ~1 per precursor (not 100x overcounting)
- ✓ MS2 errors: ~12-15 per precursor (not 1000x overcounting from all spectra)
- ✓ RT pairs: ~1 per precursor
- ✓ QC data includes peptide/charge/is_target metadata
- ✓ Mass errors in correct units (PPM or m/z)
- ⚠️ RT correlation: Will show limited spread due to narrow RT window
