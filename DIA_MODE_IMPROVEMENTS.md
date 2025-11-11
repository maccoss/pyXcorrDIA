# DIA Mode Improvements - Summary

## Overview
This document describes the major improvements made to the DIA (Data-Independent Acquisition) peptide-centric search mode in pyXcorrDIA.

## Key Improvements

### 1. Unified XCorr Calculation
**New:** Single `calculate_xcorr()` function handles all scoring modes

- **Supports both vector and matrix operations** automatically
- **Spectrum-centric mode:** Uses 0.005 scaling factor
- **Peptide-centric mode:** Uses 0.0001 scaling factor (50x smaller)
- **Matrix multiplication:** Scores N peptides × M spectra in one optimized BLAS operation
- **Code simplification:** Eliminates duplicate XCorr implementations
- **Convenience wrappers:** `calculate_fast_xcorr()` and `calculate_peptide_centric_xcorr()` for backward compatibility

### 2. Comprehensive Scoring
**Previous:** Limited scoring between peptides and spectra
**New:** Score ALL spectra against ALL peptides within each isolation window

- Creates a complete N×M scoring matrix (N peptides × M spectra)
- Uses vectorized matrix multiplication for maximum performance
- More thorough chromatographic profile extraction
- Better peptide identification and quantification

### 3. Savitzky-Golay Smoothing
**Added:** Chromatogram smoothing using scipy.signal.savgol_filter

- Applied to raw XCorr chromatograms for each peptide
- Parameters: window_length=7 (or min(len(chromatogram), 7)), polyorder=2
- Adaptive: Falls back to raw values if fewer than 5 data points
- Helps reduce noise and identify true peak maxima

### 5. Optimized Parquet Storage
**New:** Store both raw AND smoothed XCorr values

- Raw XCorr: Used for statistics (mean, median, std) and raw e-values
- Smoothed XCorr: Used for best peak identification and smoothed e-values
- Provides flexibility for downstream analysis
**Previous:** Redundant storage - peptide metadata repeated for every spectrum
**New:** Separate storage of metadata and chromatogram points

#### Two Parquet Files Per Isolation Window:

1. **Chromatogram Points File:** `window_{lower}_{upper}.parquet`
   - Columns: `peptide_id`, `spectrum_idx`, `scan_id`, `rt`, `xcorr_raw`, `xcorr_smoothed`
   - N rows per peptide (where N = number of spectra)
   - Compact storage of chromatographic data

2. **Peptide Metadata File:** `window_{lower}_{upper}_peptides.parquet`
   - Columns: `peptide_id`, `peptide_sequence`, `protein_id`, `charge`, `is_target`, `peptide_mass`, `isolation_window_lower`, `isolation_window_upper`
   - One row per peptide-charge-target combination
   - Eliminates redundancy

**Benefit:** Reduced file size and improved query performance

### 6. Incremental Writing
**Feature:** Batch writing every 100 peptides

- Prevents memory overflow on large datasets
- Enables progress monitoring during long searches
- Data persisted regularly (not just at end)

### 7. Enhanced E-value Calculation
**Method:** Peptide-centric e-values

- E-value = count of XCorr scores >= best XCorr for that peptide
- Calculated separately for both raw and smoothed XCorr
- More appropriate for DIA data than spectrum-centric e-values

### 8. Improved Results Output
**New TSV Format:** Enhanced DIA results with dual metrics

#### Columns:
- `Peptide`: Peptide sequence
- `Charge`: Charge state
- `ProteinID`: Protein identifier
- `Mass`: Peptide mass
- `IsTarget`: Target or Decoy
- `IsolationWindow`: m/z range [lower-upper]
- `BestXCorrRaw`: Best raw XCorr score
- `BestXCorrSmoothed`: Best smoothed XCorr score
- `BestRT`: Retention time at best smoothed peak
- `BestScan`: Scan ID at best smoothed peak
- `EValueRaw`: E-value from raw XCorr distribution
- `EValueSmoothed`: E-value from smoothed XCorr distribution
- `NumSpectraScored`: Total number of spectra scored
- `MeanXCorrRaw`: Mean of all raw XCorr values
- `MedianXCorrRaw`: Median of all raw XCorr values
- `StdXCorrRaw`: Standard deviation of raw XCorr values

## File Structure

### Output Files
For input file `data.mzML`, DIA mode creates:

```
data.dia.tsv                           # Summary results (TSV format)
data_dia_chromatograms/                # Directory for chromatogram data
  ├── window_400.0_410.0.parquet       # Chromatogram points for window 1
  ├── window_400.0_410.0_peptides.parquet  # Peptide metadata for window 1
  ├── window_410.0_420.0.parquet       # Chromatogram points for window 2
  ├── window_410.0_420.0_peptides.parquet  # Peptide metadata for window 2
  └── ...
```

## Usage Example

```bash
# Run DIA mode search
python3 pyXcorrDIA.py \
    uniprot_human.fasta \
    dia_data.mzML \
    --dia_mode \
    --dia_output results.dia.tsv \
    --dia_rt_window 5
```

## Technical Details

### Savitzky-Golay Filter Parameters
- **Window Length:** Dynamically adjusted based on chromatogram length
  - Maximum: 7 data points
  - Minimum: 5 data points (otherwise no smoothing)
  - Must be odd number
- **Polynomial Order:** 2 (quadratic fit)

### Parquet Schema

#### Chromatogram Points:
```
peptide_id: string (unique ID: "PEPTIDEK_2_T")
spectrum_idx: int64 (0-based spectrum index)
scan_id: string (scan identifier)
rt: float64 (retention time)
xcorr_raw: float64 (raw cross-correlation score)
xcorr_smoothed: float64 (Savitzky-Golay smoothed score)
```

#### Peptide Metadata:
```
peptide_id: string (unique ID: "PEPTIDEK_2_T")
peptide_sequence: string (amino acid sequence)
protein_id: string (protein identifier)
charge: int64 (charge state)
is_target: bool (true=target, false=decoy)
peptide_mass: float64 (neutral mass)
isolation_window_lower: float64 (m/z lower bound)
isolation_window_upper: float64 (m/z upper bound)
```

## Performance Benefits

1. **Storage Efficiency:** 
   - Previous: ~11 fields × N spectra per peptide
   - New: 8 fields (metadata) + 6 fields × N spectra
   - Savings: ~30-40% reduction in file size

2. **Query Performance:**
   - Peptide metadata can be loaded once
   - Chromatogram points filtered by peptide_id
   - Parquet's columnar format enables efficient queries

3. **Memory Management:**
   - Incremental batch writing (100 peptides)
   - Prevents memory exhaustion on large datasets
   - Enables processing of arbitrarily large searches

## Statistics Interpretation

- **Mean/Median/Std XCorr:** Calculated from **raw** XCorr values across all spectra
  - Represents the full distribution of scores for that peptide
  - Useful for understanding score variability

- **Best XCorr:** Reported for both **raw** and **smoothed**
  - Smoothed typically more reliable for peak identification
  - Raw preserved for comparison and validation

- **E-values:** Both raw and smoothed
  - Lower e-value = more significant identification
  - E-value = 1 means the best score is unique
  - Higher e-values indicate multiple similar scores

## Future Enhancements

Potential areas for future improvement:
1. Parallel processing of isolation windows
2. Advanced peak detection algorithms
3. Integration with FDR calculation tools
4. Peak area calculation for quantification
5. Alternative smoothing methods (e.g., Gaussian, moving average)

## Dependencies

- `scipy>=1.0.0` (for savgol_filter)
- `pyarrow>=10.0.0` (for Parquet I/O)
- `pandas>=1.3.0` (for DataFrame operations)

## Testing

**Test Suite:** 104 tests total (all passing)
- 17 tests specifically for unified XCorr function (`test_unified_xcorr.py`)
- 16 tests for peptide-centric scoring (`test_peptide_centric.py`, `test_peptide_centric_real_data.py`)
- Tests validate:
  - Unified function works for both single and matrix operations
  - Matrix scoring matches single scoring (consistency)
  - Correct scaling factors (0.005 vs 0.0001)
  - Convenience wrappers work correctly
  - Edge cases (empty arrays, mismatched lengths, 1×1 matrices)
  - Real data preprocessing and scoring

No breaking changes to existing functionality.

---

**Last Updated:** November 2025
**Author:** pyXcorrDIA Development Team
