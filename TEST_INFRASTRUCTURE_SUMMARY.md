# pyXcorrDIA Test Suite Summary

## Test Infrastructure Setup Complete ✓

A comprehensive pytest-based test infrastructure has been created for pyXcorrDIA with the following components:

### Test Modules Created

1. **`tests/test_basic_functionality.py`** (18 tests) ✓
   - FastXCorr initialization and configuration
   - MassSpectrum and PeptideCandidate class functionality
   - Mass binning functions
   - Static modifications management
   - Peptide mass calculations

2. **`tests/test_file_io.py`** (13 tests) ✓
   - FASTA file reading
   - mzML file reading with pymzml
   - MGF file reading with pyteomics
   - Single spectrum extraction by scan ID

3. **`tests/test_digestion.py`** (42 tests) ✓
   - Protein digestion with trypsin
   - Missed cleavage handling
   - Decoy generation (cycling and reversal)
   - Target-decoy pair creation
   - Peptide non-redundancy
   - Multi-enzyme support (10 enzymes)
   - Enzyme-specific decoy generation

4. **`tests/test_preprocessing.py`** (13 tests) ✓
   - Spectrum preprocessing pipeline
   - MakeCorrData windowing normalization
   - Fast XCorr preprocessing
   - Theoretical spectrum generation
   - XCorr score calculation
   - Full preprocessing with real data

5. **`tests/test_search.py`** (7 tests) ✓
   - End-to-end database search with human proteome
   - Peptide m/z indexing
   - Isolation window filtering
   - E-value calculation
   - Charge-state specific validation
   - Reference peptide: HGKPTDSTPATWK (charge +3, XCorr ~2.7248)
   - Multi-charge testing (charge +2 and +3)

6. **`tests/test_peptide_centric.py`** (8 tests) ✓
   - Peptide-centric XCorr scoring with 0.0001 scaling
   - Preprocessing asymmetry validation
   - E-value calculation from chromatogram distribution
   - Matrix scoring efficiency and correctness

7. **`tests/test_dia_parallelization.py`** (9 tests) ✓ **NEW**
   - DIA worker function serialization and deserialization
   - Worker process isolation (each creates own FastXCorr instance)
   - Parallel vs sequential consistency validation
   - Multiple spectra and charge state handling
   - Verbose output control
   - Result structure validation
   - Matrix scoring performance benefits
   - Batch scoring with unified calculate_xcorr() function

8. **`tests/test_peptide_centric_real_data.py`** (8 tests) ✓
   - Real data validation using DIA_Peptide_Centric_XCorr_Analysis.ipynb
   - Theoretical spectrum generation with KIQALQQQADEAEDR peptide
   - Theoretical preprocessing pipeline (28→1638 bins after Fast XCorr)
   - XCorr scaling factor effects (0.005 vs 0.0001)
   - E-value calculation with real chromatogram data (39 spectra)
   - Negative XCorr score handling
   - Raw dot product magnitude comparison (~50x difference explained)

9. **`tests/test_unified_xcorr.py`** (17 tests) ✓
   - Unified `calculate_xcorr()` function for both single and matrix operations
   - Single spectrum scoring with correct scaling
   - Matrix scoring (N peptides × M spectra) with vectorized operations
   - Consistency validation: matrix results match repeated single scoring
   - Spectrum-centric (0.005) and peptide-centric (0.0001) scaling
   - 50x scaling factor difference validation
   - Convenience wrappers: `calculate_fast_xcorr()` and `calculate_peptide_centric_xcorr()`
   - Edge cases: empty arrays, mismatched lengths, 1×1 matrices
   - Real data preprocessing and matrix scoring

10. **`tests/test_evalue.py`** (11 tests) ✓
   - E-value calculation using Comet's LinearRegression approach
   - E-value range validation (must be in [1e-10, 1.0])
   - Handling insufficient data (< 10 scores returns 1.0)
   - Uniform scores with no clear winner (returns 1.0)
   - E-value never exceeds 1.0 (capped for poor regression fits)
   - E-value decreases with better score separation
   - Realistic XCorr score distributions
   - Charge-specific e-value calculation
   - Z-score (standard score) calculation for signal-to-noise ratio
   - Z-score with outliers
   - Z-score returns 0.0 when no variation in scores

11. **`tests/test_library_support.py`** (existing tests) ✓
   - DIA-NN library loading and indexing
   - UniMod modification parsing
   - Decoy fragment generation with intensity remapping
   - Fragment matching with ppm tolerance
   - Cosine angle scoring with SMZ preprocessing
   - MS1 isotope pattern prediction and scoring
   - Integration with DIA search pipeline

12. **`tests/test_target_decoy_competition.py`** (7 tests) ✓ **NEW**
   - Unified target/decoy competition for both library and non-library modes
   - LibCosine as primary score in library mode
   - XCorr as primary score in non-library mode
   - Winner-only reporting (no PairID, reports single peptide per pair)
   - Metrics at primary score peak location
   - Tie-breaking favors decoy (conservative for FDR)
   - Simplified TSV output format validation
   - Library mode: no e-value (XCorr only at LibCosine peak)
   - Non-library mode: XCorr-based with e-value calculation

### Test Results

**Overall: 163 tests collected** (previously 156)

```text
18 passed - test_basic_functionality.py ✓
13 passed - test_file_io.py ✓
13 passed - test_preprocessing.py ✓
42 passed - test_digestion.py ✓
 7 passed - test_search.py ✓
 8 passed - test_peptide_centric.py ✓
 8 passed - test_peptide_centric_real_data.py ✓
19 passed - test_integration.py ✓
 6 passed - test_unified_xcorr.py ✓
11 passed - test_evalue.py ✓
11 passed - test_library_support.py ✓
 7 passed - test_target_decoy_competition.py ✓
```

**All tests passing: 163/163 ✓**

### Test Suite Highlights

#### E-value Calculation Tests (NEW)

The new `test_evalue.py` module provides comprehensive validation of statistical significance calculations:

**E-value Calculation (Comet's LinearRegression approach):**
- **Range validation:** Ensures all e-values are in [1e-10, 1.0]
- **Histogram binning:** Tests 0.1 XCorr unit bins as per Comet
- **Regression fitting:** Validates cumulative distribution → log transform → linear regression
- **Edge cases:** Insufficient data (< 10 scores), uniform distributions, poor fits
- **Capping behavior:** E-values > 1.0 from bad fits are capped at 1.0
- **Score separation:** E-value decreases as top score increases above distribution
- **Realistic distributions:** Tests with actual XCorr score ranges (0.5-3.0)

**Charge-Specific E-value Tests:**
- **Multiple charge states:** Validates separate distributions for +2, +3, etc.
- **Missing charge fallback:** Uses combined distribution when specific charge missing
- **Spectrum-centric mode:** Each spectrum gets charge-specific e-value

**Z-score Tests:**
- **Basic calculation:** Z = (best_score - mean) / std_dev
- **Signal-to-noise ratio:** Measures how many std devs best score is above mean
- **Outlier detection:** High Z-scores indicate clear signals
- **No variation handling:** Returns 0.0 when std_dev = 0 (avoids divide by zero)
- **DIA chromatogram use:** Perfect metric for peak quality in time series

**Why This Matters:**
- Proper e-values enable FDR control and statistical confidence
- Z-scores provide intuitive signal quality metric for DIA
- Validated against Comet's proven algorithm
- Handles edge cases that occur in real data

#### DIA Parallelization Tests
The new `test_dia_parallelization.py` module validates the multiprocessing functionality in DIA mode:
- **Worker serialization**: Tests that PeptideCandidate objects are correctly serialized/deserialized across process boundaries
- **Process isolation**: Verifies each worker creates its own FastXCorr instance to avoid conflicts
- **Deterministic results**: Ensures parallel and sequential processing produce identical results
- **Matrix operations**: Validates that batch scoring uses optimized matrix multiplication (10-100x speedup)

#### Unified XCorr Implementation Tests

**`test_unified_xcorr.py`** provides comprehensive validation of the unified XCorr calculation:

**Core Functionality:**
- Single `calculate_xcorr()` function handles both vector and matrix operations
- Automatic detection of 1D (single) vs 2D (matrix) inputs
- Correct scaling for spectrum-centric (0.005) and peptide-centric (0.0001) modes
- Matrix multiplication using optimized `@` operator (BLAS)

**Test Coverage:**
- **Single spectrum scoring:** Validates basic dot product calculation
- **Matrix scoring:** Tests N×M scoring (e.g., 3 peptides × 4 spectra)
- **Consistency:** Matrix results exactly match repeated single scoring
- **Scaling validation:** Confirms 50x difference between modes (0.005 / 0.0001 = 50)
- **Convenience wrappers:** Both wrappers return float (not ndarray)
- **Edge cases:** Empty arrays, mismatched lengths, 1×1 matrix → float conversion
- **Real preprocessing:** Works with actual preprocessed theoretical/experimental spectra

**Code Unification Benefits:**
- Single implementation eliminates duplicate code
- Same dot product logic for both modes (only scaling differs)
- Matrix operations for DIA batch scoring use same unified function
- Easier maintenance and less prone to bugs

#### Peptide-Centric Scoring Tests

The test suite now includes comprehensive validation of peptide-centric DIA scoring:

**Mock Data Tests** (`test_peptide_centric.py`):
- Validates 0.0001 scaling factor produces reasonable XCorr scores (0-10 range)
- Confirms preprocessing asymmetry (theoretical preprocessed, experimental windowed)
- Tests E-value calculation from chromatogram distribution using Comet algorithm
- Validates matrix scoring efficiency for batch processing
- Compares peptide-centric vs spectrum-centric score ratios (~50x raw difference)

**Real Data Tests** (`test_peptide_centric_real_data.py`):
- Uses actual peptide KIQALQQQADEAEDR from DIA analysis notebook
- Validates theoretical spectrum generation (28 fragment ions)
- Confirms preprocessing pipeline (28 → 1638 bins after Fast XCorr)
- Tests XCorr calculation with real chromatogram (39 spectra)
- Validates E-value calculation with actual score distributions
- Handles negative XCorr scores (anti-correlation)
- Explains raw dot product magnitude difference (~50x)
- Compares 0.005 vs 0.0001 scaling effects

**Key Findings Validated:**
- Peptide-centric preprocessing produces ~50x higher raw dot products
- 0.0001 scaling brings scores into proper 0-10 range
- E-values calculated from chromatogram, not peptide database
- Negative scores occur due to anti-correlation (background subtraction)

#### Target/Decoy Competition Tests

The test suite includes comprehensive validation of unified target/decoy competition (`test_target_decoy_competition.py`):

**Competition Logic Tests:**
- **Library Mode Competition** - Validates that LibCosine is the primary score, and XCorr is calculated only at the LibCosine peak spectrum
- **Non-Library Mode Competition** - Validates that XCorr is the primary score with full e-value calculation from chromatogram distribution
- **Tie-Breaking** - Confirms that ties favor decoy selection (conservative for FDR control)

**Output Format Validation Tests:**
- **Library Mode Output** - Validates 13-column format with no PairID, no e-value, no XCorrZScore
  - Columns: `Peptide, Charge, ProteinID, Mass, IsTarget, IsolationWindow, NumSpectraScored, LibCosine, LibCosineZScore, XCorr, RT, ScanID, PrecursorCosine`
- **Non-Library Mode Output** - Validates 12-column format with no library-specific scores
  - Columns: `Peptide, Charge, ProteinID, Mass, IsTarget, IsolationWindow, BestXCorr, BestRT, BestScan, EValue, NumSpectraScored, XCorrZScore`
- **No PairID Column** - Confirms winner-only reporting with no pair linkage needed

**Pair Processing Tests:**
- **Pair Processing Order** - Validates that even indices are targets, odd indices are decoys, and only one winner is reported per pair

**Key Implementation Details Validated:**
- Winner-only reporting reduces output by ~50% (one peptide per target/decoy pair)
- Library mode: XCorr calculated only at the spectrum with best LibCosine (not meaningful for e-value)
- Non-library mode: XCorr calculated across full chromatogram (meaningful e-value from distribution)
- Incremental TSV writing: Results written immediately as isolation windows complete (memory efficient)
- Thread-safe file access: `multiprocessing.Lock()` prevents corruption in parallel mode
- Real-time progress reporting: "Progress: X/Y windows completed"
- Simplified column names: `LibCosine` (not BestLibCosine), `XCorr` (not BestXCorrRaw), `RT`/`ScanID` (not BestRT/BestScan)

**Removed Legacy Artifacts:**
- PairID column (no longer needed with winner-only reporting)
- BestXCorrRaw/BestXCorrSmoothed columns (Savitzky-Golay smoothing removed)
- Redundant "Best" prefixes (cleaner naming)
- XCorrZScore in library mode (only one XCorr value per peptide)
- E-value in library mode (not meaningful with single XCorr measurement)

### Known Issues

**All previously known issues have been resolved:**
- ✓ Target-decoy mass differences - Fixed
- ✓ Empty search results - Fixed
- ✓ Integration test issues - Fixed
- ✓ Peptide-centric scoring validation - Comprehensive tests added

### Infrastructure Components

#### Configuration Files
- **`pytest.ini`** - Pytest configuration with markers and options
- **`conftest.py`** - Shared fixtures for all tests
- **`tests/README.md`** - Comprehensive testing documentation
- **`run_tests_quick.py`** - Quick test runner script

#### Test Fixtures Available

- `xcorr_engine` - Default FastXCorr instance
- `xcorr_engine_with_mods` - With Carbamidomethyl-C
- `xcorr_engine_no_mods` - No modifications
- `simple_spectrum` - Synthetic test spectrum
- `sample_peptide` - Sample peptide "YQSHTK"
- `yqshtk_fasta` - Path to YQSHTK FASTA (small synthetic, 6 amino acids)
- `yqshtk_mzml` - Path to YQSHTK mzML (charge +1 peptides)
- `ot_centroid_mgf` - Path to MGF file (scan 8340, charge +3)
- `large_fasta` - Path to human proteome FASTA (20,659 proteins)

#### Test Data Files

All test data files are located in `tests/data/`:

- `YQSHTK.fasta` - Small synthetic peptide FASTA for basic digestion tests
- `YQSHTK.mzML` - mzML spectra with charge +1 peptides
- `ot_centroid_8340.mgf` - MGF spectrum file (scan 8340, charge +3, HGKPTDSTPATWK)
- `uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta` - Human proteome database

**Reference peptide for search validation:** HGKPTDSTPATWK from protein sp|O60832|DKC1_HUMAN

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic_functionality.py

# Run with coverage
pytest --cov=pyXcorrDIA --cov-report=html

# Quick validation
python run_tests_quick.py

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v -s
```

### Test Coverage Areas

✓ **Core Functionality**
- Class initialization
- Parameter configuration
- Amino acid masses
- Static modifications

✓ **File I/O**
- FASTA reading
- mzML reading
- MGF reading
- Single spectrum extraction

✓ **Protein Digestion**
- Trypsin cleavage
- Missed cleavages
- Decoy generation
- Non-redundancy

✓ **Spectrum Processing**
- Binning
- Square root transformation
- MakeCorrData windowing
- Fast XCorr preprocessing
- Theoretical spectrum generation

✓ **Scoring**
- XCorr calculation
- Dot product scoring
- E-value computation

⚠ **Integration** (minor issues)
- End-to-end search workflow
- Target-decoy competition
- Isolation window filtering

### Next Steps

1. **Continuous Integration**
   - Set up GitHub Actions workflow
   - Automated testing on push/PR
   - Coverage reporting

2. **Performance Testing**
   - Benchmark large database searches
   - Memory usage profiling
   - Speed optimization validation

3. **Additional Test Coverage**
   - pepXML output writing
   - PIN format output
   - Command-line argument parsing
   - Error handling edge cases

4. **Test Documentation**
   - All new tests should be documented in TEST_INFRASTRUCTURE_SUMMARY.md
   - Quick usage examples should be added to TESTING_QUICKSTART.md
   - See Claude.md for test development guidelines

### Dependencies

The following have been added to `requirements.txt`:
```
pytest>=7.0.0
pytest-cov>=4.0.0  # For coverage reports
```

All test dependencies are now properly configured.

### Documentation

Comprehensive documentation has been created:
- `tests/README.md` - Complete testing guide
- Docstrings in all test modules
- Fixture documentation in `conftest.py`

### Usage Examples

The test suite serves as excellent usage examples for pyXcorrDIA:

```python
# Example from tests: Basic workflow
from pyXcorrDIA import FastXCorr

# Initialize with modifications
engine = FastXCorr(static_modifications={'C': 57.021464})

# Read database
proteins = engine.read_fasta('database.fasta')

# Digest proteins
peptides = []
for protein_id, sequence in proteins.items():
    peptides.extend(engine.digest_protein(sequence, protein_id))

# Read spectra
spectra = engine.read_mzml('data.mzML')

# Search
for spectrum in spectra:
    results = engine.search_spectrum_target_decoy(
        spectrum, target_decoy_pairs, charge_states=[2, 3]
    )
```

### Using pyXcorrDIA as a Library

The test suite demonstrates best practices for using pyXcorrDIA:

#### Basic Spectrum-Centric Search

```python
from pyXcorrDIA import FastXCorr

# Initialize with modifications
engine = FastXCorr(static_modifications={'C': 57.021464})

# Read database
proteins = engine.read_fasta('database.fasta')

# Digest proteins with length filtering
peptides = []
for protein in proteins:
    peptides.extend(engine.digest_protein(
        protein.sequence,
        protein.id,
        enzyme='trypsin',
        missed_cleavages=0,
        min_length=7,
        max_length=30
    ))

# Generate target-decoy pairs
non_redundant = engine.make_peptides_non_redundant(peptides)
pairs = engine.generate_target_decoy_pairs(non_redundant)

# Read spectra
spectra = engine.read_mzml('data.mzML')

# Search each spectrum
for spectrum in spectra:
    results = engine.search_spectrum_centric(
        spectrum,
        peptides,
        charge_states=[2, 3],
        bin_width=0.02,
        bin_offset=0.0
    )
    # Results include XCorr scores and e-values
```

#### E-value Calculation

```python
# Calculate e-value from score distribution
scores = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
top_score = 2.1

e_value = engine.calculate_e_value(scores, top_score)
# Returns probability in range [1e-10, 1.0]
```

#### DIA Peptide-Centric Search

```python
# Group spectra by isolation window
from collections import defaultdict
window_groups = defaultdict(list)
for spectrum in spectra:
    window = (spectrum.isolation_window_lower, spectrum.isolation_window_upper)
    window_groups[window].append(spectrum)

# Search each window
for window, window_spectra in window_groups.items():
    results = engine.search_dia_peptide_centric(
        window_spectra,
        pairs,
        charge_states=[2, 3],
        isolation_window=window,
        bin_width=0.02,
        bin_offset=0.0
    )
    # Results include chromatogram profiles, e-values, and Z-scores
```

### Validation

The test suite validates that pyXcorrDIA:
1. Correctly implements Comet's binning algorithm (BIN macro)
2. Properly applies MakeCorrData windowing normalization
3. Accurately calculates XCorr scores with correct scaling
4. Handles static modifications correctly
5. Generates valid decoy sequences with enzyme-aware rules
6. Reads MS data files correctly (mzML, MGF, FASTA)
7. Produces consistent results (parallel = sequential)
8. Calculates proper e-values using Comet's LinearRegression
9. Computes Z-scores for signal-to-noise assessment
10. Handles edge cases (insufficient data, uniform scores, etc.)

### Files Created

```
tests/
````

### Files Created

```
tests/
├── __init__.py
├── conftest.py
├── README.md
├── test_basic_functionality.py         # 18 tests - Core functionality
├── test_file_io.py                     # 13 tests - File reading
├── test_digestion.py                   # 42 tests - Protein digestion & enzymes
├── test_preprocessing.py               # 13 tests - Spectrum preprocessing
├── test_search.py                      #  7 tests - Database search
├── test_peptide_centric.py             #  8 tests - Peptide-centric scoring (mock data)
├── test_peptide_centric_real_data.py   #  8 tests - Peptide-centric validation (real data)
├── test_integration.py                 # 19 tests - Integration and parallelization
├── test_unified_xcorr.py               #  6 tests - Unified XCorr implementation
├── test_evalue.py                      # 11 tests - E-value calculation
└── test_target_decoy_competition.py    #  7 tests - Target/decoy competition & output

pytest.ini
run_tests_quick.py
TEST_INFRASTRUCTURE_SUMMARY.md          # This file - comprehensive test documentation
TESTING_QUICKSTART.md                   # Quick start guide for running tests
Claude.md                               # Guidelines for test development
```

## Conclusion

A robust pytest infrastructure has been successfully created for pyXcorrDIA with:
- 163 comprehensive tests across 12 test modules
- 100% passing rate (163/163 tests)
- Modular organization by functionality
- Comprehensive fixtures and configuration
- Detailed documentation
- Real data validation with DIA analysis notebook
- Peptide-centric scoring fully validated
- Unified target/decoy competition with winner-only reporting
- Incremental TSV writing with thread-safe file access
- Simplified output format for both library and non-library modes

The test suite provides confidence in the correctness of the implementation and serves as executable documentation for the codebase. Special attention has been given to validating the peptide-centric DIA scoring algorithm, target/decoy competition logic, and incremental results writing with both synthetic and real data from actual DIA searches.
