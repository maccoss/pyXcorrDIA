# pyXcorrDIA Testing Quick Start Guide

## ✓ Test Infrastructure is Ready!

A complete pytest test suite with **124 tests** has been created for pyXcorrDIA, including comprehensive validation of the unified XCorr implementation, peptide-centric DIA scoring with real data, DIA parallelization, and E-value calculation.

## Quick Commands

### 1. Install Test Dependencies (if not already done)

```bash
pip install pytest pytest-cov
```

### 2. Run Quick Validation

```bash
# Quick check that everything is working (runs 18 basic tests)
python run_tests_quick.py
```

### 3. Run All Tests

```bash
# Run all 124 tests
pytest

# With verbose output
pytest -v

# Show print statements (useful for real data tests)
pytest -v -s
```

### 4. Run Specific Test Modules

```bash
# Test basic functionality (initialization, modifications, mass calculations)
pytest tests/test_basic_functionality.py

# Test file I/O (FASTA, mzML, MGF reading)
pytest tests/test_file_io.py

# Test protein digestion and decoys (includes 10 enzymes)
pytest tests/test_digestion.py

# Test spectrum preprocessing and XCorr
pytest tests/test_preprocessing.py

# Test database search workflow
pytest tests/test_search.py

# Test peptide-centric scoring (mock data)
pytest tests/test_peptide_centric.py

# Test peptide-centric scoring (real data validation)
pytest tests/test_peptide_centric_real_data.py -v -s

# Test unified XCorr function (single and matrix operations)
pytest tests/test_unified_xcorr.py -v

# Test E-value calculation and Z-scores
pytest tests/test_evalue.py -v

# Test DIA parallelization
pytest tests/test_dia_parallelization.py -v
```

### 5. Run with Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=pyXcorrDIA --cov-report=html

# View the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Organization

```text
tests/
├── __init__.py                      # Package initialization
├── conftest.py                      # Shared fixtures and configuration
├── README.md                        # Detailed testing documentation
├── data/                            # Test data files
│   ├── YQSHTK.fasta                # Small synthetic peptide FASTA
│   ├── YQSHTK.mzML                 # mzML spectra (charge +1)
│   ├── ot_centroid_8340.mgf        # MGF spectrum (charge +3)
│   └── uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta  # Human proteome
├── test_basic_functionality.py      # 18 tests - Core classes & functions
├── test_file_io.py                  # 13 tests - FASTA/mzML/MGF reading
├── test_digestion.py                # 42 tests - Protein digestion, decoys & enzymes
├── test_preprocessing.py            # 13 tests - Spectrum preprocessing & XCorr
├── test_search.py                   #  7 tests - Database search workflow
├── test_peptide_centric.py          #  8 tests - Peptide-centric scoring (mock data)
├── test_peptide_centric_real_data.py #  8 tests - Real data validation
├── test_unified_xcorr.py            # 17 tests - Unified XCorr implementation
├── test_dia_parallelization.py      #  9 tests - DIA parallel processing
└── test_evalue.py                   # 11 tests - E-value calculation & Z-scores

Total: 124 tests (100% passing)
```

## What's Being Tested

### ✓ E-value Calculation (11 tests) **NEW**

- E-value range validation (must be in [1e-10, 1.0])
- Handling insufficient data (returns 1.0)
- Uniform scores (no clear winner returns 1.0)
- E-value never exceeds 1.0 (capped for poor fits)
- E-value decreases with better score separation
- Realistic XCorr score distributions
- Charge-specific e-value calculation
- Z-score calculation for signal-to-noise ratio
- Z-score with outliers
- Z-score when no variation (returns 0.0)

### ✓ Unified XCorr Implementation (17 tests)

- Single `calculate_xcorr()` function for both vector and matrix operations
- Single spectrum scoring with correct scaling (0.005 and 0.0001)
- Matrix scoring: N peptides × M spectra using vectorized operations
- Consistency validation: matrix results match repeated single scoring
- 50x scaling factor difference validation (0.005 / 0.0001)
- Convenience wrappers: `calculate_fast_xcorr()` and `calculate_peptide_centric_xcorr()`
- Edge cases: empty arrays, mismatched lengths, 1×1 matrices
- Real data preprocessing and matrix scoring

### ✓ Core Functionality (18 tests)

- FastXCorr initialization with various parameters
- Static modifications (add, remove, affect on mass)
- MassSpectrum and PeptideCandidate classes
- Mass binning (Comet's BIN macro implementation)
- Peptide mass calculations

### ✓ File I/O (13 tests)

- Reading FASTA protein databases
- Reading mzML spectra files (with pymzml)
- Reading MGF spectra files (with pyteomics)
- Fast single spectrum extraction by scan ID
- Spectrum metadata extraction

### ✓ Protein Digestion (42 tests)

- Trypsin digestion with configurable missed cleavages
- Decoy generation (cycling and reversal methods)
- Target-decoy pair creation with collision detection
- Making peptide lists non-redundant
- Multi-enzyme support (10 enzymes: Trypsin, LysC, LysN, ArgC, AspN, CNBr, GluC, PepsinA, Chymotrypsin, Trypsin/P)
- Enzyme-specific decoy preservation rules

### ✓ Spectrum Preprocessing (13 tests)

- Complete preprocessing pipeline (binning, sqrt, windowing)
- MakeCorrData windowing normalization to 50.0
- Fast XCorr preprocessing (sliding window)
- Theoretical spectrum generation (b/y ions)
- XCorr score calculation

### ✓ Database Search (7 tests)

- End-to-end search workflow with human proteome
- Peptide m/z indexing for fast lookup
- Isolation window filtering
- E-value calculation
- Charge-state specific validation
- Reference peptide: HGKPTDSTPATWK (charge +3, XCorr ~2.7248)
- Multi-charge testing (charge +2 and +3)

### ✓ Peptide-Centric Scoring (8 tests - Mock Data)

- Validates 0.0001 scaling factor produces reasonable XCorr scores (0-10 range)
- Confirms preprocessing asymmetry (theoretical preprocessed, experimental windowed)
- Tests E-value calculation from chromatogram distribution
- Validates matrix scoring for batch processing
- Compares peptide-centric vs spectrum-centric scoring

### ✓ Peptide-Centric Real Data (8 tests - Validation)

- Uses actual peptide KIQALQQQADEAEDR from DIA analysis notebook
- Validates theoretical spectrum generation (28 fragment ions)
- Confirms preprocessing pipeline (28 → 1638 bins after Fast XCorr)
- Tests XCorr calculation with real chromatogram (39 spectra)
- Validates E-value calculation with actual score distributions
- Handles negative XCorr scores (anti-correlation)
- Explains raw dot product magnitude (~50x difference)
- Compares 0.005 vs 0.0001 scaling effects

## Current Status

All tests passing: 113/113 (100%)

Previously known issues have been resolved:

- ✓ Target-decoy mass handling - Fixed
- ✓ Integration test empty results - Fixed
- ✓ Peptide-centric scoring validation - Comprehensive tests added

## Test Data Files Used

The tests use these files from the test data directory:

- ✓ `tests/data/YQSHTK.fasta` - Small synthetic peptide FASTA (6 amino acids)
- ✓ `tests/data/YQSHTK.mzML` - mzML spectra file (charge +1 peptides)
- ✓ `tests/data/ot_centroid_8340.mgf` - MGF spectrum file (scan 8340, charge +3)
- ✓ `tests/data/uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta` - Human proteome database (20,659 proteins)

**Note:** The human proteome FASTA is used for realistic search testing with the reference peptide HGKPTDSTPATWK from protein sp|O60832|DKC1_HUMAN.

## Tips for Development

### Run Tests After Changes

```bash
# Quick check during development
python run_tests_quick.py

# Full check before commit
pytest
```

### Run Specific Test

```bash
# Run just one test function
pytest tests/test_basic_functionality.py::TestFastXCorrInitialization::test_default_initialization -v
```

### Debug Failing Tests

```bash
# Show full output including prints
pytest tests/test_search.py -v -s

# Stop at first failure
pytest -x

# Show local variables on failure
pytest --showlocals
```

### Skip Slow Tests

```bash
# Skip tests marked as slow
pytest -m "not slow"
```

## Example Test Usage

The tests show how to use pyXcorrDIA:

```python
from pyXcorrDIA import FastXCorr

# Initialize with modifications
engine = FastXCorr(static_modifications={'C': 57.021464})

# Read data
proteins = engine.read_fasta('database.fasta')
spectra = engine.read_mzml('data.mzML')

# Digest proteins
peptides = []
for protein_id, sequence in proteins.items():
    peptides.extend(engine.digest_protein(sequence, protein_id))

# Create decoys
non_redundant = engine.make_peptides_non_redundant(peptides)
pairs = engine.generate_target_decoy_pairs(non_redundant)

# Search
for spectrum in spectra:
    results = engine.search_spectrum_target_decoy(
        spectrum, pairs, charge_states=[2, 3]
    )
    # Process results...
```

## Configuration Files Created

- **`pytest.ini`** - Pytest configuration with test discovery patterns and markers
- **`tests/conftest.py`** - Shared fixtures for all tests
- **`requirements.txt`** - Updated with pytest dependencies
- **`.github_workflows_tests.yml.example`** - GitHub Actions CI/CD template

## Next Steps

1. **Run tests regularly** during development
2. **Add new tests** when adding features
3. **Check coverage** to find untested code
4. **Set up CI/CD** using the provided GitHub Actions template

## Getting Help

- See `tests/README.md` for detailed testing documentation
- Each test file has comprehensive docstrings
- Run `pytest --help` for all pytest options
- Check `conftest.py` for available fixtures

---

**Happy Testing! 🧪**

The test suite validates that pyXcorrDIA correctly implements the Comet XCorr algorithm and provides confidence in the search results.
