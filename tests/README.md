# Test Suite for pyXcorrDIA

This directory contains the pytest-based test suite for pyXcorrDIA.

## Overview

The test suite is organized into multiple test modules, each focusing on specific functionality:

- **`test_basic_functionality.py`**: Tests for core classes and basic operations
  - FastXCorr initialization
  - MassSpectrum and PeptideCandidate classes
  - Binning functions
  - Static modifications management
  - Peptide mass calculations

- **`test_file_io.py`**: Tests for file reading operations
  - FASTA file reading
  - mzML file reading
  - MGF file reading
  - Single spectrum extraction

- **`test_digestion.py`**: Tests for protein digestion and peptide generation
  - Trypsin digestion
  - Missed cleavages
  - Decoy generation (cycling and reversal)
  - Target-decoy pair creation
  - Peptide non-redundancy

- **`test_preprocessing.py`**: Tests for spectrum preprocessing and XCorr
  - Spectrum preprocessing pipeline
  - MakeCorrData windowing
  - Fast XCorr preprocessing
  - Theoretical spectrum generation
  - XCorr score calculation

- **`test_search.py`**: Tests for database search functionality
  - End-to-end search workflow
  - Peptide indexing
  - Isolation window filtering
  - E-value calculation
  - Integration tests

## Test Data

The tests use the following test data files:

**In `tests/data/` directory:**
- `YQSHTK.fasta` - Small FASTA file for basic testing
- `YQSHTK.mzML` - mzML spectra file
- `ot_centroid_8340.mgf` - MGF spectra file

**In project root:**
- `uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta` - Larger database for integration testing (also used for actual searches)

## Running Tests

### Install Test Dependencies

First, ensure pytest is installed:

```bash
pip install pytest pytest-cov
```

Or install all requirements including test dependencies:

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# From the project root directory
pytest

# Or from anywhere with explicit path
pytest tests/
```

### Run Specific Test Files

```bash
# Run only basic functionality tests
pytest tests/test_basic_functionality.py

# Run only file I/O tests
pytest tests/test_file_io.py

# Run only preprocessing tests
pytest tests/test_preprocessing.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_basic_functionality.py::TestFastXCorrInitialization

# Run a specific test function
pytest tests/test_basic_functionality.py::TestFastXCorrInitialization::test_default_initialization
```

### Run with Verbose Output

```bash
# Show more details
pytest -v

# Show even more details including print statements
pytest -v -s
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=pyXcorrDIA --cov-report=html

# View the HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Skip Slow Tests

Some tests are marked as slow (integration tests). To skip them:

```bash
pytest -m "not slow"
```

### Run Only Fast Unit Tests

```bash
pytest -m unit
```

## Test Markers

Tests can be marked with the following markers:

- `@pytest.mark.slow` - Integration tests that take longer to run
- `@pytest.mark.integration` - End-to-end integration tests
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.file_io` - Tests that require test data files

## Writing New Tests

### Test Structure

Follow the existing patterns:

```python
"""
Module docstring describing what is being tested.
"""
import pytest
from pyXcorrDIA import FastXCorr

class TestFeatureName:
    """Test a specific feature or component."""
    
    def test_specific_behavior(self, xcorr_engine):
        """Test a specific behavior with descriptive name."""
        # Arrange
        # ... setup
        
        # Act
        result = xcorr_engine.some_method()
        
        # Assert
        assert result is not None
        assert result > 0
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

- `xcorr_engine` - FastXCorr with default settings
- `xcorr_engine_with_mods` - FastXCorr with Carbamidomethyl-C
- `xcorr_engine_no_mods` - FastXCorr with no modifications
- `simple_spectrum` - A basic test spectrum
- `sample_peptide` - A simple test peptide
- `yqshtk_fasta` - Path to YQSHTK FASTA file
- `yqshtk_mzml` - Path to YQSHTK mzML file
- `ot_centroid_mgf` - Path to MGF file
- `large_fasta` - Path to large FASTA database

### Best Practices

1. **Test one thing per test** - Each test should verify a single behavior
2. **Use descriptive names** - Test names should describe what is being tested
3. **Follow AAA pattern** - Arrange, Act, Assert
4. **Use fixtures** - Reuse common setup via fixtures
5. **Add docstrings** - Explain what the test verifies
6. **Check edge cases** - Test boundary conditions and error cases
7. **Keep tests independent** - Tests should not depend on each other
8. **Print debug info** - Use print statements to help with debugging failures

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest --cov=pyXcorrDIA --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Not Found

If pytest can't find tests, ensure:
- You're running from the project root directory
- Test files are named `test_*.py`
- Test functions are named `test_*`
- Test classes are named `Test*`

### Import Errors

If you see import errors:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Missing Test Data

If tests fail due to missing test data files:
- Ensure test data files are in the `tests/data/` directory
- Check that fixture paths in `conftest.py` are correct
- Tests will skip automatically if data files are missing
- See `tests/data/README.md` for information about test data files

## Contact

For questions or issues with the test suite, please open an issue on GitHub.
