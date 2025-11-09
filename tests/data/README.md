# Test Data Files

This directory contains test data files used by the pyXcorrDIA test suite.

## Files

### YQSHTK.fasta
Small FASTA protein database file for basic testing. Contains a minimal set of proteins to test digestion and search functionality quickly.

**Used by:**
- `test_file_io.py::TestFASTAReading::test_read_yqshtk_fasta`
- `test_digestion.py::TestProteinDigestion::test_digest_yqshtk_sequence`
- `test_search.py::TestDatabaseSearch::test_yqshtk_search`
- `test_search.py::TestDatabaseSearch::test_search_with_modifications`
- `test_search.py::TestIntegrationWorkflow::test_complete_workflow_yqshtk`

### YQSHTK.mzML
mzML format mass spectrometry data file containing MS/MS spectra. Used for testing spectrum reading and search functionality.

**Used by:**
- `test_file_io.py::TestMzMLReading` (all tests)
- `test_file_io.py::TestSingleSpectrumReading::test_read_single_spectrum_mzml`
- `test_preprocessing.py::TestFullPreprocessingPipeline::test_yqshtk_spectrum_preprocessing`
- `test_search.py::TestDatabaseSearch` (all tests)

### ot_centroid_8340.mgf
MGF (Mascot Generic Format) file containing centroided MS/MS spectra. Used for testing MGF file reading capabilities.

**Used by:**
- `test_file_io.py::TestMGFReading` (all tests)
- `test_file_io.py::TestSingleSpectrumReading::test_read_single_spectrum_mgf`
- `test_preprocessing.py::TestFullPreprocessingPipeline::test_mgf_spectrum_preprocessing`

## Purpose

These test files serve several purposes:

1. **Unit Testing** - Verify that file reading functions work correctly
2. **Integration Testing** - Test complete search workflows with known data
3. **Regression Testing** - Ensure changes don't break existing functionality
4. **Documentation** - Provide examples of expected input formats

## File Format Information

### FASTA Format
```
>ProteinID Description
PEPTIDESEQUENCE
```

### mzML Format
XML-based format for mass spectrometry data. Contains:
- Spectrum metadata (scan number, precursor m/z, charge)
- Peak lists (m/z and intensity arrays)
- Instrument information

### MGF Format
Text-based format commonly used in proteomics. Contains:
```
BEGIN IONS
TITLE=Spectrum_ID
PEPMASS=precursor_mz
CHARGE=charge_state
mz1 intensity1
mz2 intensity2
...
END IONS
```

## Adding New Test Data

If you need to add new test data files:

1. Place the file in this directory
2. Add a fixture in `tests/conftest.py` to provide the file path
3. Document the file in this README
4. Reference the fixture in your test functions

Example fixture:
```python
@pytest.fixture(scope="session")
def my_test_file(test_data_dir):
    """Path to my test file."""
    file_path = test_data_dir / "my_test_file.mzML"
    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")
    return str(file_path)
```

## Notes

- These files are small and specifically chosen for fast test execution
- Larger database files (like `uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta`) remain in the project root since they're also used for actual searches, not just testing
- All test files should be committed to version control to ensure consistent test results across environments
