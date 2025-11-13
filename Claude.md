# pyXcorrDIA Project Context

## Project Overview

pyXcorrDIA is a fast proteomics database search engine that implements the SEQUEST Cross-Correlation (XCorr) algorithm based on Comet's approach. It performs peptide identification from mass spectrometry data using target-decoy competition for FDR estimation.

**Key Purpose**: High-performance peptide-spectrum matching (PSM) for data-independent acquisition (DIA) proteomics experiments.

## Project Architecture

### Core Components

1. **FastXCorr Class** (`pyXcorrDIA.py`)
   - Main search engine implementing Comet's XCorr algorithm
   - Handles spectrum preprocessing, theoretical spectrum generation, and scoring
   - Implements target-decoy competition for FDR control

2. **Spectrum Processing Pipeline**
   - Binning (1.0005079 Da bins, Comet default)
   - Square root transformation of intensities
   - MakeCorrData windowing normalization (10 windows, normalize to 50.0)
   - Fast XCorr preprocessing (sliding window with offset=75)

3. **Peptide Generation**
   - Protein digestion (trypsin with configurable missed cleavages)
   - Peptide length filtering (default: 7-30 amino acids, configurable)
   - Static modifications support (default: Carbamidomethyl-C +57.021464)
   - Decoy generation (reversal method, keeping C-terminal K/R)
   - Non-redundant peptide list creation

4. **Scoring & Statistics**
   - XCorr score calculation (dot product of preprocessed spectra)
   - E-value calculation using Comet's LinearRegression approach
     - Histogram binning (0.1 XCorr units)
     - Cumulative distribution → log transform → linear regression
     - Valid range: [1e-10, 1.0] (capped at 1.0 for poor fits)
   - Charge-specific e-value calculation for spectrum-centric search
   - Z-score (standard score) calculation: (best_score - mean) / std_dev
   - Target-decoy competition for peptide identification

### File I/O Support

- **Input formats**: mzML (via pymzml), MGF (via pyteomics), FASTA
- **Output formats**: pepXML, Percolator Input (PIN)
- Fast single-spectrum lookup using mzML indexing

## Key Implementation Details

### Comet Compatibility

This implementation closely follows Comet's algorithm to ensure reproducibility:

- **BIN macro**: `bin_idx = int(mass * inverse_bin_width + bin_offset)` where bin_offset = 0.4
- **MakeCorrData**: 10-window normalization to 50.0
- **Fast XCorr**: Sliding window (offset=75) preprocessing
- **XCorr scoring**: Unified `calculate_xcorr()` function supports both vector and matrix operations
  - Spectrum-centric: 0.005 scaling factor (Comet standard)
  - Peptide-centric: 0.0001 scaling factor (50x smaller due to preprocessing asymmetry)
  - Matrix multiplication: Scores N peptides × M spectra using optimized BLAS operations

### Unified XCorr Implementation

**Key Innovation**: Single `calculate_xcorr()` function eliminates duplicate code:

- **Automatic mode detection**: Handles 1D (single) or 2D (matrix) inputs automatically
- **Spectrum-centric**: `calculate_fast_xcorr(theoretical_raw, experimental_preprocessed, 0.005)`
- **Peptide-centric**: `calculate_peptide_centric_xcorr(experimental_windowed, theoretical_preprocessed, 0.0001)`
- **DIA batch scoring**: Matrix multiplication for vectorized N×M scoring
- **Convenience wrappers**: Maintain backward compatibility with existing code

**Why two scaling factors?**
- Preprocessing the **experimental** spectrum (spectrum-centric): Lower signal amplification → 0.005 scaling
- Preprocessing the **theoretical** spectrum (peptide-centric): ~50x higher raw scores → 0.0001 scaling
- Both produce comparable final XCorr scores in the 0-10 range

### Critical Mass Calculations

- **Neutral mass**: Sum of amino acid masses + H2O (18.010565)
- **m/z calculation**: `(neutral_mass + charge * proton_mass) / charge`
- **Proton mass**: 1.007276 Da
- **Static modifications**: Applied to aa_masses dictionary, affects all calculations

### Charge State Handling

- Reads charge from mzML using MS:1000633 (possible charge state) or MS:1000041 (charge state)
- **Important**: YQSHTK test data has 1+ charge, not 2+ or 3+
- Search typically tests multiple charge states (user-configurable)

## Testing Infrastructure

### Test Organization (`tests/`)

- **test_basic_functionality.py** (18 tests) - Core classes, modifications, mass calculations
- **test_file_io.py** (13 tests) - FASTA, mzML, MGF reading
- **test_digestion.py** (42 tests) - Protein digestion, decoys, multi-enzyme support
- **test_preprocessing.py** (13 tests) - Spectrum preprocessing, XCorr calculation
- **test_search.py** (7 tests) - Database search workflow, integration tests
- **test_peptide_centric.py** (8 tests) - Peptide-centric DIA scoring with mock data
- **test_peptide_centric_real_data.py** (8 tests) - Real data validation from notebook
- **test_unified_xcorr.py** (17 tests) - Unified XCorr function (single & matrix operations)
- **test_dia_parallelization.py** (9 tests) - DIA parallel processing validation
- **test_evalue.py** (11 tests) - **NEW**: E-value calculation and Z-score validation

**Total: 124 tests (100% passing)**

### E-value & Z-score Tests (NEW)

**`test_evalue.py`** validates statistical significance calculations:

- E-value range bounds [1e-10, 1.0]
- Proper handling of insufficient data and uniform scores
- E-value capping at 1.0 for poor regression fits
- Charge-specific e-value calculation
- Z-score calculation: (best_score - mean) / std_dev
- Z-score behavior with outliers and no variation

### Unified XCorr Tests

**`test_unified_xcorr.py`** validates the unified XCorr implementation:

- Single spectrum scoring with correct scaling (0.005 and 0.0001)
- Matrix scoring: N×M operations using vectorized multiplication
- Consistency: Matrix results exactly match repeated single scoring
- 50x scaling factor difference (0.005 / 0.0001 = 50)
- Convenience wrappers return floats (not ndarrays)
- Edge cases: empty arrays, mismatched lengths, 1×1 matrices
- Real preprocessing and matrix operations

### Test Data (`tests/data/`)

- **YQSHTK.fasta** - Small test database (3 proteins, all YQSHTK variants)
- **YQSHTK.mzML** - Single MS/MS spectrum, **1+ charge**, precursor m/z 763.3733
- **ot_centroid_8340.mgf** - MGF format test spectrum

### Running Tests

```bash
# Quick validation
python run_tests_quick.py

# All tests
pytest

# With coverage
pytest --cov=pyXcorrDIA --cov-report=html

# Specific module
pytest tests/test_preprocessing.py -v

# Peptide-centric tests with output
pytest tests/test_peptide_centric_real_data.py -v -s

# Unified XCorr tests
pytest tests/test_unified_xcorr.py -v
```

### Test Documentation

**When adding new tests:**
1. Update **TEST_INFRASTRUCTURE_SUMMARY.md** with detailed test descriptions
2. Update **TESTING_QUICKSTART.md** with quick usage examples
3. Use descriptive test names that explain what is being validated
4. Include docstrings explaining test purpose and expected outcomes
5. For real data tests, reference the source notebook or data file

## Common Development Tasks

### Adding New Static Modifications

```python
engine = FastXCorr()
engine.add_static_modification('M', 15.994915)  # Oxidation
engine.add_static_modification('K', 8.014199)   # SILAC
```

### Reading Spectra

```python
# Read all MS2 spectra
spectra = engine.read_mzml('data.mzML')

# Read limited number
spectra = engine.read_mzml('data.mzML', max_spectra=100)

# Fast single spectrum lookup
spectrum = engine.read_single_spectrum('data.mzML', 'scan_1234')
```

### Complete Search Workflow

```python
# 1. Initialize with modifications
engine = FastXCorr(static_modifications={'C': 57.021464})

# 2. Read and digest database
proteins = engine.read_fasta('database.fasta')
peptides = []
for protein_id, sequence in proteins.items():
    peptides.extend(engine.digest_protein(sequence, protein_id, 
                                         enzyme='trypsin', 
                                         missed_cleavages=2,
                                         min_length=7,
                                         max_length=30))

# 3. Make non-redundant and generate decoys
non_redundant = engine.make_peptides_non_redundant(peptides)
target_decoy_pairs = engine.generate_target_decoy_pairs(non_redundant)

# 4. Read spectra and search
spectra = engine.read_mzml('data.mzML')
for spectrum in spectra:
    results = engine.search_spectrum_target_decoy(spectrum, 
                                                  target_decoy_pairs, 
                                                  charge_states=[2, 3])
```

## Important Conventions

### Variable Naming
- `spectrum` - MassSpectrum object
- `peptide` - PeptideCandidate object
- `xcorr_engine` or `engine` - FastXCorr instance
- `target_decoy_pairs` - List of (target, decoy) tuples

### Method Patterns
- `read_*` methods - Return data structures from files
- `generate_*` methods - Create new objects (theoretical spectra, decoys)
- `calculate_*` methods - Compute numerical values
- `preprocess_*` methods - Transform spectra

## Known Issues & Quirks

1. **Charge state defaults**: If charge not found in mzML, defaults to 2+ (line 230 in pyXcorrDIA.py)
2. **Test data specifics**: YQSHTK test data is 1+ charge, not 2+ or 3+
3. **pymzml warnings**: "term not found (MS:1000031)" is harmless (missing instrument model)
4. **Decoy mass**: Always recalculated, so masses match between target and decoy
5. **Bin offset**: Default 0.4 matches Comet; can be changed for different binning strategies

## Dependencies

### Core
- numpy - Array operations
- pymzml - mzML file reading
- pyteomics - MGF file reading, additional MS utilities

### Testing
- pytest - Test framework
- pytest-cov - Coverage reporting

### Optional
- matplotlib - Visualization (notebook)
- pandas - Data analysis (notebook)

## Documentation Files

- **README.md** - Project overview and usage
- **TESTING_QUICKSTART.md** - Quick guide to running tests
- **TEST_INFRASTRUCTURE_SUMMARY.md** - Detailed test suite documentation
- **tests/README.md** - Comprehensive testing guide
- **tests/data/README.md** - Test data file descriptions

## Command-Line Usage

```bash
# Basic search
python pyXcorrDIA.py database.fasta spectra.mzML

# With options
python pyXcorrDIA.py database.fasta spectra.mzML \
    --output results.pepXML \
    --pin_output results.pin \
    --charge_states 2,3 \
    --top_hits 10 \
    --static_mods "C:57.021464,M:15.994915"
```

## Performance Considerations

- **Binning**: 1.0005079 Da bins means ~2000 bins for 0-2000 Da range
- **Memory**: Theoretical spectra cached per peptide-charge combination
- **Speed**: Fast XCorr preprocessing is O(n) in spectrum length
- **Isolation window**: Binary search used for fast peptide filtering

## Jupyter Notebook

**XCorr_Preprocessing_Analysis.ipynb** - Interactive analysis and validation
- Visualizes preprocessing steps
- Tests XCorr functions
- Demonstrates search workflow
- Useful for debugging and understanding algorithm

## Future Development Notes

When modifying the code:
1. Run tests after changes: `pytest`
2. Check that XCorr scores remain consistent with Comet
3. Update test fixtures if changing default modifications
4. Test with real data, not just YQSHTK test case
5. Consider charge state handling for different instruments

## Getting Help

- Tests serve as usage examples
- Notebook demonstrates interactive usage
- Comet source code for algorithm questions
- Test failures often indicate charge state or modification issues
