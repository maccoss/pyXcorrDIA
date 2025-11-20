# pyXcorrDIA Project Context

## Project Overview

pyXcorrDIA is a proteomics database search engine that implements the SEQUEST Cross-Correlation (XCorr) algorithm based on Comet's approach. It performs peptide identification from mass spectrometry data using target-decoy competition for FDR estimation.

**Key Purpose**: High-performance peptide-spectrum matching (PSM) for data-independent acquisition (DIA) proteomics experiments with spectral library support.

**Recent Performance Improvements** (November 2025): Major optimizations provide 20-40 minute speedup for DIA searches with spectral libraries through library object passing, pre-vectorized preprocessing, combined mzML reading, and quality filtering.

## Project Architecture

### Core Components

1. **FastXCorr Class** (`pyXcorrDIA.py`)
   - Main search engine implementing Comet's XCorr algorithm
   - Handles spectrum preprocessing, theoretical spectrum generation, and scoring
   - Implements target-decoy competition for FDR control
   - Unified `calculate_xcorr()` for vector and matrix operations

2. **SpectrumLibrary Class** (`pyXcorrDIA.py`)
   - DIA-NN spectral library integration
   - Automatic decoy and Q-value filtering (Decoy==0, Q.Value<=0.01)
   - Pre-vectorized fragment preprocessing (SMZ computation during load)
   - Decoy fragment generation with intensity remapping
   - Pickle-serializable for multiprocessing efficiency

3. **Spectrum Processing Pipeline**
   - Binning (1.0005079 Da bins, Comet default)
   - Square root transformation of intensities
   - MakeCorrData windowing normalization (10 windows, normalize to 50.0)
   - Fast XCorr preprocessing (sliding window with offset=75)
   - Optional SMZ preprocessing during mzML read

4. **Peptide Generation**
   - Protein digestion (trypsin with configurable missed cleavages)
   - Peptide length filtering (default: 7-30 amino acids, configurable)
   - Static modifications support (default: Carbamidomethyl-C +57.021464)
   - Decoy generation (reversal method, keeping C-terminal K/R)
   - Non-redundant peptide list creation

5. **Scoring & Statistics**
   - XCorr score calculation (dot product of preprocessed spectra)
   - Library cosine similarity with ppm fragment tolerance
   - MS1 precursor isotope pattern scoring
   - E-value calculation using Comet's LinearRegression approach
     - Histogram binning (0.1 XCorr units)
     - Cumulative distribution → log transform → linear regression
     - Valid range: [1e-10, 1.0] (capped at 1.0 for poor fits)
   - Charge-specific e-value calculation for spectrum-centric search
   - Z-score (standard score) calculation: (best_score - mean) / std_dev
   - Target-decoy competition for peptide identification

### Performance Optimizations

1. **Library Object Passing** (20-40 min speedup)
   - Load spectral library once in main process
   - Pass library object to 250 workers via pickle serialization
   - Eliminates 250× redundant parquet file reads

2. **Pre-Vectorized Preprocessing** (5-10% speedup)
   - Compute SMZ values during library loading
   - Store normalized vectors in `preprocessed_fragments`
   - Eliminates redundant computation across isolation windows

3. **Combined mzML Reading** (30-50% I/O reduction)
   - Single-pass reader for MS1 and MS2 spectra
   - `read_mzml_combined()` method checks `spectrum.ms_level`
   - Automatic for `--dia_mode --speclib` searches

4. **Quality Filtering**
   - Remove decoys from library (Decoy == 0)
   - Filter low-confidence entries (Q.Value <= 0.01)
   - Report precursor counts at each filtering stage

### File I/O Support

- **Input formats**: mzML (via pymzml), MGF (via pyteomics), FASTA, DIA-NN parquet libraries
- **Output formats**: pepXML, Percolator Input (PIN), DIA TSV with chromatograms
- Fast single-spectrum lookup using mzML indexing
- Combined MS1+MS2 reading for efficiency

## Key Implementation Details

### Comet Compatibility

This implementation closely follows Comet's algorithm to ensure reproducibility:

- **BIN macro**: `bin_idx = int(mass * inverse_bin_width + bin_offset)` where bin_offset = 0.4
- **MakeCorrData**: 10-window normalization to 50.0
- **Fast XCorr**: Sliding window (offset=75) preprocessing
- **XCorr scoring**: Unified `calculate_xcorr()` function supports both vector and matrix operations
  - DIA mode uses spectrum-centric preprocessing: 0.005 scaling factor
    - Experimental spectrum fully preprocessed (windowed + Fast XCorr background subtraction)
    - Theoretical spectrum windowed only (no Fast XCorr preprocessing)
    - More efficient when library size is large relative to spectra per window
  - Matrix multiplication: Scores N peptides × M spectra using optimized BLAS operations

### Unified XCorr Implementation

**Key Innovation**: Single `calculate_xcorr()` function eliminates duplicate code:

- **Automatic mode detection**: Handles 1D (single) or 2D (matrix) inputs automatically
- **DIA Mode**: `calculate_peptide_centric_xcorr(experimental_preprocessed, theoretical_windowed)`
  - Uses spectrum-centric preprocessing approach with 0.005 scaling
  - Experimental spectra are fully preprocessed once, then scored against all theoretical spectra
- **DIA batch scoring**: Matrix multiplication for vectorized N×M scoring
- **Convenience wrapper**: Maintains backward compatibility with existing code

### Critical Mass Calculations

- **Neutral mass**: Sum of amino acid masses + H2O (18.010565)
- **m/z calculation**: `(neutral_mass + charge * proton_mass) / charge`
- **Proton mass**: 1.007276 Da
- **Static modifications**: Applied to aa_masses dictionary, affects all calculations

### Charge State Handling

- Reads charge from mzML using MS:1000633 (possible charge state) or MS:1000041 (charge state)
- **Important**: YQSHTK test data has 1+ charge, not 2+ or 3+
- Search typically tests multiple charge states (user-configurable)

### Spectrum Library Support (DIA-NN Integration)

**Feature** (November 2025): Comprehensive support for DIA-NN predicted spectral libraries with advanced scoring.

**Core Components:**

1. **SpectrumLibrary Class** (lines 69-346)
   - Reads DIA-NN parquet format libraries (`report-lib.parquet`)
   - Indexes peptides by (sequence, charge) for fast lookup
   - Parses UniMod modifications (e.g., C(UniMod:4) → +57.021464 Da)
   - Generates decoy fragment spectra with intensity remapping
   - Note: Requires `.parquet` format, not binary `.speclib` format

2. **Fragment-Level Library Scoring**
   - **SMZ Preprocessing**: `preprocessed_intensity = sqrt(intensity) * mz²`
   - **Cosine Angle Score**: `dot(exp, lib) / (norm(exp) * norm(lib))`
   - **PPM Tolerance Matching**: Default 10 ppm for fragments
   - **Decoy Generation Strategy**: Reverse sequence (keep C-term K/R), remap fragment intensities
     - Example: PEPTIDER → EDITPEPR
     - Target y2 intensity → Decoy y2 (same position from C-terminus)
     - Recalculate m/z for new amino acid composition

3. **MS1 Precursor Isotope Scoring**
   - **Isotope Pattern Prediction**: Averagine-based model for M-1, M+0, M+1, M+2, M+3
     - M-1 intensity = 0.0 (theoretical)
     - M+0, M+1, M+2, M+3 calculated using binomial distribution
     - Neutron mass difference: 1.002868 Da / charge
   - **MS1 Spectrum Indexing**: Read and sort by retention time for binary search
   - **Envelope Extraction**: Match 5 isotope peaks with ppm tolerance (default: 10 ppm)
   - **Cosine Scoring**: Compare experimental vs theoretical isotope pattern
   - **Smart MS1 Selection**: Uses RT of best library fragment score to find closest MS1

### Target-Decoy Competition (Unified Implementation)

**Feature** (January 2025): Unified competition logic for both library and non-library DIA search modes.

**Design Philosophy:**
- Single code path eliminates duplication and ensures consistency
- Mode-specific primary score selection (LibCosine vs XCorr)
- Winner-only reporting simplifies downstream FDR calculation
- Incremental writing improves memory efficiency

**Competition Algorithm (lines 2270-2363):**

```python
def select_competition_winner(target_data, decoy_data, primary_score_key, library_mode=False):
    """
    Select winner between target and decoy based on primary score.
    
    Args:
        target_data: Dictionary with target scores and metrics
        decoy_data: Dictionary with decoy scores and metrics
        primary_score_key: 'lib_cosine' (library) or 'xcorr' (non-library)
        library_mode: If True, XCorr calculated only at primary score peak
    
    Returns:
        winner_data: Dictionary with winner's scores and metrics
        is_target: 1 if target won, 0 if decoy won
    """
```

**Primary Score Selection:**
- **Library mode** (`--speclib` provided):
  - Primary score: `LibCosine` (cosine similarity to library spectrum)
  - XCorr calculated only at the single spectrum with best LibCosine
  - E-value set to 0.0 (not meaningful with single XCorr measurement)
  - Rationale: LibCosine already captures spectral similarity, XCorr is confirmatory
  
- **Non-library mode** (no `--speclib`):
  - Primary score: `XCorr` (cross-correlation score)
  - XCorr calculated across full chromatogram (all spectra in window)
  - E-value meaningful (calculated from XCorr distribution)
  - Rationale: XCorr is primary metric, e-value provides statistical significance

**Winner Selection Rules:**
1. Compare primary scores (LibCosine or XCorr) between target and decoy
2. Higher primary score wins
3. **Tie-breaking**: Decoy wins (conservative for FDR estimation)
4. Winner's metrics reported at primary score peak location

**Output Format:**

*Library Mode* (13 columns):
```
Peptide, Charge, ProteinID, Mass, IsTarget, IsolationWindow, NumSpectraScored,
LibCosine, LibCosineZScore, XCorr, RT, ScanID, PrecursorCosine
```

*Non-Library Mode* (12 columns):
```
Peptide, Charge, ProteinID, Mass, IsTarget, IsolationWindow, BestXCorr,
BestRT, BestScan, EValue, NumSpectraScored, XCorrZScore
```

**Key Simplifications:**
- **No PairID**: Winner-only reporting eliminates need for pair tracking
- **No Smoothing Artifacts**: Removed BestXCorrRaw, BestXCorrSmoothed columns
- **Clean Column Names**: `LibCosine`, `XCorr`, `RT`, `ScanID` (no "Best" prefixes)
- **Conditional Columns**: E-value and XCorrZScore only in non-library mode

**Incremental TSV Writing (lines 3456-3520):**

```python
class DIAResultsWriter:
    def __init__(self, filepath, library_mode=False):
        self.write_lock = None  # Set by caller for parallel mode
        
    def open_for_append(self):
        self.file_handle = open(self.filepath, 'a')
        
    def write_dia_results_synchronized(self, results):
        if self.write_lock:
            with self.write_lock:
                self._write_results(results)
        else:
            self._write_results(results)
```

**Benefits:**
- **Memory Efficient**: Write immediately, don't accumulate all results
- **Real-Time Progress**: "Progress: X/Y windows completed" updates
- **Thread-Safe**: `multiprocessing.Manager().Lock()` prevents corruption
- **Streaming**: Uses `pool.imap_unordered()` instead of `pool.map()`

**Command-Line Usage:**

```bash
# Library mode search
python pyXcorrDIA.py --dia_mode \
    --speclib DIANN-Output/report-lib.parquet \
    --lib_fragment_tol 10.0 \
    --lib_fragment_tol_unit ppm \
    --lib_precursor_tol 10.0 \
    --lib_precursor_tol_unit ppm \
    database.fasta \
    data.mzML
```

**Key Features:**
- Parallel processing support (library loaded per-worker)
- Backward compatible (runs without library, outputs "NA")
- Z-score calculation for library scores: `(best_score - mean) / std_dev`
- Integrates seamlessly with existing XCorr scoring

**Workflow:**
1. Load library from parquet file (per parallel worker)
2. Read MS1 spectra and index by RT (once at startup)
3. For each isolation window:
   - Score all library fragments against all MS2 spectra
   - Generate and score decoy fragments
   - Find best scoring spectrum
   - Extract isotope envelope from closest MS1 scan
   - Calculate precursor isotope cosine scores
4. Output chromatograms and summary statistics

## Testing Infrastructure

### Test Organization (`tests/`)

- **test_basic_functionality.py** (18 tests) - Core classes, modifications, mass calculations
- **test_file_io.py** (13 tests) - FASTA, mzML, MGF reading
- **test_digestion.py** (42 tests) - Protein digestion, decoys, multi-enzyme support
- **test_preprocessing.py** (13 tests) - Spectrum preprocessing, XCorr calculation
- **test_search.py** (7 tests) - Database search workflow, integration tests
- **test_peptide_centric.py** (8 tests) - Peptide-centric DIA scoring with mock data
- **test_peptide_centric_real_data.py** (8 tests) - Real data validation from notebook
- **test_integration.py** (19 tests) - Integration tests and DIA parallelization
- **test_unified_xcorr.py** (6 tests) - Unified XCorr function (single & matrix operations)
- **test_evalue.py** (11 tests) - E-value calculation and Z-score validation
- **test_library_support.py** (11 tests) - Spectrum library integration
  - Library loading and indexing
  - UniMod modification parsing
  - Decoy fragment generation with intensity remapping
  - Fragment matching with ppm tolerance
  - Cosine angle scoring with SMZ preprocessing
  - MS1 isotope pattern prediction and scoring
  - MS1 spectrum management and envelope extraction
- **test_target_decoy_competition.py** (7 tests) - **NEW**: Target/decoy competition & incremental writing
  - Unified competition logic for library and non-library modes
  - LibCosine as primary score in library mode
  - XCorr as primary score in non-library mode
  - Winner-only reporting (no PairID column)
  - Simplified TSV output format validation
  - Incremental writing with thread-safe file access
  - Conditional e-value calculation (non-library mode only)

**Total: 163 tests (100% passing)**

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
