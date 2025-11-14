# pyXcorrDIA

A proteomics database search engine implementing the SEQUEST Cross-Correlation (XCorr) algorithm based on Comet's approach. Designed for peptide-spectrum matching with target-decoy competition for FDR estimation.

## Features

- **Unified XCorr calculation** - Single function for both vector and matrix operations, eliminating code duplication
- **Comet-compatible algorithm** - Faithful implementation matching Comet's preprocessing and scoring
- **DIA peptide-centric mode** - Optimized search for data-independent acquisition with RT profiling
- **Spectral library support** - Search against DIA-NN spectral libraries with cosine similarity scoring
- **Unified target-decoy competition** - Winner-only reporting for both library and non-library modes
- **Incremental TSV writing** - Real-time results output with thread-safe file access for memory efficiency
- **Vectorized matrix scoring** - N×M peptide-spectrum scoring using optimized BLAS operations
- **Multi-enzyme support** - 10 protease digestion options including Trypsin, Lys-C, Arg-C, and more
- **Fast spectrum preprocessing** - Efficient binning, windowing normalization, and Fast XCorr calculation
- **Target-decoy search** - Built-in decoy generation and target-decoy competition
- **Multiple file formats** - Supports mzML (via pymzml) and MGF (via pyteomics) input
- **Flexible modifications** - Configurable static modifications (default: Carbamidomethyl-C)
- **Multiple output formats** - pepXML and Percolator Input (PIN) files for spectrum-centric; simplified TSV for DIA
- **E-value calculation** - Comet's LinearRegression approach with proper statistical significance (non-library mode)
- **Z-score reporting** - Signal-to-noise ratio for chromatographic peaks in DIA mode
- **Comprehensive testing** - 163 tests covering all major functionality including competition logic and incremental writing

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/maccoss/pyXcorrDIA.git
cd pyXcorrDIA

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Search a mass spectrometry file against a protein database:

```bash
python pyXcorrDIA.py database.fasta spectra.mzML
```

This will create:
- `spectra.pepXML` - pepXML format results
- `spectra.pin` - Percolator Input format results

### Common Usage Examples

**Search with specific charge states:**
```bash
python pyXcorrDIA.py database.fasta spectra.mzML --charge_states 2,3,4
```

**Custom output files:**
```bash
python pyXcorrDIA.py database.fasta spectra.mzML \
    --output results.pepXML \
    --pin_output results.pin
```

**Add additional static modifications:**
```bash
# Carbamidomethyl-C (default) + Oxidation on M
python pyXcorrDIA.py database.fasta spectra.mzML \
    --static_mods "C:57.021464,M:15.994915"
```

**Process limited number of spectra (for testing):**
```bash
python pyXcorrDIA.py database.fasta spectra.mzML --max_spectra 100
```

**Report more PSMs per spectrum:**
```bash
python pyXcorrDIA.py database.fasta spectra.mzML --top_hits 20
```

**Use different enzyme for digestion:**
```bash
# Lys-C digestion
python pyXcorrDIA.py database.fasta spectra.mzML --enzyme lysc

# Arg-C digestion
python pyXcorrDIA.py database.fasta spectra.mzML --enzyme argc

# Glu-C digestion
python pyXcorrDIA.py database.fasta spectra.mzML --enzyme gluc
```

**Control peptide length range:**
```bash
# Use defaults (7-30 amino acids)
python pyXcorrDIA.py database.fasta spectra.mzML

# Custom range for shorter peptides
python pyXcorrDIA.py database.fasta spectra.mzML \
    --min_peptide_length 6 \
    --max_peptide_length 50

# Stringent range for longer, more confident peptides
python pyXcorrDIA.py database.fasta spectra.mzML \
    --min_peptide_length 8 \
    --max_peptide_length 20
```

### Test Data

Try the included test data:

```bash
python pyXcorrDIA.py YQSHTK.fasta YQSHTK.mzML --charge_states 1 --min_peptide_length 6
```

**Note:** YQSHTK is only 6 amino acids, so you need `--min_peptide_length 6` to include it (default is 7).

## DIA Peptide-Centric Search Mode

pyXcorrDIA supports **DIA (Data-Independent Acquisition) peptide-centric search**, optimized for analyzing DIA data where multiple MS/MS spectra share the same precursor isolation window.

### DIA vs Standard Mode

| Mode | Preprocessing | Use Case | Output |
|------|---------------|----------|--------|
| **Spectrum-Centric** (default) | Experimental spectra preprocessed | DDA, single peptide/spectrum | Best peptides per spectrum |
| **Peptide-Centric** (--dia_mode) | Theoretical spectra preprocessed | DIA, multiple spectra/window | Best XCorr per peptide across RT |

### DIA Usage

**Basic DIA search:**
```bash
python pyXcorrDIA.py database.fasta dia_data.mzML --dia_mode
```

**DIA with custom parameters:**
```bash
python pyXcorrDIA.py database.fasta dia_data.mzML \
    --dia_mode \
    --dia_output results.dia.tsv \
    --dia_rt_window 10 \
    --charge_states 2,3,4
```

### DIA Workflow

1. **Group spectra** by isolation window
2. **Preprocess theoretical spectra** for peptides in each window (binning → MakeCorrData → Fast XCorr)
3. **Score each peptide** across all spectra in the window
4. **Target-decoy competition**: Select winner based on primary score (LibCosine in library mode, XCorr in non-library mode)
5. **Incremental writing**: Write results immediately as each window completes (memory efficient)
6. **Output** peptide-level results with unified competition scoring

### DIA Output Format

The output format differs between library and non-library search modes:

#### Library Mode Output (`--speclib` provided)

Tab-delimited TSV with 13 columns (one row per peptide - winner only):

| Column | Description |
|--------|-------------|
| `Peptide` | Peptide sequence |
| `Charge` | Charge state |
| `ProteinID` | Source protein identifier |
| `Mass` | Neutral peptide mass |
| `IsTarget` | 1 for target, 0 for decoy (winner only) |
| `IsolationWindow` | Precursor m/z window `[lower-upper]` |
| `NumSpectraScored` | Number of spectra in RT window |
| `LibCosine` | Library cosine similarity score (primary score) |
| `LibCosineZScore` | Z-score of LibCosine across chromatogram |
| `XCorr` | XCorr at the spectrum with best LibCosine |
| `RT` | Retention time at best LibCosine peak |
| `ScanID` | Scan ID at best LibCosine peak |
| `PrecursorCosine` | MS1 precursor isotope pattern similarity |

**Note:** In library mode, XCorr is calculated only at the single spectrum with the best LibCosine score, so e-value calculation is not meaningful.

#### Non-Library Mode Output (no `--speclib`)

Tab-delimited TSV with 12 columns (one row per peptide - winner only):

| Column | Description |
|--------|-------------|
| `Peptide` | Peptide sequence |
| `Charge` | Charge state |
| `ProteinID` | Source protein identifier |
| `Mass` | Neutral peptide mass |
| `IsTarget` | 1 for target, 0 for decoy (winner only) |
| `IsolationWindow` | Precursor m/z window `[lower-upper]` |
| `BestXCorr` | Highest XCorr across chromatogram (primary score) |
| `BestRT` | Retention time at best XCorr |
| `BestScan` | Scan ID at best XCorr |
| `EValue` | Statistical significance from XCorr distribution |
| `NumSpectraScored` | Number of spectra in RT window |
| `XCorrZScore` | Z-score of best XCorr relative to distribution |

**Note:** In non-library mode, XCorr is calculated across the full chromatogram, making e-value calculation meaningful.

### DIA Mode Notes

- **Winner-only reporting**: Each peptide reported once (target OR decoy, whichever wins)
- **Performance**: Faster for true DIA data (theoretical spectra reused across RT)
- **Memory**: Incremental writing reduces memory usage (results written immediately)
- **FDR**: Use `IsTarget` field to compute false discovery rate from winner-only results
- **Library mode**: Primary score is LibCosine, XCorr is confirmatory
- **Non-library mode**: Primary score is XCorr with full statistical significance (e-value)
- **Thread safety**: Results writing uses multiprocessing.Lock() for safe parallel processing
- **Best for**: Data with multiple spectra per isolation window

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `fasta_file` | FASTA protein database (positional) | Required |
| `mzml_file` | mzML or MGF spectrum file (positional) | Required |
| `-o, --output` | Output pepXML file path | `{mzml}.pepXML` |
| `-p, --pin_output` | Percolator PIN file path | `{mzml}.pin` |
| `--dia_mode` | Enable DIA peptide-centric search | Off |
| `--dia_output` | DIA results output file | `{mzml}.dia.tsv` |
| `--speclib` | Path to DIA-NN spectral library (.tsv) | None |
| `--dia_rt_window` | RT window in seconds for DIA grouping | 10.0 |
| `-t, --threads` | Number of threads (0 = auto) | 0 |
| `-v, --verbose` | Increase output verbosity | 0 |
| `-n, --top_hits` | Top PSMs to report per spectrum | 10 |
| `-m, --max_spectra` | Max spectra to process (0 = all) | 0 |
| `-c, --charge_states` | Charge states to search (comma-separated) | `2,3` |
| `-e, --enzyme` | Enzyme for protein digestion | `trypsin` |
| `--missed_cleavages` | Number of missed cleavages allowed | 1 |
| `--min_peptide_length` | Minimum peptide length (amino acids) | 7 |
| `--max_peptide_length` | Maximum peptide length (amino acids) | 30 |
| `-d, --decoy_cycle_length` | Amino acids to cycle for decoys | 1 |
| `-s, --static_mods` | Static mods as `AA:mass` pairs | `C:57.021464` |
| `-bw, --bin_width` | Mass bin width (Th) | 1.0005079 |
| `-bo, --bin_offset` | Bin offset for binning calculation | 0.4 |

## Enzyme Support

pyXcorrDIA supports 10 different proteolytic enzymes for protein digestion:

| Enzyme | Cleavage Specificity | Common Use |
|--------|----------------------|------------|
| **trypsin** | After K, R (NOT before P) | Default, most common |
| **trypsin_no_proline** | After K, R (including before P) | Without proline suppression |
| **lysc** | After K | High specificity |
| **lysn** | Before K | N-terminal labeling |
| **argc** | After R | Alternative to trypsin |
| **aspn** | Before D | Acid-rich regions |
| **cnbr** | After M | Chemical cleavage |
| **gluc** | After D, E | Acidic peptides |
| **pepsina** | After F, L | Low pH digestion |
| **chymotrypsin** | After F, W, Y, L | Large hydrophobic peptides |

**Note:** The default enzyme is `trypsin` (trypsin with proline suppression), which follows the standard proteomics rule where cleavage does NOT occur when proline follows the cleavage site.

### Enzyme-Aware Decoy Generation

Decoy sequences are generated with enzyme-specific terminal residue preservation:
- **C-terminal cleavage** (e.g., trypsin, Lys-C): Preserves C-terminal residue
- **N-terminal cleavage** (e.g., Lys-N, Asp-N): Preserves N-terminal residue
- This ensures decoys have the same enzyme-specific properties as targets

## Algorithm Details

### XCorr Implementation

pyXcorrDIA implements the Fast XCorr algorithm as described in Eng et al. (2008) and implemented in Comet:

1. **Spectrum Binning** - 1.0005079 Da bins (Comet default)
2. **Intensity Transformation** - Square root normalization
3. **MakeCorrData Windowing** - 10-window normalization to 50.0
4. **Fast XCorr Preprocessing** - Sliding window background subtraction (offset=75)
5. **Theoretical Spectrum Generation** - Fragment ions for peptide candidates
6. **XCorr Scoring** - Dot product of preprocessed spectra
7. **E-value Calculation (Non-Library Mode Only)** - Comet's LinearRegression approach
   - Histogram binning (0.1 XCorr units)
   - Cumulative distribution from right to left
   - Log transformation and linear regression
   - E-value range: [1e-10, 1.0] (capped at 1.0 for poor fits)
   - Spectrum-centric: charge-specific score distributions
   - Peptide-centric: distribution of spectrum scores per peptide
   - **Not calculated in library mode** (XCorr only at single spectrum with best LibCosine)
8. **Z-score Calculation (DIA mode)** - Signal-to-noise ratio
   - Z-score = (best_score - mean_score) / std_dev
   - Measures how many standard deviations the peak is above background
   - Library mode: LibCosineZScore across chromatogram
   - Non-library mode: XCorrZScore from score distribution

### Target-Decoy Competition

pyXcorrDIA uses unified target-decoy competition for both library and non-library modes:

- **Primary Score Selection**:
  - Library mode: LibCosine (cosine similarity to library spectrum)
  - Non-library mode: XCorr (cross-correlation score)
  
- **Winner Selection**:
  - Compare primary scores between target and decoy
  - Higher primary score wins
  - Ties favor decoy (conservative for FDR estimation)
  
- **Winner-Only Reporting**:
  - One peptide per target/decoy pair (50% fewer rows than reporting both)
  - `IsTarget` field indicates if winner was target (1) or decoy (0)
  - No PairID needed (simplified output)
  
- **Metrics at Winner's Peak**:
  - Library mode: XCorr, RT, ScanID at spectrum with best LibCosine
  - Non-library mode: XCorr, RT, ScanID, and e-value at best XCorr

### Incremental TSV Writing

DIA mode writes results progressively as isolation windows complete:

- **Memory Efficiency**: Results written immediately, not accumulated
- **Real-Time Progress**: "Progress: X/Y windows completed" updates
- **Thread Safety**: multiprocessing.Lock() prevents file corruption in parallel mode
- **Sequential Mode**: Write after each window completes
- **Parallel Mode**: pool.imap_unordered() streams results from workers

### Decoy Generation

- Reversal method (reverse peptide sequence)
- Enzyme-aware terminal residue preservation (see Enzyme Support above)
- Mass recalculated to ensure target-decoy mass matching
- Target-decoy competition ensures proper FDR control

## Python API Usage

You can also use pyXcorrDIA as a Python library:

```python
from pyXcorrDIA import FastXCorr

# Initialize search engine
engine = FastXCorr(static_modifications={'C': 57.021464})

# Read and digest database
proteins = engine.read_fasta('database.fasta')
peptides = []
for protein_id, sequence in proteins.items():
    peptides.extend(engine.digest_protein(
        sequence, protein_id, 
        enzyme='trypsin', 
        missed_cleavages=2
    ))

# Generate target-decoy pairs
non_redundant = engine.make_peptides_non_redundant(peptides)
target_decoy_pairs = engine.generate_target_decoy_pairs(non_redundant)

# Read spectra and search
spectra = engine.read_mzml('spectra.mzML')
for spectrum in spectra:
    results = engine.search_spectrum_target_decoy(
        spectrum, 
        target_decoy_pairs, 
        charge_states=[2, 3]
    )
    # Process results...
```

See `XCorr_Preprocessing_Analysis.ipynb` for more detailed examples.

## Testing

pyXcorrDIA has a comprehensive test suite with **156 tests** covering all major functionality:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=pyXcorrDIA --cov-report=html

# Quick validation script
python run_tests_quick.py

# Run specific test modules
pytest tests/test_search.py -v              # Database search
pytest tests/test_evalue.py -v              # E-value calculation
pytest tests/test_unified_xcorr.py -v       # Unified XCorr function
pytest tests/test_dia_parallelization.py -v # DIA parallel processing
pytest tests/test_library_search.py -v      # Spectral library search
```

### Test Coverage

- **18 tests** - Core functionality (initialization, modifications, mass calculations)
- **13 tests** - File I/O (FASTA, mzML, MGF reading)
- **42 tests** - Protein digestion, decoys, and multi-enzyme support
- **13 tests** - Spectrum preprocessing and XCorr calculation
- **7 tests** - Database search workflow with real data
- **8 tests** - Peptide-centric scoring (mock data)
- **8 tests** - Peptide-centric scoring (real data validation)
- **17 tests** - Unified XCorr implementation (vector and matrix operations)
- **9 tests** - DIA parallelization
- **11 tests** - E-value calculation and Z-scores
- **Additional tests** - Spectral library search and other features

See `TESTING_QUICKSTART.md` for quick start guide and `TEST_INFRASTRUCTURE_SUMMARY.md` for detailed testing documentation.

## Output Formats

### pepXML
Standard pepXML format containing:
- Search parameters
- Peptide-spectrum matches
- XCorr scores
- E-values
- Decoy information

### Percolator Input (PIN)
Tab-separated format for Percolator post-processing:
- Feature columns for machine learning
- Target/decoy labels
- Peptide and protein information

## Performance Notes

- **Memory efficient** - Streaming spectrum processing
- **Fast indexing** - mzML random access for quick spectrum lookup
- **Optimized binning** - NumPy-based array operations
- **Typical speed** - ~1000-5000 spectra/minute (depends on database size)

## Known Limitations

- Static modifications only (no variable modifications yet)
- No PTM localization scoring
- Single precursor tolerance window

## Documentation

- **README.md** (this file) - Project overview and usage
- **tests/README.md** - Comprehensive testing guide
- **Claude.md** - Project context for AI assistants
- **XCorr_Preprocessing_Analysis.ipynb** - Interactive algorithm exploration

## Contributing

Contributions welcome! Please:
1. Run tests before submitting: `pytest`
2. Follow existing code style: `ruff check pyXcorrDIA.py`
3. Add tests for new features
4. Update documentation

## License

See LICENSE file for details.

## Citation

If you use pyXcorrDIA in your research, please cite:

- Eng JK, et al. (2008) "A Fast SEQUEST Cross Correlation Algorithm" *J Proteome Res* 7(10):4598-4602
- Comet: http://comet-ms.sourceforge.net/

## Contact

Project maintained by the MacCoss Lab: https://github.com/maccoss

## References

- **Comet**: http://comet-ms.sourceforge.net/
- **SEQUEST**: Eng JK, McCormack AL, Yates JR (1994) *J Am Soc Mass Spectrom* 5(11):976-989
- **Fast XCorr**: Eng JK, et al. (2008) *J Proteome Res* 7(10):4598-4602
- **Target-Decoy**: Elias JE, Gygi SP (2007) *Nat Methods* 4(3):207-214
- **DIA**: Gillet LC, et al. (2012) *Mol Cell Proteomics* 11(6):O111.016717

