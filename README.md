# pyXcorrDIA

A fast proteomics database search engine implementing the SEQUEST Cross-Correlation (XCorr) algorithm based on Comet's approach. Designed for peptide-spectrum matching with target-decoy competition for FDR estimation.

## Features

- **Comet-compatible XCorr algorithm** - Faithful implementation matching Comet's preprocessing and scoring
- **Fast spectrum preprocessing** - Efficient binning, windowing normalization, and Fast XCorr calculation
- **Target-decoy search** - Built-in decoy generation and target-decoy competition
- **Multiple file formats** - Supports mzML (via pymzml) and MGF (via pyteomics) input
- **Flexible modifications** - Configurable static modifications (default: Carbamidomethyl-C)
- **Multiple output formats** - pepXML and Percolator Input (PIN) files
- **E-value calculation** - Statistical scoring with charge-specific score distributions
- **Comprehensive testing** - 56 tests covering all major functionality

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

### Test Data

Try the included test data:

```bash
python pyXcorrDIA.py YQSHTK.fasta YQSHTK.mzML --charge_states 1
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `fasta_file` | FASTA protein database (positional) | Required |
| `mzml_file` | mzML or MGF spectrum file (positional) | Required |
| `-o, --output` | Output pepXML file path | `{mzml}.pepXML` |
| `-p, --pin_output` | Percolator PIN file path | `{mzml}.pin` |
| `-n, --top_hits` | Top PSMs to report per spectrum | 10 |
| `-m, --max_spectra` | Max spectra to process (0 = all) | 0 |
| `-c, --charge_states` | Charge states to search (comma-separated) | `2,3` |
| `-s, --static_mods` | Static mods as `AA:mass` pairs | `C:57.021464` |
| `-d, --decoy_cycle_length` | Amino acids to cycle for decoys | 1 |
| `-bw, --bin_width` | Mass bin width (Th) | 1.0005079 |
| `-bo, --bin_offset` | Bin offset for binning calculation | 0.4 |

## Algorithm Details

### XCorr Implementation

pyXcorrDIA implements the Fast XCorr algorithm as described in Eng et al. (2008) and implemented in Comet:

1. **Spectrum Binning** - 1.0005079 Da bins (Comet default)
2. **Intensity Transformation** - Square root normalization
3. **MakeCorrData Windowing** - 10-window normalization to 50.0
4. **Fast XCorr Preprocessing** - Sliding window background subtraction (offset=75)
5. **Theoretical Spectrum Generation** - Fragment ions for peptide candidates
6. **XCorr Scoring** - Dot product of preprocessed spectra
7. **E-value Calculation** - Charge-specific score distributions

### Decoy Generation

- Reversal method (reverse peptide sequence)
- Keeps C-terminal K/R in place for tryptic peptides
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

pyXcorrDIA has a comprehensive test suite with 56 tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pyXcorrDIA --cov-report=html

# Quick validation script
python run_tests_quick.py

# Run specific test module
pytest tests/test_search.py -v
```

See `tests/README.md` for detailed testing documentation.

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
- Trypsin digestion only
- No PTM localization
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

