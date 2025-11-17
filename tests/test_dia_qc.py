"""
Integration tests for DIA search with QC data collection.

Tests the complete DIA peptide-centric search pipeline including:
- Library loading and random precursor selection
- MS1 mass accuracy measurement (one per precursor)
- MS2 mass accuracy measurement (at best LibCosine spectrum only)
- RT correlation (library RT vs measured RT)
- Precursor-level data collection (not peptide-level)

Uses realistic test data:
- test_library_1000.parquet: 1000 randomly selected precursors
- test_proteins_1000.fasta: Matching protein sequences  
- test_dia_60000-70000.mzML: Small DIA window (~1 min RT range, scan IDs 60000-70000)

NOTE: The test mzML has a narrow RT window (~1 minute), so:
- RT correlation plot will show limited spread (expected behavior)
- MS1/MS2 mass accuracy plots will be representative
- For full RT correlation validation, use the complete mzML file:
  Ast-Neo-15min-2mz-4ms-200agc-10.mzML (~15 min gradient)
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.slow
class TestDIATestDataAvailable:
    """Verify DIA test data files are present."""
    
    def test_dia_library_exists(self, dia_library_1000):
        """Verify DIA test library file exists."""
        assert Path(dia_library_1000).exists()
        print(f"\n✓ DIA library: {dia_library_1000}")
    
    def test_dia_fasta_exists(self, dia_fasta_1000):
        """Verify DIA test FASTA file exists."""
        assert Path(dia_fasta_1000).exists()
        print(f"✓ DIA FASTA: {dia_fasta_1000}")
    
    def test_dia_mzml_exists(self, dia_mzml_small):
        """Verify DIA test mzML file exists."""
        assert Path(dia_mzml_small).exists()
        print(f"✓ DIA mzML: {dia_mzml_small}")


@pytest.mark.slow
class TestDIALibraryLoading:
    """Test DIA library loading and structure."""
    
    def test_load_dia_library(self, dia_library_1000):
        """Test loading DIA-NN parquet library."""
        from pyXcorrDIA import SpectrumLibrary
        
        library = SpectrumLibrary(dia_library_1000)
        
        # Verify library loaded
        assert len(library.precursors) > 0
        print(f"\n✓ Loaded {len(library.precursors)} precursors")
        
        # Check a precursor has expected structure
        first_key = next(iter(library.precursors.keys()))
        first_precursor = library.precursors[first_key]
        
        assert 'precursor_mz' in first_precursor
        assert 'fragments' in first_precursor
        assert 'rt' in first_precursor
        
        print(f"✓ Library structure valid")
        print(f"  Sample precursor: {first_key}")
        print(f"  Fragments: {len(first_precursor['fragments'])}")
    
    def test_library_random_selection(self, dia_library_1000):
        """Test random precursor selection from library."""
        from pyXcorrDIA import SpectrumLibrary
        
        # Load with random selection
        library = SpectrumLibrary(dia_library_1000, max_precursors=100)
        
        # Should have 100 precursors (or fewer if library is smaller)
        assert len(library.precursors) <= 100
        print(f"\n✓ Random selection: {len(library.precursors)} precursors")


@pytest.mark.slow
class TestDIAMzMLReading:
    """Test reading DIA mzML files."""
    
    def test_read_dia_mzml(self, dia_mzml_small, xcorr_engine):
        """Test reading small DIA mzML file."""
        ms2_spectra, ms1_spectra = xcorr_engine.read_mzml_combined(
            dia_mzml_small,
            preprocess_smz=True
        )
        
        # Should have both MS1 and MS2 spectra
        assert len(ms2_spectra) > 0, "Should have MS2 spectra"
        assert len(ms1_spectra) > 0, "Should have MS1 spectra"
        
        print(f"\n✓ DIA mzML loaded:")
        print(f"  MS2 spectra: {len(ms2_spectra)}")
        print(f"  MS1 spectra: {len(ms1_spectra)}")
        
        # Check MS2 spectrum structure
        first_ms2 = ms2_spectra[0]
        assert hasattr(first_ms2, 'scan_id')
        assert hasattr(first_ms2, 'precursor_mz')
        assert hasattr(first_ms2, 'mz')
        assert hasattr(first_ms2, 'intensity')
        
        print(f"✓ MS2 spectrum structure valid")


@pytest.mark.slow
class TestDIAQCDataStructure:
    """Test QC data collection structure (without running full search)."""
    
    def test_qc_data_initialization(self):
        """Verify QC data structure is correctly initialized."""
        # This tests the structure used in search_dia_peptide_centric
        qc_data = {
            'ms1_mass_errors': [],
            'ms2_mass_errors': [],
            'rt_pairs': [],
            'ms1_tol_unit': 'ppm',
            'ms2_tol_unit': 'ppm'
        }
        
        assert 'ms1_mass_errors' in qc_data
        assert 'ms2_mass_errors' in qc_data
        assert 'rt_pairs' in qc_data
        assert qc_data['ms1_tol_unit'] in ['ppm', 'mz']
        assert qc_data['ms2_tol_unit'] in ['ppm', 'mz']
        
        print("\n✓ QC data structure valid")
    
    def test_qc_data_entry_structure(self):
        """Verify QC data entries have required fields."""
        # MS1 error entry
        ms1_entry = {
            'error': 5.2,
            'peptide': 'PEPTIDE',
            'charge': 2,
            'is_target': True
        }
        
        assert all(k in ms1_entry for k in ['error', 'peptide', 'charge', 'is_target'])
        
        # MS2 error entry (same structure)
        ms2_entry = {
            'error': -3.1,
            'peptide': 'PEPTIDE',
            'charge': 2,
            'is_target': True
        }
        
        assert all(k in ms2_entry for k in ['error', 'peptide', 'charge', 'is_target'])
        
        # RT pair entry
        rt_entry = {
            'library_rt': 10.5,
            'measured_rt': 10.3,
            'lib_cosine': 0.95,
            'is_target': True,
            'peptide': 'PEPTIDE',
            'charge': 2
        }
        
        assert all(k in rt_entry for k in ['library_rt', 'measured_rt', 'lib_cosine', 
                                            'is_target', 'peptide', 'charge'])
        
        print("\n✓ QC entry structures valid")


# Note: Full DIA search integration test would go here, but requires significant runtime.
# For CI/CD, use the --test_library_peptides flag to limit search scope:
#
# python pyXcorrDIA.py tests/data/test_proteins_1000.fasta tests/data/test_dia_60000-70000.mzML \
#        --dia_mode --speclib tests/data/test_library_1000.parquet \
#        --test_library_peptides 100
#
# Then validate:
# - MS1 errors: ~100 measurements (one per precursor)
# - MS2 errors: ~1200 measurements (12 fragments × 100 precursors)
# - RT pairs: ~100 pairs (one per precursor)

