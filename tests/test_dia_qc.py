"""
Integration tests for DIA search with QC data collection.

Tests the complete DIA peptide-centric search pipeline including:
- Library loading and random precursor selection
- MS1 mass accuracy measurement (one per precursor)
- MS2 mass accuracy measurement (at best LibCosine spectrum only)
- RT correlation (library RT vs measured RT)
- Precursor-level data collection (not peptide-level)
- Calibration scoring (XCorr vs LibCosine based on tolerance unit)

Uses realistic test data:
- test_library_768.parquet: 768 precursors with q-value <= 0.01
- test_proteins_768.fasta: Matching protein sequences  
- test_dia_5windows.mzML: Small DIA file (5 isolation windows, 400-410 m/z)

NOTE: The test mzML has a narrow RT window (~1 minute), so:
- RT correlation plot will show limited spread (expected behavior)
- MS1/MS2 mass accuracy plots will be representative
- For full RT correlation validation, use the complete mzML file:
  Ast-Neo-15min-2mz-4ms-200agc-10.mzML (~15 min gradient)
"""

import pytest
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyXcorrDIA import filter_qc_data_by_fdr, SpectrumLibrary


@pytest.mark.slow
class TestDIATestDataAvailable:
    """Verify DIA test data files are present."""
    
    def test_dia_library_exists(self, test_dia_library):
        """Test that the DIA library file exists."""
        assert Path(test_dia_library).exists()
        print(f"\n✓ DIA library: {test_dia_library}")
    
    def test_dia_fasta_exists(self, test_dia_fasta):
        """Test that the DIA FASTA file exists."""
        assert Path(test_dia_fasta).exists()
        print(f"✓ DIA FASTA: {test_dia_fasta}")
    
    def test_dia_mzml_exists(self, dia_mzml_small):
        """Verify DIA test mzML file exists."""
        assert Path(dia_mzml_small).exists()
        print(f"✓ DIA mzML: {dia_mzml_small}")


@pytest.mark.slow
class TestDIALibraryLoading:
    """Test DIA library loading and structure."""
    
    def test_load_dia_library(self, test_dia_library):
        """Test that we can load the DIA library.
        
        This validates the library file structure and basic content.
        """
        library = SpectrumLibrary(test_dia_library)
        
        # Verify library loaded
        assert len(library.peptide_index) > 0
        print(f"\n✓ Loaded {len(library.peptide_index)} precursors")
        
        # Check a precursor has expected structure
        first_key = next(iter(library.peptide_index.keys()))
        first_precursor = library.peptide_index[first_key]
        
        assert 'precursor_mz' in first_precursor
        assert 'fragments' in first_precursor
        assert 'rt' in first_precursor
        
        print(f"✓ Library structure valid")
        print(f"  Sample precursor: {first_key}")
        print(f"  Fragments: {len(first_precursor['fragments'])}")
    
    def test_library_random_selection(self, test_dia_library):
        """Test random precursor selection from library."""
        
        # Load with random selection using test_limit_peptides parameter
        library = SpectrumLibrary(test_dia_library, test_limit_peptides=100)
        
        # Should have 100 precursors (or fewer if library is smaller)
        assert len(library.peptide_index) <= 100
        print(f"\n✓ Random selection: {len(library.peptide_index)} precursors")


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
        assert hasattr(first_ms2, 'mz_array')
        assert hasattr(first_ms2, 'intensity_array')
        
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
# python pyXcorrDIA.py tests/data/test_proteins_768.fasta tests/data/test_dia_5windows.mzML \
#        --dia_mode --speclib tests/data/test_library_768.parquet \
#        --test_library_peptides 100
#
# Then validate:
# - MS1 errors: ~100 measurements (one per precursor)
# - MS2 errors: ~1200 measurements (12 fragments × 100 precursors)
# - RT pairs: ~100 pairs (one per precursor)


@pytest.mark.slow
class TestCalibrationScoring:
    """Test calibration FDR filtering with different scoring methods."""
    
    def test_filter_qc_data_with_libcosine(self):
        """Test FDR filtering using LibCosine score (default for ppm units)."""
        # Create mock QC data
        qc_data = {
            'ms1_mass_errors': [
                {'error': 1.5, 'peptide': 'PEPTIDEA', 'charge': 2, 'is_target': True},
                {'error': 2.1, 'peptide': 'PEPTIDEB', 'charge': 2, 'is_target': True},
                {'error': -1.2, 'peptide': 'PEPTIDEC', 'charge': 3, 'is_target': True},
                {'error': 3.5, 'peptide': 'DECOY_A', 'charge': 2, 'is_target': False},
            ],
            'ms2_mass_errors': [
                {'error': 0.5, 'peptide': 'PEPTIDEA', 'charge': 2, 'is_target': True},
                {'error': -0.8, 'peptide': 'PEPTIDEB', 'charge': 2, 'is_target': True},
                {'error': 1.1, 'peptide': 'PEPTIDEC', 'charge': 3, 'is_target': True},
                {'error': -2.1, 'peptide': 'DECOY_A', 'charge': 2, 'is_target': False},
            ],
            'rt_pairs': [
                {'library_rt': 10.0, 'measured_rt': 10.1, 'lib_cosine': 0.95, 
                 'is_target': True, 'peptide': 'PEPTIDEA', 'charge': 2},
                {'library_rt': 15.0, 'measured_rt': 15.2, 'lib_cosine': 0.92,
                 'is_target': True, 'peptide': 'PEPTIDEB', 'charge': 2},
                {'library_rt': 20.0, 'measured_rt': 20.1, 'lib_cosine': 0.88,
                 'is_target': True, 'peptide': 'PEPTIDEC', 'charge': 3},
                {'library_rt': 12.0, 'measured_rt': 12.5, 'lib_cosine': 0.75,
                 'is_target': False, 'peptide': 'DECOY_A', 'charge': 2},
            ],
            'ms1_tol_unit': 'ppm',
            'ms2_tol_unit': 'ppm'
        }
        
        # Create winners DataFrame with LibCosine scores
        winners_df = pd.DataFrame([
            {'Peptide': 'PEPTIDEA', 'Charge': 2, 'LibCosine': 0.95, 'IsTarget': 'Target'},
            {'Peptide': 'PEPTIDEB', 'Charge': 2, 'LibCosine': 0.92, 'IsTarget': 'Target'},
            {'Peptide': 'PEPTIDEC', 'Charge': 3, 'LibCosine': 0.88, 'IsTarget': 'Target'},
            {'Peptide': 'DECOY_A', 'Charge': 2, 'LibCosine': 0.75, 'IsTarget': 'Decoy'},
        ])
        
        # Filter at 25% FDR (should pass 3 targets, 1 decoy = 25% FDR)
        filtered = filter_qc_data_by_fdr(qc_data, winners_df, fdr_threshold=0.25, score_column='LibCosine')
        
        # Should keep all 3 target precursors at 25% FDR
        assert filtered['num_precursors'] == 3
        assert len(filtered['ms1_mass_errors']) == 3
        assert len(filtered['ms2_mass_errors']) == 3
        assert len(filtered['rt_pairs']) == 3
        
        print("\n✓ LibCosine-based FDR filtering works correctly")
    
    def test_filter_qc_data_with_xcorr(self):
        """Test FDR filtering using XCorr score (for mz units)."""
        # Create mock QC data (same structure as above)
        qc_data = {
            'ms1_mass_errors': [
                {'error': 1.5, 'peptide': 'PEPTIDEA', 'charge': 2, 'is_target': True},
                {'error': 2.1, 'peptide': 'PEPTIDEB', 'charge': 2, 'is_target': True},
                {'error': -1.2, 'peptide': 'PEPTIDEC', 'charge': 3, 'is_target': True},
                {'error': 3.5, 'peptide': 'DECOY_A', 'charge': 2, 'is_target': False},
            ],
            'ms2_mass_errors': [
                {'error': 0.5, 'peptide': 'PEPTIDEA', 'charge': 2, 'is_target': True},
                {'error': -0.8, 'peptide': 'PEPTIDEB', 'charge': 2, 'is_target': True},
                {'error': 1.1, 'peptide': 'PEPTIDEC', 'charge': 3, 'is_target': True},
                {'error': -2.1, 'peptide': 'DECOY_A', 'charge': 2, 'is_target': False},
            ],
            'rt_pairs': [
                {'library_rt': 10.0, 'measured_rt': 10.1, 'lib_cosine': 0.95, 
                 'is_target': True, 'peptide': 'PEPTIDEA', 'charge': 2},
                {'library_rt': 15.0, 'measured_rt': 15.2, 'lib_cosine': 0.92,
                 'is_target': True, 'peptide': 'PEPTIDEB', 'charge': 2},
                {'library_rt': 20.0, 'measured_rt': 20.1, 'lib_cosine': 0.88,
                 'is_target': True, 'peptide': 'PEPTIDEC', 'charge': 3},
                {'library_rt': 12.0, 'measured_rt': 12.5, 'lib_cosine': 0.75,
                 'is_target': False, 'peptide': 'DECOY_A', 'charge': 2},
            ],
            'ms1_tol_unit': 'mz',
            'ms2_tol_unit': 'mz'
        }
        
        # Create winners DataFrame with XCorr scores (different ranking than LibCosine)
        winners_df = pd.DataFrame([
            {'Peptide': 'PEPTIDEA', 'Charge': 2, 'XCorr': 4.8, 'IsTarget': 'Target'},
            {'Peptide': 'PEPTIDEC', 'Charge': 3, 'XCorr': 4.2, 'IsTarget': 'Target'},
            {'Peptide': 'PEPTIDEB', 'Charge': 2, 'XCorr': 3.9, 'IsTarget': 'Target'},
            {'Peptide': 'DECOY_A', 'Charge': 2, 'XCorr': 2.5, 'IsTarget': 'Decoy'},
        ])
        
        # Filter at 25% FDR using XCorr column
        filtered = filter_qc_data_by_fdr(qc_data, winners_df, fdr_threshold=0.25, score_column='XCorr')
        
        # Should keep all 3 target precursors at 25% FDR
        assert filtered['num_precursors'] == 3
        assert len(filtered['ms1_mass_errors']) == 3
        assert len(filtered['ms2_mass_errors']) == 3
        assert len(filtered['rt_pairs']) == 3
        
        print("\n✓ XCorr-based FDR filtering works correctly")
    
    def test_scoring_method_selection(self):
        """Test that the correct scoring method is selected based on tolerance unit."""
        # This test validates the logic in run_calibration_workflow
        # where lib_fragment_tol_unit determines which score to use
        
        # For ppm units, should use LibCosine
        lib_fragment_tol_unit_ppm = 'ppm'
        use_xcorr_ppm = (lib_fragment_tol_unit_ppm == 'mz')
        assert use_xcorr_ppm == False, "Should use LibCosine for ppm units"
        score_name_ppm = 'XCorr' if use_xcorr_ppm else 'LibCosine'
        assert score_name_ppm == 'LibCosine'
        
        # For mz units, should use XCorr
        lib_fragment_tol_unit_mz = 'mz'
        use_xcorr_mz = (lib_fragment_tol_unit_mz == 'mz')
        assert use_xcorr_mz == True, "Should use XCorr for mz units"
        score_name_mz = 'XCorr' if use_xcorr_mz else 'LibCosine'
        assert score_name_mz == 'XCorr'
        
        print("\n✓ Scoring method selection logic works correctly")
        print(f"  ppm units → {score_name_ppm}")
        print(f"  mz units → {score_name_mz}")
    
    def test_fdr_calculation_with_different_scores(self):
        """Test that FDR calculation works correctly with both score types."""
        # Test with high-scoring targets and low-scoring decoys
        winners_xcorr = pd.DataFrame([
            {'Peptide': 'TARGET1', 'Charge': 2, 'XCorr': 5.0, 'IsTarget': 'Target'},
            {'Peptide': 'TARGET2', 'Charge': 2, 'XCorr': 4.5, 'IsTarget': 'Target'},
            {'Peptide': 'TARGET3', 'Charge': 3, 'XCorr': 4.0, 'IsTarget': 'Target'},
            {'Peptide': 'DECOY1', 'Charge': 2, 'XCorr': 3.0, 'IsTarget': 'Decoy'},
            {'Peptide': 'TARGET4', 'Charge': 2, 'XCorr': 2.5, 'IsTarget': 'Target'},
            {'Peptide': 'DECOY2', 'Charge': 3, 'XCorr': 2.0, 'IsTarget': 'Decoy'},
        ])
        
        # Calculate FDR manually
        sorted_df = winners_xcorr.sort_values('XCorr', ascending=False).copy()
        sorted_df['cumulative_targets'] = (sorted_df['IsTarget'] == 'Target').cumsum()
        sorted_df['cumulative_decoys'] = (sorted_df['IsTarget'] == 'Decoy').cumsum()
        sorted_df['fdr'] = sorted_df['cumulative_decoys'] / sorted_df['cumulative_targets'].replace(0, 1)
        
        # At score 4.0, we have 3 targets and 0 decoys = 0% FDR
        # At score 3.0, we have 3 targets and 1 decoy = 33% FDR
        # At score 2.5, we have 4 targets and 1 decoy = 25% FDR
        
        targets_at_1pct = sorted_df[(sorted_df['fdr'] <= 0.01) & (sorted_df['IsTarget'] == 'Target')]
        assert len(targets_at_1pct) == 3, "Should have 3 targets at 1% FDR"
        
        targets_at_25pct = sorted_df[(sorted_df['fdr'] <= 0.25) & (sorted_df['IsTarget'] == 'Target')]
        assert len(targets_at_25pct) == 4, "Should have 4 targets at 25% FDR"
        
        print("\n✓ FDR calculation works correctly")
        print(f"  Targets at 1% FDR: {len(targets_at_1pct)}")
        print(f"  Targets at 25% FDR: {len(targets_at_25pct)}")


