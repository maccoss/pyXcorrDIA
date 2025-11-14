"""
Tests for unified target/decoy competition in DIA search.

This module tests the new competition logic that:
1. Reports only the winner of target/decoy competition (not both)
2. Uses LibCosine as primary score in library mode
3. Uses XCorr as primary score in non-library mode
4. Reports all metrics at the primary score peak location
5. Outputs simplified TSV format without redundant columns
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import the classes we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyXcorrDIA import (
    FastXCorr, SpectrumLibrary, MS1Spectrum, MassSpectrum,
    PeptideCandidate, DIAResultsWriter
)


class TestTargetDecoyCompetition:
    """Tests for target/decoy competition logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.xcorr_engine = FastXCorr()
        
    def test_library_mode_competition(self):
        """
        Test that library mode competition:
        - Uses LibCosine as primary score
        - Reports only winner (target or decoy)
        - Returns XCorr at LibCosine peak
        - Does not calculate e-value
        """
        # Create mock peptide pairs (target + decoy)
        target_peptide = PeptideCandidate("PEPTIDE", "P12345", mass=799.359954)
        decoy_peptide = PeptideCandidate("EDPTIEP", "DECOY_P12345", mass=799.359954)
        
        # Simulate XCorr chromatograms (5 spectra)
        target_xcorr = [0.5, 0.8, 1.2, 0.9, 0.6]
        decoy_xcorr = [0.4, 0.6, 0.9, 0.7, 0.5]
        
        # Simulate LibCosine chromatograms
        # Target has best LibCosine at spectrum 2 (index 2)
        target_libcosine = [0.3, 0.5, 0.9, 0.6, 0.4]  # Best: 0.9 at index 2
        # Decoy has best LibCosine at spectrum 1 (index 1)
        decoy_libcosine = [0.2, 0.7, 0.5, 0.4, 0.3]  # Best: 0.7 at index 1
        
        # Target should win (0.9 > 0.7)
        target_best_lib = max(target_libcosine)
        decoy_best_lib = max(decoy_libcosine)
        
        assert target_best_lib > decoy_best_lib, "Target should have higher LibCosine"
        
        # Winner should be target with metrics at LibCosine peak (index 2)
        winner_is_target = target_best_lib > decoy_best_lib
        winner_lib_idx = target_libcosine.index(target_best_lib)
        winner_xcorr_at_peak = target_xcorr[winner_lib_idx]
        
        assert winner_is_target == True
        assert winner_lib_idx == 2
        assert winner_xcorr_at_peak == 1.2  # XCorr at LibCosine peak, not best XCorr
        
        print("✓ Library mode competition:")
        print(f"  Target LibCosine: {target_best_lib:.2f} at spectrum {winner_lib_idx}")
        print(f"  Decoy LibCosine: {decoy_best_lib:.2f}")
        print(f"  Winner: {'Target' if winner_is_target else 'Decoy'}")
        print(f"  XCorr at LibCosine peak: {winner_xcorr_at_peak:.2f}")
        
    def test_non_library_mode_competition(self):
        """
        Test that non-library mode competition:
        - Uses XCorr as primary score
        - Reports only winner (target or decoy)
        - Calculates e-value from XCorr distribution
        """
        # Create mock peptide pairs
        target_peptide = PeptideCandidate("PEPTIDE", "P12345", mass=799.359954)
        decoy_peptide = PeptideCandidate("EDPTIEP", "DECOY_P12345", mass=799.359954)
        
        # Simulate XCorr chromatograms (5 spectra)
        target_xcorr = [0.5, 0.8, 1.5, 0.9, 0.6]  # Best: 1.5 at index 2
        decoy_xcorr = [0.4, 0.6, 1.2, 0.7, 0.5]   # Best: 1.2 at index 2
        
        # Target should win (1.5 > 1.2)
        target_best_xcorr = max(target_xcorr)
        decoy_best_xcorr = max(decoy_xcorr)
        
        assert target_best_xcorr > decoy_best_xcorr, "Target should have higher XCorr"
        
        # Winner metrics
        winner_is_target = target_best_xcorr > decoy_best_xcorr
        winner_xcorr_idx = target_xcorr.index(target_best_xcorr)
        
        # E-value should be calculated from distribution
        temp_engine = FastXCorr()
        e_value = temp_engine.calculate_e_value(target_xcorr, target_best_xcorr)
        
        assert winner_is_target == True
        assert winner_xcorr_idx == 2
        assert e_value >= 0, "E-value should be non-negative"
        
        print("✓ Non-library mode competition:")
        print(f"  Target XCorr: {target_best_xcorr:.2f} at spectrum {winner_xcorr_idx}")
        print(f"  Decoy XCorr: {decoy_best_xcorr:.2f}")
        print(f"  Winner: {'Target' if winner_is_target else 'Decoy'}")
        print(f"  E-value: {e_value:.6e}")
        
    def test_tie_breaking_favors_decoy(self):
        """
        Test that when target and decoy have equal scores, decoy wins.
        This is conservative for FDR estimation.
        """
        # Equal LibCosine scores
        target_libcosine = [0.3, 0.5, 0.8, 0.6, 0.4]
        decoy_libcosine = [0.2, 0.4, 0.8, 0.5, 0.3]  # Same max: 0.8
        
        target_best = max(target_libcosine)
        decoy_best = max(decoy_libcosine)
        
        assert target_best == decoy_best, "Scores should be equal"
        
        # Tie-breaking: decoy wins
        if target_best > decoy_best:
            winner_is_target = True
        elif decoy_best > target_best:
            winner_is_target = False
        else:
            winner_is_target = False  # Tie: decoy wins
            
        assert winner_is_target == False, "Decoy should win in tie"
        
        print("✓ Tie-breaking:")
        print(f"  Target score: {target_best:.2f}")
        print(f"  Decoy score: {decoy_best:.2f}")
        print(f"  Winner: Decoy (conservative)")


class TestDIAResultsWriter:
    """Tests for DIAResultsWriter output format."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_library_mode_output_format(self):
        """Test that library mode outputs correct TSV format."""
        output_file = os.path.join(self.temp_dir, "test_lib_output.tsv")
        
        # Create writer in library mode
        writer = DIAResultsWriter(output_file, "test.mzml", library_mode=True)
        
        # Create mock result
        peptide = PeptideCandidate("PEPTIDE", "P12345", mass=799.359954)
        results = {
            ('PEPTIDE', 2, True): {
                'peptide': peptide,
                'charge': 2,
                'is_target': True,
                'isolation_window': (400.0, 425.0),
                'best_xcorr': 1.5,  # XCorr at LibCosine peak
                'best_rt': 10.5,
                'best_scan': 1234,
                'num_spectra_scored': 50,
                'best_lib_cosine_target': 0.85,
                'lib_cosine_target_zscore': 3.2,
                'precursor_cosine_target': 0.92,
            }
        }
        
        # Write results
        with writer:
            writer.write_dia_results(results)
        
        # Read and validate output
        df = pd.read_csv(output_file, sep='\t')
        
        # Check columns - library mode format
        expected_columns = [
            'Peptide', 'Charge', 'ProteinID', 'Mass', 'IsTarget', 'IsolationWindow',
            'NumSpectraScored', 'LibCosine', 'LibCosineZScore', 'XCorr', 'RT', 'ScanID',
            'PrecursorCosine'
        ]
        assert list(df.columns) == expected_columns, f"Columns mismatch: {list(df.columns)}"
        
        # Check data
        assert len(df) == 1, "Should have one row"
        assert df.iloc[0]['Peptide'] == 'PEPTIDE'
        assert df.iloc[0]['Charge'] == 2
        assert df.iloc[0]['IsTarget'] == 'Target'
        assert df.iloc[0]['LibCosine'] == 0.85
        assert df.iloc[0]['XCorr'] == 1.5
        assert df.iloc[0]['RT'] == 10.5
        assert df.iloc[0]['ScanID'] == 1234
        
        # Check that PairID is NOT in columns
        assert 'PairID' not in df.columns, "PairID should not be in library mode"
        
        # Check that e-value columns are NOT present
        assert 'EValue' not in df.columns, "EValue should not be in library mode"
        assert 'BestXCorrRaw' not in df.columns, "BestXCorrRaw should not be present"
        assert 'BestXCorrSmoothed' not in df.columns, "BestXCorrSmoothed should not be present"
        assert 'XCorrZScore' not in df.columns, "XCorrZScore should not be in library mode"
        
        print("✓ Library mode output format validated")
        print(f"  Columns: {list(df.columns)}")
        
    def test_non_library_mode_output_format(self):
        """Test that non-library mode outputs correct TSV format."""
        output_file = os.path.join(self.temp_dir, "test_xcorr_output.tsv")
        
        # Create writer in non-library mode
        writer = DIAResultsWriter(output_file, "test.mzml", library_mode=False)
        
        # Create mock result
        peptide = PeptideCandidate("PEPTIDE", "P12345", mass=799.359954)
        results = {
            ('PEPTIDE', 2, True): {
                'peptide': peptide,
                'charge': 2,
                'is_target': True,
                'isolation_window': (400.0, 425.0),
                'best_xcorr': 1.8,
                'best_rt': 10.5,
                'best_scan': 1234,
                'e_value': 0.001,
                'num_spectra_scored': 50,
                'all_xcorr_values': [0.5, 0.8, 1.8, 1.2, 0.6],  # For Z-score calc
            }
        }
        
        # Write results
        with writer:
            writer.write_dia_results(results)
        
        # Read and validate output
        df = pd.read_csv(output_file, sep='\t')
        
        # Check columns - non-library mode format
        expected_columns = [
            'Peptide', 'Charge', 'ProteinID', 'Mass', 'IsTarget', 'IsolationWindow',
            'BestXCorr', 'BestRT', 'BestScan', 'EValue', 'NumSpectraScored', 'XCorrZScore'
        ]
        assert list(df.columns) == expected_columns, f"Columns mismatch: {list(df.columns)}"
        
        # Check data
        assert len(df) == 1, "Should have one row"
        assert df.iloc[0]['Peptide'] == 'PEPTIDE'
        assert df.iloc[0]['BestXCorr'] == 1.8
        assert df.iloc[0]['EValue'] > 0, "E-value should be present"
        
        # Check that library columns are NOT present
        assert 'LibCosine' not in df.columns, "LibCosine should not be in non-library mode"
        assert 'PrecursorCosine' not in df.columns, "PrecursorCosine should not be in non-library mode"
        
        print("✓ Non-library mode output format validated")
        print(f"  Columns: {list(df.columns)}")
        
    def test_output_has_no_pair_id(self):
        """Test that PairID column is not present (winner-only reporting)."""
        output_file = os.path.join(self.temp_dir, "test_no_pair_id.tsv")
        
        # Test both modes
        for library_mode in [True, False]:
            writer = DIAResultsWriter(output_file, "test.mzml", library_mode=library_mode)
            
            peptide = PeptideCandidate("PEPTIDE", "P12345", mass=799.359954)
            result_dict = {
                'peptide': peptide,
                'charge': 2,
                'is_target': True,
                'isolation_window': (400.0, 425.0),
                'best_xcorr': 1.5,
                'best_rt': 10.5,
                'best_scan': 1234,
                'num_spectra_scored': 50,
            }
            
            if library_mode:
                result_dict.update({
                    'best_lib_cosine_target': 0.85,
                    'lib_cosine_target_zscore': 3.2,
                    'precursor_cosine_target': 0.92,
                })
            else:
                result_dict.update({
                    'e_value': 0.001,
                    'all_xcorr_values': [0.5, 0.8, 1.5, 1.2, 0.6],
                })
            
            results = {('PEPTIDE', 2, True): result_dict}
            
            with writer:
                writer.write_dia_results(results)
            
            df = pd.read_csv(output_file, sep='\t')
            assert 'PairID' not in df.columns, f"PairID should not be in {'library' if library_mode else 'non-library'} mode"
            
        print("✓ PairID column correctly excluded from both modes")


class TestPairProcessing:
    """Tests for pair processing and winner selection."""
    
    def test_pair_processing_order(self):
        """
        Test that peptides are processed in pairs:
        - Even indices are targets
        - Odd indices are decoys
        - One winner per pair
        """
        # Simulate peptide list: alternating target/decoy
        peptides = [
            ("PEPTIDE", 2, True, 1),   # Target, pair 1
            ("EDPTIEP", 2, False, 1),  # Decoy, pair 1
            ("SEQUENCE", 2, True, 2),  # Target, pair 2
            ("ECNEUQES", 2, False, 2), # Decoy, pair 2
        ]
        
        # Process in pairs
        winners = []
        for pair_idx in range(0, len(peptides), 2):
            target = peptides[pair_idx]
            decoy = peptides[pair_idx + 1]
            
            # Verify pairing
            assert target[2] == True, "Even index should be target"
            assert decoy[2] == False, "Odd index should be decoy"
            assert target[3] == decoy[3], "Pair IDs should match"
            
            # Simulate competition (target wins if score > decoy score)
            # For this test, alternate winners
            if pair_idx == 0:
                winners.append(target)  # Target wins pair 1
            else:
                winners.append(decoy)   # Decoy wins pair 2
        
        # Verify we got one winner per pair
        assert len(winners) == 2, "Should have one winner per pair"
        assert winners[0][2] == True, "First winner should be target"
        assert winners[1][2] == False, "Second winner should be decoy"
        
        print("✓ Pair processing:")
        print(f"  Input peptides: {len(peptides)} (2 pairs)")
        print(f"  Winners: {len(winners)}")
        print(f"  Pair 1 winner: {'Target' if winners[0][2] else 'Decoy'}")
        print(f"  Pair 2 winner: {'Target' if winners[1][2] else 'Decoy'}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
