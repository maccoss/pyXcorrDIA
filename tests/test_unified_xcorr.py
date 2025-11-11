"""
Tests for the unified XCorr calculation function.

This test module validates that:
1. The unified calculate_xcorr() function works for both single and batch scoring
2. Single spectrum scoring matches the original implementation
3. Matrix scoring is consistent with single scoring
4. Both spectrum-centric (0.005) and peptide-centric (0.0001) scaling work correctly
5. The convenience wrappers (calculate_fast_xcorr, calculate_peptide_centric_xcorr) work correctly
"""

import numpy as np
from pyXcorrDIA import FastXCorr, PeptideCandidate


class TestUnifiedXCorrFunction:
    """Test the unified calculate_xcorr() function."""
    
    def test_single_spectrum_scoring(self):
        """Test that single spectrum scoring works correctly."""
        xcorr_engine = FastXCorr()
        
        # Create simple test spectra
        spectrum_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectrum_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Calculate XCorr with spectrum-centric scaling
        xcorr = xcorr_engine.calculate_xcorr(spectrum_a, spectrum_b, scaling_factor=0.005)
        
        # Verify it's a float
        assert isinstance(xcorr, float)
        
        # Verify the calculation: dot product * scaling
        expected = np.dot(spectrum_a, spectrum_b) * 0.005
        assert abs(xcorr - round(expected, 4)) < 1e-6
    
    def test_matrix_scoring_2d_by_2d(self):
        """Test matrix scoring with two 2D arrays."""
        xcorr_engine = FastXCorr()
        
        # Create multiple spectra (3 peptides vs 4 spectra)
        peptides = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0]
        ])
        
        spectra = np.array([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0]
        ])
        
        # Calculate XCorr matrix
        xcorr_matrix = xcorr_engine.calculate_xcorr(peptides, spectra, scaling_factor=0.0001)
        
        # Verify it's a 2D array
        assert isinstance(xcorr_matrix, np.ndarray)
        assert xcorr_matrix.ndim == 2
        assert xcorr_matrix.shape == (3, 4)  # 3 peptides × 4 spectra
        
        # Verify a specific calculation
        # peptides[0] dot spectra[0] = [1,2,3,4,5] · [5,4,3,2,1] = 5+8+9+8+5 = 35
        expected = 35.0 * 0.0001
        assert abs(xcorr_matrix[0, 0] - expected) < 1e-6
    
    def test_matrix_vs_single_consistency(self):
        """Test that matrix scoring gives same results as repeated single scoring."""
        xcorr_engine = FastXCorr()
        
        # Create test data
        peptides = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0]
        ])
        
        spectra = np.array([
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ])
        
        scaling = 0.005
        
        # Matrix scoring
        xcorr_matrix = xcorr_engine.calculate_xcorr(peptides, spectra, scaling_factor=scaling)
        
        # Result should be ndarray for matrix operations
        assert isinstance(xcorr_matrix, np.ndarray)
        
        # Single scoring for each pair
        for i in range(len(peptides)):
            for j in range(len(spectra)):
                single_xcorr = xcorr_engine.calculate_xcorr(peptides[i], spectra[j], 
                                                            scaling_factor=scaling)
                matrix_xcorr = float(xcorr_matrix[i, j])
                
                # Should match within floating point precision
                assert abs(single_xcorr - matrix_xcorr) < 1e-6, \
                    f"Mismatch at [{i},{j}]: single={single_xcorr}, matrix={matrix_xcorr}"
    
    def test_spectrum_centric_scaling(self):
        """Test that spectrum-centric scaling (0.005) works correctly."""
        xcorr_engine = FastXCorr()
        
        spectrum_a = np.array([10.0, 20.0, 30.0])
        spectrum_b = np.array([1.0, 2.0, 3.0])
        
        xcorr = xcorr_engine.calculate_xcorr(spectrum_a, spectrum_b, scaling_factor=0.005)
        
        # Expected: (10*1 + 20*2 + 30*3) * 0.005 = (10+40+90) * 0.005 = 140 * 0.005 = 0.7
        assert abs(xcorr - 0.7) < 1e-6
    
    def test_peptide_centric_scaling(self):
        """Test that peptide-centric scaling (0.0001) works correctly."""
        xcorr_engine = FastXCorr()
        
        spectrum_a = np.array([10.0, 20.0, 30.0])
        spectrum_b = np.array([1.0, 2.0, 3.0])
        
        xcorr = xcorr_engine.calculate_xcorr(spectrum_a, spectrum_b, scaling_factor=0.0001)
        
        # Expected: (10*1 + 20*2 + 30*3) * 0.0001 = 140 * 0.0001 = 0.014
        assert abs(xcorr - 0.014) < 1e-6
    
    def test_scaling_factor_50x_difference(self):
        """Test that peptide-centric scaling is 50x smaller than spectrum-centric."""
        xcorr_engine = FastXCorr()
        
        spectrum_a = np.array([100.0, 200.0, 300.0])
        spectrum_b = np.array([1.0, 2.0, 3.0])
        
        xcorr_spectrum = xcorr_engine.calculate_xcorr(spectrum_a, spectrum_b, scaling_factor=0.005)
        xcorr_peptide = xcorr_engine.calculate_xcorr(spectrum_a, spectrum_b, scaling_factor=0.0001)
        
        # Should be 50x difference (0.005 / 0.0001 = 50)
        ratio = xcorr_spectrum / xcorr_peptide
        assert abs(ratio - 50.0) < 1e-6


class TestConvenienceWrappers:
    """Test the convenience wrapper functions."""
    
    def test_calculate_fast_xcorr_wrapper(self):
        """Test that calculate_fast_xcorr() wrapper uses correct scaling."""
        xcorr_engine = FastXCorr()
        
        theoretical = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        experimental = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Call wrapper
        xcorr_wrapper = xcorr_engine.calculate_fast_xcorr(theoretical, experimental)
        
        # Call unified function directly with same scaling
        xcorr_direct = xcorr_engine.calculate_xcorr(theoretical, experimental, 
                                                     scaling_factor=0.005)
        
        # Should be identical
        assert abs(xcorr_wrapper - xcorr_direct) < 1e-9
        assert isinstance(xcorr_wrapper, float)
    
    def test_calculate_peptide_centric_xcorr_wrapper(self):
        """Test that calculate_peptide_centric_xcorr() wrapper uses correct scaling."""
        xcorr_engine = FastXCorr()
        
        experimental = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        theoretical = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Call wrapper
        xcorr_wrapper = xcorr_engine.calculate_peptide_centric_xcorr(experimental, theoretical)
        
        # Call unified function directly with same scaling
        xcorr_direct = xcorr_engine.calculate_xcorr(experimental, theoretical, 
                                                     scaling_factor=0.0001)
        
        # Should be identical
        assert abs(xcorr_wrapper - xcorr_direct) < 1e-9
        assert isinstance(xcorr_wrapper, float)
    
    def test_wrappers_return_float(self):
        """Test that wrapper functions always return float, not ndarray."""
        xcorr_engine = FastXCorr()
        
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([3.0, 2.0, 1.0])
        
        # Test spectrum-centric wrapper
        result1 = xcorr_engine.calculate_fast_xcorr(vec_a, vec_b)
        assert isinstance(result1, float)
        assert not isinstance(result1, np.ndarray)
        
        # Test peptide-centric wrapper
        result2 = xcorr_engine.calculate_peptide_centric_xcorr(vec_a, vec_b)
        assert isinstance(result2, float)
        assert not isinstance(result2, np.ndarray)


class TestMatrixScoringEdgeCases:
    """Test edge cases for matrix scoring."""
    
    def test_single_peptide_multiple_spectra(self):
        """Test scoring one peptide against multiple spectra."""
        xcorr_engine = FastXCorr()
        
        # One peptide (will be reshaped to 1×n)
        peptide = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Multiple spectra
        spectra = np.array([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0]
        ])
        
        # Should produce 1×3 matrix (or 3-element vector)
        xcorr_result = xcorr_engine.calculate_xcorr(peptide, spectra, scaling_factor=0.005)
        
        # Verify shape
        if isinstance(xcorr_result, np.ndarray):
            assert xcorr_result.size == 3
    
    def test_multiple_peptides_single_spectrum(self):
        """Test scoring multiple peptides against one spectrum."""
        xcorr_engine = FastXCorr()
        
        # Multiple peptides
        peptides = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0]
        ])
        
        # One spectrum (will be reshaped to 1×n)
        spectrum = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Should produce 2×1 matrix (or 2-element vector)
        xcorr_result = xcorr_engine.calculate_xcorr(peptides, spectrum, scaling_factor=0.005)
        
        # Verify shape
        if isinstance(xcorr_result, np.ndarray):
            assert xcorr_result.size == 2
    
    def test_1x1_matrix_returns_float(self):
        """Test that 1×1 matrix result is converted to float."""
        xcorr_engine = FastXCorr()
        
        # Single vectors (will create 1×1 matrix internally)
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([3.0, 2.0, 1.0])
        
        result = xcorr_engine.calculate_xcorr(vec_a, vec_b, scaling_factor=0.005)
        
        # Should be float, not array
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)
    
    def test_empty_spectra_handling(self):
        """Test handling of zero-length spectra."""
        xcorr_engine = FastXCorr()
        
        # Empty arrays should produce zero score
        empty_a = np.array([])
        empty_b = np.array([])
        
        result = xcorr_engine.calculate_xcorr(empty_a, empty_b, scaling_factor=0.005)
        assert result == 0.0
    
    def test_mismatched_lengths_truncation(self):
        """Test that mismatched vector lengths are handled (truncation to min_len)."""
        xcorr_engine = FastXCorr()
        
        # Different length vectors
        vec_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vec_b = np.array([5.0, 4.0, 3.0])  # Shorter
        
        # Should truncate to length 3 and calculate
        result = xcorr_engine.calculate_xcorr(vec_a, vec_b, scaling_factor=0.005)
        
        # Expected: (1*5 + 2*4 + 3*3) * 0.005 = (5+8+9) * 0.005 = 22 * 0.005 = 0.11
        assert abs(result - 0.11) < 1e-6


class TestRealDataMatrixScoring:
    """Test matrix scoring with realistic preprocessed spectra."""
    
    def test_matrix_scoring_with_real_preprocessing(self):
        """Test that matrix scoring works with actual preprocessed spectra."""
        xcorr_engine = FastXCorr()
        
        # Create a realistic test case: 3 peptides vs 2 spectra
        peptide_sequences = ["PEPTIDE", "SEQUENCE", "PROTEIN"]
        
        # Generate theoretical spectra and preprocess them
        theoretical_spectra = []
        for seq in peptide_sequences:
            peptide = PeptideCandidate(seq, "test_protein", 
                                      xcorr_engine.calculate_peptide_mass(seq))
            theoretical = xcorr_engine.generate_theoretical_spectrum(peptide, charge=2)
            
            # For peptide-centric: preprocess theoretical spectrum
            theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical)
            theoretical_spectra.append(theoretical_preprocessed)
        
        # Stack into matrix
        theoretical_matrix = np.vstack(theoretical_spectra)
        
        # Create two experimental spectra (just dummy data)
        experimental_matrix = np.random.rand(2, xcorr_engine.num_bins) * 10.0
        
        # Score with peptide-centric scaling
        xcorr_matrix = xcorr_engine.calculate_xcorr(theoretical_matrix, experimental_matrix,
                                                     scaling_factor=0.0001)
        
        # Verify shape and type
        assert isinstance(xcorr_matrix, np.ndarray)
        assert xcorr_matrix.shape == (3, 2)  # 3 peptides × 2 spectra
        
        # Verify all scores are reasonable (not NaN, not infinite)
        assert np.all(np.isfinite(xcorr_matrix))
        
        # Verify consistency: each matrix element should match single calculation
        for i in range(3):
            for j in range(2):
                single_score = xcorr_engine.calculate_xcorr(
                    theoretical_matrix[i], experimental_matrix[j], scaling_factor=0.0001
                )
                assert abs(single_score - xcorr_matrix[i, j]) < 1e-6


class TestCodeUnification:
    """Test that spectrum-centric and peptide-centric use the same core code."""
    
    def test_same_dot_product_different_scaling(self):
        """Verify both modes use identical dot product, only scaling differs."""
        xcorr_engine = FastXCorr()
        
        # Create test spectra
        spec_a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        spec_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Calculate raw dot product
        raw_dot = np.dot(spec_a, spec_b)
        
        # Calculate with spectrum-centric scaling
        xcorr_spectrum = xcorr_engine.calculate_xcorr(spec_a, spec_b, scaling_factor=0.005)
        
        # Calculate with peptide-centric scaling
        xcorr_peptide = xcorr_engine.calculate_xcorr(spec_a, spec_b, scaling_factor=0.0001)
        
        # Verify they use the same raw dot product
        assert abs(xcorr_spectrum / 0.005 - raw_dot) < 1e-3  # Account for rounding
        assert abs(xcorr_peptide / 0.0001 - raw_dot) < 1e-3
        
        # Verify the ratio is exactly the scaling ratio
        expected_ratio = 0.005 / 0.0001  # = 50
        actual_ratio = xcorr_spectrum / xcorr_peptide
        assert abs(actual_ratio - expected_ratio) < 1e-6
    
    def test_unified_function_handles_both_modes(self):
        """Test that one function handles both spectrum-centric and peptide-centric."""
        xcorr_engine = FastXCorr()
        
        # Spectrum-centric: theoretical (raw) · experimental (preprocessed)
        theoretical_raw = np.ones(100)
        experimental_preprocessed = np.random.rand(100)
        
        xcorr_sc = xcorr_engine.calculate_xcorr(theoretical_raw, experimental_preprocessed,
                                                scaling_factor=0.005)
        
        # Peptide-centric: experimental (windowed) · theoretical (preprocessed)
        experimental_windowed = np.random.rand(100)
        theoretical_preprocessed = np.ones(100)
        
        xcorr_pc = xcorr_engine.calculate_xcorr(experimental_windowed, theoretical_preprocessed,
                                                scaling_factor=0.0001)
        
        # Both should produce valid scores
        assert isinstance(xcorr_sc, float)
        assert isinstance(xcorr_pc, float)
        assert xcorr_sc >= 0
        assert xcorr_pc >= 0
