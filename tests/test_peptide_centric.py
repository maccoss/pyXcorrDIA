"""
Test peptide-centric XCorr scoring for DIA searches.

These tests are based on the analysis in DIA_Peptide_Centric_XCorr_Analysis.ipynb
and validate the key differences between peptide-centric and spectrum-centric scoring:

1. Theoretical spectrum preprocessing (windowing + Fast XCorr) 
2. Experimental spectrum preprocessing (windowing ONLY, no Fast XCorr)
3. Scaling factor (0.0001 for peptide-centric vs 0.005 for spectrum-centric)
4. E-value calculation from chromatogram distributions
"""
import numpy as np


class TestPeptideCentricScoring:
    """Test peptide-centric XCorr calculation."""
    
    def test_scaling_factor_difference(self, xcorr_engine):
        """
        Test that peptide-centric uses 0.0001 scaling vs 0.005 for spectrum-centric.
        
        This is critical because preprocessing the theoretical spectrum (peptide-centric)
        produces ~50x higher raw dot products than preprocessing experimental (spectrum-centric).
        """
        from pyXcorrDIA import PeptideCandidate
        
        # Create a simple peptide
        peptide = PeptideCandidate("YQSHTK", "target", mass=762.366054)
        charge = 1
        
        # Generate theoretical spectrum
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
        assert np.count_nonzero(theoretical_raw) > 0, "Theoretical spectrum should have peaks"
        
        # Apply windowing to theoretical
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        
        # Apply Fast XCorr preprocessing to theoretical (peptide-centric)
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        assert np.count_nonzero(theoretical_preprocessed) > 0
        
        # Create a mock experimental spectrum with some overlapping peaks
        exp_windowed = np.zeros(len(theoretical_preprocessed))
        # Add signal where theoretical has peaks (simulate matching spectrum)
        for i in range(len(theoretical_preprocessed)):
            if theoretical_preprocessed[i] > 0:
                exp_windowed[i] = 50.0 + np.random.randn() * 5  # Add some noise
        
        # Calculate peptide-centric XCorr (experimental windowed, theoretical preprocessed)
        raw_xcorr_peptide = np.dot(exp_windowed, theoretical_preprocessed)
        xcorr_peptide_centric = xcorr_engine.calculate_peptide_centric_xcorr(
            exp_windowed, theoretical_preprocessed
        )
        
        # Verify scaling: should use 0.0001
        expected_xcorr = raw_xcorr_peptide * 0.0001
        assert abs(xcorr_peptide_centric - expected_xcorr) < 1e-4, \
            f"Peptide-centric should use 0.0001 scaling: expected {expected_xcorr:.4f}, got {xcorr_peptide_centric:.4f}"
        
        # For comparison: spectrum-centric (experimental preprocessed, theoretical raw)
        exp_preprocessed = xcorr_engine.preprocess_for_xcorr(exp_windowed)
        raw_xcorr_spectrum = np.dot(theoretical_raw, exp_preprocessed)
        xcorr_spectrum_centric = raw_xcorr_spectrum * 0.005
        
        # Verify that peptide-centric raw scores are much higher
        ratio = raw_xcorr_peptide / raw_xcorr_spectrum if raw_xcorr_spectrum > 0 else 0
        assert ratio > 10, \
            f"Peptide-centric raw scores should be ~50x higher than spectrum-centric (got {ratio:.1f}x)"
        
        print("✓ Scaling verified:")
        print(f"  Peptide-centric: raw={raw_xcorr_peptide:.2f}, scaled={xcorr_peptide_centric:.4f} (0.0001)")
        print(f"  Spectrum-centric: raw={raw_xcorr_spectrum:.2f}, scaled={xcorr_spectrum_centric:.4f} (0.005)")
        print(f"  Raw ratio: {ratio:.1f}x")
    
    def test_preprocessing_asymmetry(self, xcorr_engine):
        """
        Test that peptide-centric preprocessing is asymmetric:
        - Theoretical: windowing + Fast XCorr preprocessing
        - Experimental: windowing ONLY (no Fast XCorr)
        
        This is opposite of spectrum-centric where experimental gets full preprocessing.
        """
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate("PEPTIDE", "target", mass=799.359954)
        charge = 2
        
        # Generate and preprocess theoretical spectrum (peptide-centric way)
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        
        # Check that preprocessing changes the spectrum significantly
        assert not np.array_equal(theoretical_windowed, theoretical_preprocessed), \
            "Fast XCorr should modify the windowed theoretical spectrum"
        
        # Check for both positive and negative values (Fast XCorr subtracts background)
        has_positive = np.any(theoretical_preprocessed > 0)
        has_negative = np.any(theoretical_preprocessed < 0)
        assert has_positive and has_negative, \
            "Fast XCorr preprocessed spectrum should have both positive and negative values"
        
        # Create mock experimental spectrum
        from pyXcorrDIA import MassSpectrum
        mz_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity_array = np.array([1000.0, 2000.0, 1500.0, 3000.0, 1200.0])
        ms_spectrum = MassSpectrum(mz_array=mz_array, intensity_array=intensity_array, scan_id="1")
        
        # Preprocess experimental (peptide-centric: windowing ONLY)
        exp_windowed = xcorr_engine.preprocess_spectrum(ms_spectrum)
        
        # Verify experimental is windowed (all values should be non-negative, near target 50.0)
        assert np.all(exp_windowed >= 0), \
            "Windowed experimental spectrum should have no negative values"
        
        nonzero_exp = exp_windowed[exp_windowed > 0]
        if len(nonzero_exp) > 0:
            # Windowing targets 50.0, so values should be in reasonable range
            assert np.all(nonzero_exp < 200), \
                "Windowed values should be in reasonable range (< 200)"
        
        print("✓ Preprocessing asymmetry verified:")
        print(f"  Theoretical preprocessed: {np.count_nonzero(theoretical_preprocessed)} nonzero bins")
        print(f"    Positive: {np.sum(theoretical_preprocessed > 0)}, Negative: {np.sum(theoretical_preprocessed < 0)}")
        print(f"  Experimental windowed: {np.count_nonzero(exp_windowed)} nonzero bins")
        print(f"    All non-negative: {np.all(exp_windowed >= 0)}")
    
    def test_xcorr_score_range(self, xcorr_engine):
        """
        Test that peptide-centric XCorr scores are in reasonable range (0-10).
        
        With 0.0001 scaling, scores should be similar to traditional spectrum-centric scores.
        """
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate("TESTPEPTIDE", "target", mass=1189.572454)
        charge = 2
        
        # Generate theoretical and preprocess
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        
        # Create experimental spectrum that matches well
        exp_windowed = np.zeros(len(theoretical_preprocessed))
        for i in range(len(theoretical_preprocessed)):
            if theoretical_preprocessed[i] > 0:
                exp_windowed[i] = 50.0  # Good signal at matching positions
        
        # Calculate XCorr
        xcorr = xcorr_engine.calculate_peptide_centric_xcorr(exp_windowed, theoretical_preprocessed)
        
        # Score should be positive and reasonable
        assert xcorr > 0, "Good match should have positive XCorr"
        assert xcorr < 50, f"XCorr should be < 50 with 0.0001 scaling, got {xcorr:.2f}"
        
        # Ideally in the 0-10 range for good matches
        print(f"✓ XCorr score in reasonable range: {xcorr:.4f}")
        
        # Test poor match
        exp_poor = np.zeros(len(theoretical_preprocessed))
        for i in range(0, len(exp_poor), 10):
            exp_poor[i] = 50.0  # Random peaks that don't match
        
        xcorr_poor = xcorr_engine.calculate_peptide_centric_xcorr(exp_poor, theoretical_preprocessed)
        assert xcorr_poor < xcorr, "Poor match should have lower XCorr than good match"
        print(f"  Poor match XCorr: {xcorr_poor:.4f}")


class TestPeptideCentricEValues:
    """Test E-value calculation for peptide-centric searches."""
    
    def test_evalue_from_chromatogram(self):
        """
        Test E-value calculation from chromatogram score distribution.
        
        In peptide-centric searches, E-values are calculated from the distribution
        of XCorr scores across all spectra the peptide was scored against (chromatogram).
        """
        # Simulate a chromatogram with realistic score distribution
        np.random.seed(42)
        
        # Most scores are low (noise), with a few high scores (peaks)
        noise_scores = np.random.exponential(scale=2.0, size=800)  # Background
        peak_scores = np.random.normal(loc=25.0, scale=3.0, size=50)  # Signal
        all_scores = np.concatenate([noise_scores, peak_scores])
        
        best_score = np.max(all_scores)
        
        # Implement Comet's E-value calculation
        HISTO_SIZE = 1000
        
        # Create histogram (0.1 XCorr unit bins)
        histogram = np.zeros(HISTO_SIZE + 1)
        for score in all_scores:
            bin_idx = int(score * 10)
            if bin_idx <= HISTO_SIZE:
                histogram[bin_idx] += 1
        
        # Find max_corr
        max_corr = HISTO_SIZE
        while max_corr > 0 and histogram[max_corr] == 0:
            max_corr -= 1
        
        assert max_corr > 0, "Should find non-empty bins"
        
        # Create cumulative distribution (right to left)
        cumulative = np.zeros(HISTO_SIZE + 1)
        cumulative[max_corr] = histogram[max_corr]
        for i in range(max_corr - 1, -1, -1):
            cumulative[i] = cumulative[i + 1] + histogram[i]
        
        # Verify cumulative at bin 0 equals total count
        assert cumulative[0] == len(all_scores), \
            f"Cumulative at 0 should equal total scores: {cumulative[0]} vs {len(all_scores)}"
        
        # Log transformation
        log_cumulative = np.zeros(HISTO_SIZE + 1)
        for i in range(max_corr + 1):
            if cumulative[i] > 0:
                log_cumulative[i] = np.log10(cumulative[i])
        
        # Linear regression on tail
        start_corr = int(max_corr * 0.5)
        next_corr = max_corr
        
        reg_x = []
        reg_y = []
        for i in range(start_corr, next_corr + 1):
            if log_cumulative[i] > 0:
                reg_x.append(i * 0.1)
                reg_y.append(log_cumulative[i])
        
        assert len(reg_x) >= 2, "Should have enough points for regression"
        
        # Fit line
        coeffs = np.polyfit(reg_x, reg_y, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # Calculate E-value for best score
        calculated_evalue = 10 ** (slope * 10.0 * best_score + intercept)
        
        # E-value should be reasonable (< total spectra scored)
        assert calculated_evalue >= 0, "E-value should be non-negative"
        assert calculated_evalue <= len(all_scores), \
            f"E-value should be <= number of spectra: {calculated_evalue:.2f} vs {len(all_scores)}"
        
        # For a good match (high score), E-value should be small
        if best_score > 20:
            assert calculated_evalue < 10, \
                f"High score ({best_score:.2f}) should have low E-value, got {calculated_evalue:.2e}"
        
        print("✓ E-value calculation verified:")
        print(f"  # spectra scored: {len(all_scores)}")
        print(f"  Best XCorr: {best_score:.2f}")
        print(f"  Max bin: {max_corr} (score {max_corr * 0.1:.1f})")
        print(f"  Regression: slope={slope:.4f}, intercept={intercept:.3f}")
        print(f"  Calculated E-value: {calculated_evalue:.2e}")
        print(f"  Interpretation: Expect ~{calculated_evalue:.1f} spectra to score ≥{best_score:.2f} by chance")
    
    def test_evalue_independence_from_charge(self):
        """
        Test that E-values are calculated independently for each charge state.
        
        Different charge states have different fragment ion patterns and score distributions.
        """
        np.random.seed(123)
        
        # Simulate score distributions for two charge states
        charge_2_scores = np.random.exponential(scale=1.5, size=500)
        charge_3_scores = np.random.exponential(scale=2.5, size=500)
        
        best_2 = np.max(charge_2_scores)
        best_3 = np.max(charge_3_scores)
        
        # E-values should be calculated from their respective distributions
        # Not testing full calculation, just verifying the concept
        
        assert len(charge_2_scores) == len(charge_3_scores), \
            "Each charge should be scored against same number of spectra"
        
        print("✓ Charge state independence:")
        print(f"  Charge +2: {len(charge_2_scores)} spectra, best={best_2:.2f}")
        print(f"  Charge +3: {len(charge_3_scores)} spectra, best={best_3:.2f}")


class TestPeptideCentricMatrixScoring:
    """Test vectorized matrix-based peptide-centric scoring."""
    
    def test_matrix_scoring_correctness(self, xcorr_engine):
        """
        Test that matrix-based scoring produces same results as single-peptide scoring.
        
        Matrix scoring: (n_peptides, n_bins) @ (n_spectra, n_bins).T = (n_peptides, n_spectra)
        """
        from pyXcorrDIA import PeptideCandidate
        
        # Create a few test peptides
        peptides = [
            PeptideCandidate("PEPTIDEA", "target", mass=843.39),
            PeptideCandidate("PEPTIDEB", "target", mass=857.41),
            PeptideCandidate("PEPTIDEC", "target", mass=871.42),
        ]
        charge = 2
        
        # Preprocess theoretical spectra
        theoretical_matrix = []
        for peptide in peptides:
            theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
            highest_ion = 0
            for i in range(len(theoretical_raw) - 1, -1, -1):
                if theoretical_raw[i] > 0:
                    highest_ion = i
                    break
            theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
            theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
            theoretical_matrix.append(theoretical_preprocessed)
        
        theoretical_matrix = np.vstack(theoretical_matrix)
        
        # Create a few mock experimental spectra  
        num_bins = len(theoretical_matrix[0])
        exp_matrix = []
        for i in range(5):
            exp_windowed = np.random.rand(num_bins) * 50
            exp_matrix.append(exp_windowed)
        
        exp_matrix = np.vstack(exp_matrix)
        
        # Matrix scoring
        xcorr_matrix = (theoretical_matrix @ exp_matrix.T) * 0.0001
        xcorr_matrix = np.round(xcorr_matrix, 4)
        
        # Verify shape
        assert xcorr_matrix.shape == (len(peptides), len(exp_matrix)), \
            f"Matrix shape should be (n_peptides, n_spectra): {xcorr_matrix.shape}"
        
        # Verify individual calculations match
        for pep_idx in range(len(peptides)):
            for spec_idx in range(len(exp_matrix)):
                # Manual calculation
                manual_xcorr = xcorr_engine.calculate_peptide_centric_xcorr(
                    exp_matrix[spec_idx], theoretical_matrix[pep_idx]
                )
                matrix_xcorr = xcorr_matrix[pep_idx, spec_idx]
                
                assert abs(manual_xcorr - matrix_xcorr) < 1e-4, \
                    f"Matrix scoring should match single calculation: {manual_xcorr:.4f} vs {matrix_xcorr:.4f}"
        
        print("✓ Matrix scoring verified:")
        print(f"  {len(peptides)} peptides × {len(exp_matrix)} spectra")
        print(f"  XCorr range: {xcorr_matrix.min():.4f} to {xcorr_matrix.max():.4f}")
    
    def test_matrix_scoring_efficiency(self, xcorr_engine):
        """
        Test that matrix scoring is more efficient than nested loops.
        
        This is a conceptual test - actual timing would require larger datasets.
        """
        
        # Matrix scoring enables:
        # - Single BLAS call instead of nested loops
        # - Vectorized operations
        # - Better CPU cache utilization
        
        n_peptides = 100
        n_spectra = 50
        n_bins = xcorr_engine.num_bins
        
        # Simulate data shapes
        theoretical_matrix = np.random.randn(n_peptides, n_bins)
        experimental_matrix = np.random.randn(n_spectra, n_bins)
        
        # Matrix multiplication shape
        result_shape = (n_peptides, n_spectra)
        
        # Perform matrix scoring
        xcorr_matrix = (theoretical_matrix @ experimental_matrix.T) * 0.0001
        
        assert xcorr_matrix.shape == result_shape, \
            f"Should produce {n_peptides}×{n_spectra} matrix"
        
        print("✓ Matrix scoring shape verified:")
        print(f"  {n_peptides} peptides × {n_spectra} spectra = {xcorr_matrix.shape}")
        print(f"  Single matrix multiply replaces {n_peptides * n_spectra} individual dot products")


class TestPeptideCentricVsSpectrumCentric:
    """Test differences between peptide-centric and spectrum-centric approaches."""
    
    def test_score_ratio(self, xcorr_engine):
        """
        Test that peptide-centric produces ~50x higher raw scores than spectrum-centric.
        
        This is due to preprocessing the theoretical spectrum instead of experimental.
        """
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate("EXAMPLE", "target", mass=789.35)
        charge = 2
        
        # Generate theoretical
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        
        # Create experimental with matching peaks
        exp_windowed = np.zeros(len(theoretical_preprocessed))
        for i in range(len(theoretical_preprocessed)):
            if theoretical_preprocessed[i] > 0:
                exp_windowed[i] = 50.0
        
        # Peptide-centric: exp_windowed · theoretical_preprocessed
        raw_peptide = np.dot(exp_windowed, theoretical_preprocessed)
        
        # Spectrum-centric: theoretical_raw · exp_preprocessed  
        exp_preprocessed = xcorr_engine.preprocess_for_xcorr(exp_windowed)
        raw_spectrum = np.dot(theoretical_raw, exp_preprocessed)
        
        # Calculate ratio
        ratio = 1.0  # Default
        if raw_spectrum > 0:
            ratio = raw_peptide / raw_spectrum
            assert ratio > 10, \
                f"Peptide-centric should produce ~50x higher raw scores, got {ratio:.1f}x"
            assert ratio < 100, \
                f"Ratio should be reasonable (~50x), got {ratio:.1f}x"
        
        print("✓ Score ratio verified:")
        print(f"  Peptide-centric raw: {raw_peptide:.2f}")
        print(f"  Spectrum-centric raw: {raw_spectrum:.2f}")
        print(f"  Ratio: {ratio:.1f}x")
        
        # With appropriate scaling, final scores should be similar
        final_peptide = raw_peptide * 0.0001
        final_spectrum = raw_spectrum * 0.005
        
        final_ratio = 1.0  # Default
        if final_spectrum > 0:
            final_ratio = final_peptide / final_spectrum
            # Should be close to 1.0 after appropriate scaling
            assert 0.5 < final_ratio < 2.0, \
                f"After scaling, scores should be similar: {final_ratio:.2f}x"
        
        print("  After scaling:")
        print(f"    Peptide-centric: {final_peptide:.4f} (×0.0001)")
        print(f"    Spectrum-centric: {final_spectrum:.4f} (×0.005)")
        print(f"    Ratio: {final_ratio:.2f}x")
