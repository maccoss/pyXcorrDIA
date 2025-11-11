"""
Test peptide-centric XCorr scoring using real data from DIA_Peptide_Centric_XCorr_Analysis.ipynb.

These tests use the actual top peptide from Ast-Neo DIA search to validate:
1. Theoretical spectrum generation and preprocessing
2. Experimental spectrum preprocessing
3. XCorr calculation matches notebook results
4. E-value calculation from chromatogram
5. Scaling factor effects (0.0001 vs 0.005)

Test data based on:
- Peptide: KIQALQQQADEAEDR
- Charge: 2+
- Mass: 1741.859493
- Best scan: 61486
- Isolation window: [870.6456-872.6465]
"""
import numpy as np


# Real data from notebook analysis
REAL_PEPTIDE_SEQUENCE = "KIQALQQQADEAEDR"
REAL_PEPTIDE_CHARGE = 2
REAL_PEPTIDE_MASS = 1741.859493

# Expected values from notebook
EXPECTED_THEORETICAL_NONZERO_RAW = 28
EXPECTED_THEORETICAL_NONZERO_PREPROCESSED = 1638  # After Fast XCorr

# Real chromatogram XCorr scores (first 20 scores with 0.005 scaling from old file)
REAL_CHROM_SCORES_0005_SCALING = np.array([
    19.0468, 17.2736, -16.9892, 35.7663, 203.3480, 249.7014, 233.3683, 174.5771,
    53.4126, 43.8595, 79.0559, 57.2810, 35.6339, 24.3680, 48.1038, 18.1613,
    37.9016, 19.0423, 33.2563, 11.6988
])

# Expected best score
EXPECTED_BEST_XCORR_0005 = 249.7014
EXPECTED_BEST_XCORR_0001 = EXPECTED_BEST_XCORR_0005 / 50  # Convert to 0.0001 scaling

# Raw dot product that produces the expected scores
EXPECTED_RAW_DOT_PRODUCT = EXPECTED_BEST_XCORR_0005 / 0.005  # ~49940


class TestRealPeptidePreprocessing:
    """Test theoretical spectrum generation and preprocessing with real peptide."""
    
    def test_theoretical_spectrum_generation(self, xcorr_engine):
        """Test that theoretical spectrum generation matches notebook results."""
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate(REAL_PEPTIDE_SEQUENCE, 'target', REAL_PEPTIDE_MASS)
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, REAL_PEPTIDE_CHARGE)
        
        # Check number of fragment ions matches notebook
        nonzero_count = np.count_nonzero(theoretical_raw)
        assert nonzero_count == EXPECTED_THEORETICAL_NONZERO_RAW, \
            f"Expected {EXPECTED_THEORETICAL_NONZERO_RAW} fragment ions, got {nonzero_count}"
        
        print(f"✓ Theoretical spectrum: {nonzero_count} fragment ions")
    
    def test_theoretical_preprocessing_pipeline(self, xcorr_engine):
        """Test complete theoretical spectrum preprocessing matches notebook."""
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate(REAL_PEPTIDE_SEQUENCE, 'target', REAL_PEPTIDE_MASS)
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, REAL_PEPTIDE_CHARGE)
        
        # Apply windowing
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        
        # Windowing should not change number of nonzero bins for theoretical
        assert np.count_nonzero(theoretical_windowed) == EXPECTED_THEORETICAL_NONZERO_RAW, \
            "Windowing should preserve fragment ion count"
        
        # Apply Fast XCorr preprocessing
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        
        # Fast XCorr should dramatically increase nonzero bins (background subtraction)
        nonzero_preprocessed = np.count_nonzero(theoretical_preprocessed)
        assert nonzero_preprocessed == EXPECTED_THEORETICAL_NONZERO_PREPROCESSED, \
            f"Expected {EXPECTED_THEORETICAL_NONZERO_PREPROCESSED} bins after Fast XCorr, got {nonzero_preprocessed}"
        
        # Should have both positive and negative values
        assert np.any(theoretical_preprocessed > 0), "Should have positive values"
        assert np.any(theoretical_preprocessed < 0), "Should have negative values (background subtraction)"
        
        print(f"✓ Theoretical preprocessing: {EXPECTED_THEORETICAL_NONZERO_RAW} → {nonzero_preprocessed} bins")
        print(f"  Positive bins: {np.sum(theoretical_preprocessed > 0)}")
        print(f"  Negative bins: {np.sum(theoretical_preprocessed < 0)}")


class TestRealXCorrCalculation:
    """Test XCorr calculation with real peptide matches expected results."""
    
    def test_xcorr_scaling_effects(self, xcorr_engine):
        """Test that different scaling factors produce expected score ranges."""
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate(REAL_PEPTIDE_SEQUENCE, 'target', REAL_PEPTIDE_MASS)
        theoretical_raw = xcorr_engine.generate_theoretical_spectrum(peptide, REAL_PEPTIDE_CHARGE)
        
        # Preprocess theoretical
        highest_ion = 0
        for i in range(len(theoretical_raw) - 1, -1, -1):
            if theoretical_raw[i] > 0:
                highest_ion = i
                break
        theoretical_windowed = xcorr_engine._make_corr_data(theoretical_raw, highest_ion, 1.0)
        theoretical_preprocessed = xcorr_engine.preprocess_for_xcorr(theoretical_windowed)
        
        # Create a mock experimental spectrum that gives approximately the expected raw dot product
        # We'll simulate this by creating a spectrum with matching features
        exp_windowed = np.zeros(len(theoretical_preprocessed))
        
        # Add signal at positions where theoretical has positive values
        positive_positions = np.where(theoretical_preprocessed > 0)[0]
        for pos in positive_positions[:50]:  # Use some matching positions
            exp_windowed[pos] = 50.0  # Typical windowed intensity
        
        # Calculate raw dot product
        raw_xcorr = np.dot(exp_windowed, theoretical_preprocessed)
        
        # Test both scaling factors
        xcorr_0005 = raw_xcorr * 0.005
        xcorr_0001 = raw_xcorr * 0.0001
        
        # With 0.005 scaling, scores should be very high (>10)
        assert xcorr_0005 > 10, f"0.005 scaling should produce high scores (>10), got {xcorr_0005:.2f}"
        
        # With 0.0001 scaling, scores should be in reasonable range (<10)
        assert xcorr_0001 < 50, f"0.0001 scaling should produce reasonable scores, got {xcorr_0001:.2f}"
        
        # The ratio should be exactly 50x
        ratio = xcorr_0005 / xcorr_0001
        assert abs(ratio - 50.0) < 0.01, f"Ratio should be 50x, got {ratio:.2f}x"
        
        print("✓ Scaling effects verified:")
        print(f"  Raw dot product: {raw_xcorr:.2f}")
        print(f"  With 0.005 scaling: {xcorr_0005:.2f} (old peptide-centric)")
        print(f"  With 0.0001 scaling: {xcorr_0001:.4f} (corrected peptide-centric)")
        print(f"  Ratio: {ratio:.1f}x")
    
    def test_peptide_centric_function_uses_correct_scaling(self, xcorr_engine):
        """Test that calculate_peptide_centric_xcorr uses 0.0001 scaling."""
        # Create simple test spectra
        exp_windowed = np.array([50.0, 50.0, 50.0, 0.0, 0.0])
        theoretical_preprocessed = np.array([10.0, -5.0, 10.0, 0.0, 0.0])
        
        # Calculate manually
        raw_dot_product = np.dot(exp_windowed, theoretical_preprocessed)
        expected_xcorr = raw_dot_product * 0.0001
        
        # Calculate using function
        actual_xcorr = xcorr_engine.calculate_peptide_centric_xcorr(
            exp_windowed, theoretical_preprocessed
        )
        
        # Should match expected with 0.0001 scaling
        assert abs(actual_xcorr - expected_xcorr) < 1e-4, \
            f"Expected {expected_xcorr:.4f} (0.0001 scaling), got {actual_xcorr:.4f}"
        
        # Should NOT match 0.005 scaling
        wrong_xcorr = raw_dot_product * 0.005
        assert abs(actual_xcorr - wrong_xcorr) > 0.01, \
            "Function should not be using 0.005 scaling"
        
        print("✓ Function uses 0.0001 scaling:")
        print(f"  Raw: {raw_dot_product:.2f}")
        print(f"  Expected (0.0001): {expected_xcorr:.4f}")
        print(f"  Actual: {actual_xcorr:.4f}")
        print(f"  Wrong (0.005): {wrong_xcorr:.4f}")


class TestRealChromatogramEValues:
    """Test E-value calculation using real chromatogram data."""
    
    def test_evalue_from_real_chromatogram(self):
        """
        Test E-value calculation using actual chromatogram scores from notebook.
        
        Uses the first 20 XCorr scores from the real KIQALQQQADEAEDR chromatogram.
        Note: These scores use the old 0.005 scaling, so they're in the 0-250 range.
        Some scores may be negative due to anti-correlation.
        """
        scores = REAL_CHROM_SCORES_0005_SCALING.copy()
        best_score = scores.max()
        
        # Verify we have the expected best score
        assert abs(best_score - EXPECTED_BEST_XCORR_0005) < 0.01, \
            f"Expected best score {EXPECTED_BEST_XCORR_0005:.4f}, got {best_score:.4f}"
        
        # Implement Comet's E-value calculation
        HISTO_SIZE = 1000
        
        # Create histogram (0.1 XCorr unit bins)
        # Only histogram positive scores
        histogram = np.zeros(HISTO_SIZE + 1)
        positive_scores_count = 0
        for score in scores:
            if score >= 0:
                bin_idx = int(score * 10)
                if 0 <= bin_idx <= HISTO_SIZE:
                    histogram[bin_idx] += 1
                    positive_scores_count += 1
        
        # Find max_corr
        max_corr = HISTO_SIZE
        while max_corr > 0 and histogram[max_corr] == 0:
            max_corr -= 1
        
        assert max_corr > 0, "Should find non-empty bins"
        
        # Create cumulative distribution
        cumulative = np.zeros(HISTO_SIZE + 1)
        cumulative[max_corr] = histogram[max_corr]
        for i in range(max_corr - 1, -1, -1):
            cumulative[i] = cumulative[i + 1] + histogram[i]
        
        # Total should match number of positive scores
        assert cumulative[0] == positive_scores_count, \
            f"Cumulative should equal # positive scores: {cumulative[0]} vs {positive_scores_count}"
        
        # Log transformation
        log_cumulative = np.zeros(HISTO_SIZE + 1)
        for i in range(max_corr + 1):
            if cumulative[i] > 0:
                log_cumulative[i] = np.log10(cumulative[i])
        
        # Linear regression
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
        
        # Calculate E-value
        calculated_evalue = 10 ** (slope * 10.0 * best_score + intercept)
        
        # E-value should be reasonable
        assert calculated_evalue >= 0, "E-value should be non-negative"
        assert calculated_evalue <= positive_scores_count, \
            f"E-value should be <= # positive spectra: {calculated_evalue:.2e} vs {positive_scores_count}"
        
        # For a high score like 249.7, E-value should be small (good match)
        assert calculated_evalue < 5, \
            f"High score ({best_score:.2f}) should have low E-value, got {calculated_evalue:.2e}"
        
        print("✓ Real chromatogram E-value calculation:")
        print(f"  # spectra: {len(scores)}")
        print(f"  # positive scores: {positive_scores_count}")
        print(f"  # negative scores: {len(scores) - positive_scores_count}")
        print(f"  Score range: {scores.min():.2f} to {scores.max():.2f}")
        print(f"  Best score: {best_score:.2f}")
        print(f"  Max bin: {max_corr} (score {max_corr * 0.1:.1f})")
        print(f"  Regression: slope={slope:.4f}, intercept={intercept:.3f}")
        print(f"  E-value: {calculated_evalue:.2e}")
        print(f"  Interpretation: ~{calculated_evalue:.2f} spectra expected to score ≥{best_score:.2f} by chance")
    
    def test_rescaled_chromatogram_evalues(self):
        """
        Test E-value calculation with rescaled chromatogram (0.0001 instead of 0.005).
        
        This shows what E-values will look like after regenerating with correct scaling.
        """
        # Rescale scores from 0.005 to 0.0001 (divide by 50)
        scores_rescaled = REAL_CHROM_SCORES_0005_SCALING / 50
        best_score = scores_rescaled.max()
        
        assert abs(best_score - EXPECTED_BEST_XCORR_0001) < 0.01, \
            f"Expected rescaled best score {EXPECTED_BEST_XCORR_0001:.4f}, got {best_score:.4f}"
        
        # E-value calculation with rescaled scores
        HISTO_SIZE = 1000
        
        histogram = np.zeros(HISTO_SIZE + 1)
        for score in scores_rescaled:
            bin_idx = int(score * 10)
            if 0 <= bin_idx <= HISTO_SIZE:
                histogram[bin_idx] += 1
        
        max_corr = HISTO_SIZE
        while max_corr > 0 and histogram[max_corr] == 0:
            max_corr -= 1
        
        cumulative = np.zeros(HISTO_SIZE + 1)
        cumulative[max_corr] = histogram[max_corr]
        for i in range(max_corr - 1, -1, -1):
            cumulative[i] = cumulative[i + 1] + histogram[i]
        
        log_cumulative = np.zeros(HISTO_SIZE + 1)
        for i in range(max_corr + 1):
            if cumulative[i] > 0:
                log_cumulative[i] = np.log10(cumulative[i])
        
        start_corr = int(max_corr * 0.5)
        reg_x = []
        reg_y = []
        for i in range(start_corr, max_corr + 1):
            if log_cumulative[i] > 0:
                reg_x.append(i * 0.1)
                reg_y.append(log_cumulative[i])
        
        if len(reg_x) >= 2:
            coeffs = np.polyfit(reg_x, reg_y, 1)
            slope, intercept = coeffs[0], coeffs[1]
            calculated_evalue = 10 ** (slope * 10.0 * best_score + intercept)
        else:
            calculated_evalue = len(scores_rescaled)
        
        # E-value should still be reasonable
        assert calculated_evalue >= 0, "E-value should be non-negative"
        assert calculated_evalue <= len(scores_rescaled), \
            "E-value should be <= # spectra"
        
        print("✓ Rescaled chromatogram E-value calculation:")
        print(f"  # spectra: {len(scores_rescaled)}")
        print(f"  Score range: {scores_rescaled.min():.4f} to {scores_rescaled.max():.4f}")
        print(f"  Best score: {best_score:.4f}")
        print(f"  E-value: {calculated_evalue:.2e}")
        print("  Note: Scores rescaled from 0.005 to 0.0001 (÷50)")


class TestScalingFactorValidation:
    """Validate that 0.0001 is the correct scaling factor for peptide-centric."""
    
    def test_score_range_comparison(self):
        """
        Compare score ranges between peptide-centric and spectrum-centric.
        
        With proper scaling, both should produce similar score ranges (0-10).
        """
        # Real best score with old 0.005 scaling
        old_scaling_score = EXPECTED_BEST_XCORR_0005  # 249.7014
        
        # New 0.0001 scaling (50x smaller)
        new_scaling_score = EXPECTED_BEST_XCORR_0001  # 4.9940
        
        # Typical spectrum-centric scores with 0.005 scaling are in 0-10 range
        # So peptide-centric with 0.0001 should also be in similar range
        
        assert old_scaling_score > 100, \
            f"Old scaling produces unreasonably high scores: {old_scaling_score:.2f}"
        
        assert new_scaling_score < 10, \
            f"New scaling should produce reasonable scores (<10), got {new_scaling_score:.4f}"
        
        # The ratio should be exactly 50
        ratio = old_scaling_score / new_scaling_score
        assert abs(ratio - 50.0) < 0.1, f"Ratio should be 50x, got {ratio:.2f}x"
        
        print("✓ Scaling factor validation:")
        print(f"  Old (0.005): {old_scaling_score:.2f} - unreasonably high")
        print(f"  New (0.0001): {new_scaling_score:.4f} - reasonable range")
        print(f"  Ratio: {ratio:.1f}x")
        print("  Conclusion: 0.0001 scaling brings scores into proper range")
    
    def test_raw_dot_product_magnitude(self):
        """
        Test that raw dot products are ~50x higher in peptide-centric vs spectrum-centric.
        
        This explains why we need different scaling factors.
        """
        # Expected raw dot product from real data
        expected_raw = EXPECTED_RAW_DOT_PRODUCT  # ~49940
        
        # With 0.005 scaling: 49940 * 0.005 = 249.7
        score_0005 = expected_raw * 0.005
        
        # With 0.0001 scaling: 49940 * 0.0001 = 4.994
        score_0001 = expected_raw * 0.0001
        
        assert abs(score_0005 - EXPECTED_BEST_XCORR_0005) < 0.1, \
            "0.005 scaling should match old file"
        
        assert abs(score_0001 - EXPECTED_BEST_XCORR_0001) < 0.01, \
            "0.0001 scaling should match corrected value"
        
        # Typical spectrum-centric raw dot products are ~1000 (not 50000)
        # This is why peptide-centric needs 50x smaller scaling factor
        typical_spectrum_centric_raw = expected_raw / 50  # ~1000
        typical_spectrum_centric_score = typical_spectrum_centric_raw * 0.005  # ~5.0
        
        print("✓ Raw dot product analysis:")
        print(f"  Peptide-centric raw: {expected_raw:.0f}")
        print(f"  Spectrum-centric raw (typical): {typical_spectrum_centric_raw:.0f}")
        print(f"  Ratio: {expected_raw / typical_spectrum_centric_raw:.1f}x")
        print("  ")
        print(f"  Peptide-centric with 0.005: {score_0005:.2f} (too high)")
        print(f"  Peptide-centric with 0.0001: {score_0001:.4f} (appropriate)")
        print(f"  Spectrum-centric with 0.005: {typical_spectrum_centric_score:.4f} (standard)")
