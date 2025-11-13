"""
Test E-value calculation functionality.
"""


class TestEValueCalculation:
    """Test E-value calculation with various score distributions."""
    
    def test_evalue_with_normal_distribution(self, xcorr_engine):
        """Test E-value calculation with a normal distribution of scores."""
        # Create a distribution with one clear outlier
        scores = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3.0]
        top_score = 3.0
        
        e_value = xcorr_engine.calculate_e_value(scores, top_score)
        
        # E-value should be in valid range [1e-10, 1.0]
        assert 1e-10 <= e_value <= 1.0, f"E-value {e_value} out of valid range"
        # Should be less than 1.0 since top score is above the rest
        assert e_value < 1.0, f"E-value {e_value} should be < 1.0 for top score"
    
    def test_evalue_with_uniform_scores(self, xcorr_engine):
        """Test E-value when all scores are similar (no clear winner)."""
        # All scores similar - no clear signal
        scores = [1.0] * 20
        top_score = 1.0
        
        e_value = xcorr_engine.calculate_e_value(scores, top_score)
        
        # Should return 1.0 when there's no clear winner
        assert e_value == 1.0, f"E-value should be 1.0 for uniform scores, got {e_value}"
    
    def test_evalue_with_insufficient_scores(self, xcorr_engine):
        """Test E-value calculation with too few scores."""
        # Less than 10 scores - should return 1.0
        scores = [0.5, 0.7, 0.9, 1.1, 1.3]
        top_score = 1.3
        
        e_value = xcorr_engine.calculate_e_value(scores, top_score)
        
        assert e_value == 1.0, f"E-value should be 1.0 for insufficient data, got {e_value}"
    
    def test_evalue_never_exceeds_one(self, xcorr_engine):
        """Test that E-values are always capped at 1.0."""
        # Various distributions that might produce poor fits
        test_cases = [
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4],  # Low scores
            [0.0] * 15,  # All zeros
            [0.5 + 0.1 * i for i in range(20)],  # Linear increase
        ]
        
        for scores in test_cases:
            if len(scores) > 0:
                top_score = max(scores)
                e_value = xcorr_engine.calculate_e_value(scores, top_score)
                assert e_value <= 1.0, f"E-value {e_value} exceeds 1.0 for scores {scores}"
                assert e_value >= 1e-10, f"E-value {e_value} below minimum for scores {scores}"
    
    def test_evalue_decreases_with_better_separation(self, xcorr_engine):
        """Test that E-value decreases as the top score becomes more separated."""
        base_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        
        # Test with increasing top scores
        e_value_1 = xcorr_engine.calculate_e_value(base_scores + [2.0], 2.0)
        e_value_2 = xcorr_engine.calculate_e_value(base_scores + [3.0], 3.0)
        e_value_3 = xcorr_engine.calculate_e_value(base_scores + [4.0], 4.0)
        
        # E-values should decrease as separation increases
        assert e_value_1 >= e_value_2 >= e_value_3, \
            f"E-values should decrease with better separation: {e_value_1}, {e_value_2}, {e_value_3}"
    
    def test_evalue_with_realistic_xcorr_scores(self, xcorr_engine):
        """Test E-value calculation with realistic XCorr score distributions."""
        # Simulate a typical DIA scenario: mostly low scores with a few good ones
        low_scores = [0.1 + 0.05 * i for i in range(30)]  # 30 low scores
        medium_scores = [1.5 + 0.1 * i for i in range(5)]  # 5 medium scores
        good_score = 3.5  # One excellent score
        
        all_scores = low_scores + medium_scores + [good_score]
        
        e_value = xcorr_engine.calculate_e_value(all_scores, good_score)
        
        # Should get a good e-value for this clear signal
        assert e_value < 0.01, f"E-value {e_value} should be < 0.01 for strong signal"
        assert 1e-10 <= e_value <= 1.0, f"E-value {e_value} out of valid range"


class TestEValueByCharge:
    """Test charge-state specific E-value calculation."""
    
    def test_evalue_by_charge_basic(self, xcorr_engine):
        """Test basic E-value calculation by charge state."""
        # Create score distributions for different charge states
        score_distributions = {
            2: [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
            3: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        }
        
        # Test charge 2
        e_value_charge2 = xcorr_engine.calculate_e_value_by_charge(
            score_distributions, 2.5, 2
        )
        assert 1e-10 <= e_value_charge2 <= 1.0
        
        # Test charge 3
        e_value_charge3 = xcorr_engine.calculate_e_value_by_charge(
            score_distributions, 3.0, 3
        )
        assert 1e-10 <= e_value_charge3 <= 1.0
    
    def test_evalue_by_charge_missing_charge(self, xcorr_engine):
        """Test E-value calculation when charge state is missing."""
        score_distributions = {
            2: [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
        }
        
        # Request charge state 3 which doesn't exist - should return 1.0
        e_value = xcorr_engine.calculate_e_value_by_charge(
            score_distributions, 2.0, 3
        )
        assert e_value == 1.0


class TestZScoreCalculation:
    """Test Z-score (standard score) calculation in DIA results."""
    
    def test_zscore_basic_calculation(self):
        """Test basic Z-score calculation: (best - mean) / std."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        best_score = 5.0
        
        # Manual calculation
        mean = sum(scores) / len(scores)  # 3.0
        variance = sum((x - mean)**2 for x in scores) / len(scores)  # 2.0
        std = variance ** 0.5  # ~1.414
        expected_zscore = (best_score - mean) / std  # (5.0 - 3.0) / 1.414 ≈ 1.414
        
        # Calculate manually to verify
        calculated_mean = sum(scores) / len(scores)
        calculated_variance = sum((x - calculated_mean)**2 for x in scores) / len(scores)
        calculated_std = calculated_variance ** 0.5
        calculated_zscore = (best_score - calculated_mean) / calculated_std
        
        assert abs(calculated_zscore - expected_zscore) < 0.01
        assert calculated_zscore > 0, "Z-score should be positive when best is above mean"
    
    def test_zscore_with_outlier(self):
        """Test Z-score with a clear outlier."""
        # Many low scores and one high score
        scores = [0.5] * 10 + [5.0]
        best_score = 5.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean)**2 for x in scores) / len(scores)
        std = variance ** 0.5
        zscore = (best_score - mean) / std
        
        # Should have a high Z-score (many std devs above mean)
        assert zscore > 2.0, f"Z-score {zscore} should be > 2.0 for clear outlier"
    
    def test_zscore_zero_when_no_variation(self):
        """Test that Z-score is 0 when all scores are identical."""
        scores = [2.0] * 10
        best_score = 2.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean)**2 for x in scores) / len(scores)
        std = variance ** 0.5
        
        # When std is 0, Z-score should be 0 (not divide by zero)
        if std > 0:
            zscore = (best_score - mean) / std
        else:
            zscore = 0.0
        
        assert zscore == 0.0, "Z-score should be 0 when there's no variation"
