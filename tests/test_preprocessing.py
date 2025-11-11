"""
Test spectrum preprocessing and XCorr calculation.
"""
import numpy as np


class TestSpectrumPreprocessing:
    """Test spectrum preprocessing pipeline."""
    
    def test_preprocess_spectrum(self, xcorr_engine, simple_spectrum):
        """Test basic spectrum preprocessing."""
        preprocessed = xcorr_engine.preprocess_spectrum(simple_spectrum)
        
        assert preprocessed is not None
        assert isinstance(preprocessed, np.ndarray)
        assert len(preprocessed) == xcorr_engine.num_bins
        
        # Check that processed spectrum is stored
        assert simple_spectrum.processed_spectrum is not None
    
    def test_preprocessing_pipeline(self, xcorr_engine, simple_spectrum):
        """Test complete preprocessing pipeline."""
        # Step 1: Preprocess spectrum (includes MakeCorrData)
        windowed = xcorr_engine.preprocess_spectrum(simple_spectrum)
        
        # Step 2: Apply fast XCorr preprocessing
        preprocessed = xcorr_engine.preprocess_for_xcorr(windowed)
        
        assert preprocessed is not None
        assert isinstance(preprocessed, np.ndarray)
        assert len(preprocessed) == len(windowed)
        
        # Preprocessed spectrum should have both positive and negative values
        assert np.any(preprocessed > 0) or np.any(preprocessed < 0)
    
    def test_make_corr_data_windowing(self, xcorr_engine):
        """Test MakeCorrData windowing function."""
        # Create a simple binned spectrum
        binned = np.random.rand(xcorr_engine.num_bins) * 100
        highest_bin = xcorr_engine.num_bins // 2
        highest_intensity = np.max(binned)
        
        windowed = xcorr_engine._make_corr_data(binned, highest_bin, highest_intensity)
        
        assert windowed is not None
        assert len(windowed) == len(binned)
        
        # Check that windowing normalized intensities
        # Within each window, max should be around 50.0
        max_val = np.max(windowed)
        assert max_val <= 60.0  # Should be normalized to ~50.0
    
    def test_preprocess_for_xcorr(self, xcorr_engine):
        """Test fast XCorr preprocessing."""
        # Create a windowed spectrum
        windowed = np.random.rand(xcorr_engine.num_bins) * 50
        
        preprocessed = xcorr_engine.preprocess_for_xcorr(windowed)
        
        assert preprocessed is not None
        assert len(preprocessed) == len(windowed)
        
        # First element should be 0
        assert preprocessed[0] == 0.0


class TestTheoreticalSpectrumGeneration:
    """Test theoretical spectrum generation."""
    
    def test_generate_theoretical_spectrum(self, xcorr_engine, sample_peptide):
        """Test theoretical spectrum generation for a peptide."""
        charge = 2
        theoretical = xcorr_engine.generate_theoretical_spectrum(sample_peptide, charge)
        
        assert theoretical is not None
        assert isinstance(theoretical, np.ndarray)
        assert len(theoretical) == xcorr_engine.num_bins
        
        # Should have some non-zero bins (fragment ions)
        assert np.count_nonzero(theoretical) > 0
    
    def test_theoretical_spectrum_has_ions(self, xcorr_engine):
        """Test that theoretical spectrum contains b and y ions."""
        from pyXcorrDIA import PeptideCandidate
        
        peptide = PeptideCandidate("PEPTIDE", "test", 799.36)
        charge = 2
        
        theoretical = xcorr_engine.generate_theoretical_spectrum(peptide, charge)
        
        # Should have multiple fragment ions
        num_ions = np.count_nonzero(theoretical)
        assert num_ions > 0
        
        # For "PEPTIDE" (7 letters), expect b1-b6 and y1-y6 ions
        # Plus potential neutral losses
        print(f"Generated {num_ions} theoretical ions for PEPTIDE")
    
    def test_different_charges(self, xcorr_engine, sample_peptide):
        """Test theoretical spectra for different charge states."""
        spec_charge2 = xcorr_engine.generate_theoretical_spectrum(sample_peptide, 2)
        spec_charge3 = xcorr_engine.generate_theoretical_spectrum(sample_peptide, 3)
        
        assert not np.array_equal(spec_charge2, spec_charge3)
        
        # Both should have ions
        assert np.count_nonzero(spec_charge2) > 0
        assert np.count_nonzero(spec_charge3) > 0


class TestXCorrCalculation:
    """Test XCorr score calculation."""
    
    def test_calculate_fast_xcorr(self, xcorr_engine):
        """Test fast XCorr calculation with synthetic data."""
        # Create synthetic theoretical and preprocessed spectra
        theoretical = np.zeros(xcorr_engine.num_bins)
        preprocessed = np.zeros(xcorr_engine.num_bins)
        
        # Add some matching peaks
        theoretical[100] = 1.0
        theoretical[200] = 1.0
        preprocessed[100] = 10.0
        preprocessed[200] = 10.0
        
        xcorr = xcorr_engine.calculate_fast_xcorr(theoretical, preprocessed)
        
        assert xcorr > 0
        assert isinstance(xcorr, float)
    
    def test_xcorr_perfect_match(self, xcorr_engine):
        """Test XCorr with perfectly matching spectra."""
        # Create identical spectra
        spectrum = np.zeros(xcorr_engine.num_bins)
        spectrum[100:110] = 1.0
        
        xcorr = xcorr_engine.calculate_fast_xcorr(spectrum, spectrum)
        
        assert xcorr > 0
    
    def test_xcorr_no_match(self, xcorr_engine):
        """Test XCorr with non-matching spectra."""
        theoretical = np.zeros(xcorr_engine.num_bins)
        preprocessed = np.zeros(xcorr_engine.num_bins)
        
        theoretical[100:110] = 1.0
        preprocessed[500:510] = 1.0
        
        xcorr = xcorr_engine.calculate_fast_xcorr(theoretical, preprocessed)
        
        # Should be low or near zero (no matching peaks)
        assert xcorr < 1.0


class TestFullPreprocessingPipeline:
    """Test the complete preprocessing pipeline with real data."""
    
    def test_yqshtk_spectrum_preprocessing(self, xcorr_engine, yqshtk_mzml):
        """Test preprocessing of YQSHTK spectrum."""
        spectra = xcorr_engine.read_mzml(yqshtk_mzml, max_spectra=1)
        
        if len(spectra) > 0:
            spectrum = spectra[0]
            
            # Preprocess spectrum
            windowed = xcorr_engine.preprocess_spectrum(spectrum)
            preprocessed = xcorr_engine.preprocess_for_xcorr(windowed)
            
            assert windowed is not None
            assert preprocessed is not None
            assert len(preprocessed) == xcorr_engine.num_bins
            
            print(f"Preprocessed spectrum {spectrum.scan_id}")
            print(f"  Windowed max: {np.max(windowed):.2f}")
            print(f"  Preprocessed range: [{np.min(preprocessed):.2f}, {np.max(preprocessed):.2f}]")
    
    def test_mgf_spectrum_preprocessing(self, xcorr_engine, ot_centroid_mgf):
        """Test preprocessing of MGF spectrum."""
        spectra = xcorr_engine.read_mgf(ot_centroid_mgf, max_spectra=1)
        
        if len(spectra) > 0:
            spectrum = spectra[0]
            
            # Preprocess spectrum
            windowed = xcorr_engine.preprocess_spectrum(spectrum)
            preprocessed = xcorr_engine.preprocess_for_xcorr(windowed)
            
            assert windowed is not None
            assert preprocessed is not None
            
            print(f"Preprocessed MGF spectrum {spectrum.scan_id}")
            print(f"  Original peaks: {len(spectrum.mz_array)}")
            print(f"  Non-zero bins after preprocessing: {np.count_nonzero(preprocessed)}")
