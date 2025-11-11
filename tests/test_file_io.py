"""
Test file I/O operations: reading FASTA, mzML, and MGF files.
"""
import pytest


class TestFASTAReading:
    """Test FASTA file reading functionality."""
    
    def test_read_yqshtk_fasta(self, xcorr_engine, yqshtk_fasta):
        """Test reading YQSHTK FASTA file."""
        proteins = xcorr_engine.read_fasta(yqshtk_fasta)
        
        assert len(proteins) > 0
        assert isinstance(proteins, dict)
        
        # Check that we have protein entries
        for protein_id, sequence in proteins.items():
            assert isinstance(protein_id, str)
            assert isinstance(sequence, str)
            assert len(sequence) > 0
            # Check that sequence contains only valid amino acids
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            assert all(aa in valid_aa for aa in sequence)
    
    def test_read_large_fasta(self, xcorr_engine, large_fasta):
        """Test reading larger FASTA database."""
        proteins = xcorr_engine.read_fasta(large_fasta)
        
        assert len(proteins) > 0
        print(f"Loaded {len(proteins)} proteins from large FASTA")
        
        # Verify structure
        for protein_id, sequence in proteins.items():
            assert len(sequence) > 0
            assert isinstance(sequence, str)


class TestMzMLReading:
    """Test mzML file reading functionality."""
    
    def test_read_yqshtk_mzml(self, xcorr_engine, yqshtk_mzml):
        """Test reading YQSHTK mzML file."""
        spectra = xcorr_engine.read_mzml(yqshtk_mzml, max_spectra=0)
        
        assert len(spectra) > 0
        print(f"Read {len(spectra)} spectra from mzML")
        
        # Check first spectrum
        spectrum = spectra[0]
        assert hasattr(spectrum, 'mz_array')
        assert hasattr(spectrum, 'intensity_array')
        assert len(spectrum.mz_array) > 0
        assert len(spectrum.intensity_array) > 0
        assert len(spectrum.mz_array) == len(spectrum.intensity_array)
    
    def test_read_limited_spectra(self, xcorr_engine, yqshtk_mzml):
        """Test reading limited number of spectra."""
        max_spectra = 5
        spectra = xcorr_engine.read_mzml(yqshtk_mzml, max_spectra=max_spectra)
        
        assert len(spectra) <= max_spectra
        
    def test_spectrum_metadata(self, xcorr_engine, yqshtk_mzml):
        """Test that spectrum metadata is correctly read."""
        spectra = xcorr_engine.read_mzml(yqshtk_mzml, max_spectra=1)
        
        if len(spectra) > 0:
            spectrum = spectra[0]
            assert spectrum.scan_id is not None
            assert spectrum.precursor_mz > 0
            assert spectrum.charge > 0
            assert hasattr(spectrum, 'isolation_window_lower')
            assert hasattr(spectrum, 'isolation_window_upper')


class TestMGFReading:
    """Test MGF file reading functionality."""
    
    def test_read_mgf_file(self, xcorr_engine, ot_centroid_mgf):
        """Test reading MGF file."""
        spectra = xcorr_engine.read_mgf(ot_centroid_mgf, max_spectra=0)
        
        assert len(spectra) > 0
        print(f"Read {len(spectra)} spectra from MGF")
        
        # Check first spectrum structure
        spectrum = spectra[0]
        assert hasattr(spectrum, 'mz_array')
        assert hasattr(spectrum, 'intensity_array')
        assert len(spectrum.mz_array) > 0
        assert len(spectrum.intensity_array) > 0
    
    def test_mgf_spectrum_properties(self, xcorr_engine, ot_centroid_mgf):
        """Test MGF spectrum properties."""
        spectra = xcorr_engine.read_mgf(ot_centroid_mgf, max_spectra=1)
        
        if len(spectra) > 0:
            spectrum = spectra[0]
            # MGF should have precursor information
            assert hasattr(spectrum, 'precursor_mz')
            assert hasattr(spectrum, 'charge')
            assert spectrum.precursor_mz > 0


class TestSingleSpectrumReading:
    """Test reading single spectra by scan ID."""
    
    def test_read_single_spectrum_mzml(self, xcorr_engine, yqshtk_mzml):
        """Test reading a single spectrum from mzML by scan ID."""
        # First get a list of available scan IDs
        all_spectra = xcorr_engine.read_mzml(yqshtk_mzml, max_spectra=10)
        
        if len(all_spectra) > 0:
            # Try to read first spectrum by its scan ID
            target_scan_id = all_spectra[0].scan_id
            
            spectrum = xcorr_engine.read_single_spectrum(yqshtk_mzml, target_scan_id)
            
            assert spectrum is not None
            assert spectrum.scan_id == target_scan_id
            assert len(spectrum.mz_array) > 0
    
    def test_read_single_spectrum_mgf(self, xcorr_engine, ot_centroid_mgf):
        """Test reading a single spectrum from MGF by scan ID."""
        # First get a list of available scan IDs
        all_spectra = xcorr_engine.read_mgf(ot_centroid_mgf, max_spectra=10)
        
        if len(all_spectra) > 0:
            # Try to read first spectrum by its scan ID
            target_scan_id = all_spectra[0].scan_id
            
            spectrum = xcorr_engine.read_single_spectrum(ot_centroid_mgf, target_scan_id)
            
            assert spectrum is not None
            assert spectrum.scan_id == target_scan_id
            assert len(spectrum.mz_array) > 0
    
    def test_invalid_scan_id(self, xcorr_engine, yqshtk_mzml):
        """Test that invalid scan ID raises appropriate error."""
        with pytest.raises(ValueError):
            xcorr_engine.read_single_spectrum(yqshtk_mzml, "invalid_scan_999999")
