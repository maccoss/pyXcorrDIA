"""
Test basic functionality of pyXcorrDIA classes and core methods.
"""
import pytest
import numpy as np
from pyXcorrDIA import FastXCorr, MassSpectrum, PeptideCandidate


class TestFastXCorrInitialization:
    """Test FastXCorr class initialization."""
    
    def test_default_initialization(self):
        """Test FastXCorr initializes with default parameters."""
        engine = FastXCorr()
        assert engine.bin_width == 1.0005079
        assert engine.bin_offset == 0.4
        assert engine.mass_range == (0, 2000)
        assert engine.num_bins > 0
        
    def test_custom_bin_parameters(self):
        """Test FastXCorr with custom bin parameters."""
        engine = FastXCorr(bin_width=0.02, bin_offset=0.0)
        assert engine.bin_width == 0.02
        assert engine.bin_offset == 0.0
        
    def test_default_modifications(self):
        """Test default static modifications (Carbamidomethyl-C)."""
        engine = FastXCorr()
        mods = engine.get_static_modifications()
        assert 'C' in mods
        assert mods['C'] == 57.021464
        
    def test_no_modifications(self):
        """Test initialization with no modifications."""
        engine = FastXCorr(static_modifications={})
        mods = engine.get_static_modifications()
        assert len(mods) == 0
        
    def test_amino_acid_masses(self):
        """Test that amino acid masses are properly initialized."""
        engine = FastXCorr()
        # Test a few key amino acids
        assert 'A' in engine.aa_masses
        assert 'K' in engine.aa_masses
        assert 'C' in engine.aa_masses
        # With default Carbamidomethyl-C, C mass should be modified
        assert engine.aa_masses['C'] > engine.base_aa_masses['C']


class TestMassSpectrum:
    """Test MassSpectrum class."""
    
    def test_spectrum_creation(self, simple_spectrum):
        """Test creating a MassSpectrum object."""
        assert simple_spectrum.scan_id == "test_001"
        assert simple_spectrum.precursor_mz == 600.0
        assert simple_spectrum.charge == 2
        assert len(simple_spectrum.mz_array) == 5
        assert len(simple_spectrum.intensity_array) == 5
        
    def test_spectrum_attributes(self, simple_spectrum):
        """Test MassSpectrum has correct attributes."""
        assert hasattr(simple_spectrum, 'mz_array')
        assert hasattr(simple_spectrum, 'intensity_array')
        assert hasattr(simple_spectrum, 'processed_spectrum')
        assert hasattr(simple_spectrum, 'preprocessed_spectrum')
        assert simple_spectrum.processed_spectrum is None
        assert simple_spectrum.preprocessed_spectrum is None


class TestPeptideCandidate:
    """Test PeptideCandidate class."""
    
    def test_peptide_creation(self, sample_peptide):
        """Test creating a PeptideCandidate object."""
        assert sample_peptide.sequence == "YQSHTK"
        assert sample_peptide.protein_id == "test_protein"
        # Mass should be calculated correctly (with default C+57 modification)
        assert sample_peptide.mass > 0
        assert 760 < sample_peptide.mass < 770  # Approximately 762.37
        
    def test_peptide_attributes(self, sample_peptide):
        """Test PeptideCandidate has correct attributes."""
        assert hasattr(sample_peptide, 'sequence')
        assert hasattr(sample_peptide, 'protein_id')
        assert hasattr(sample_peptide, 'mass')
        assert hasattr(sample_peptide, 'theoretical_spectrum')
        assert sample_peptide.theoretical_spectrum is None


class TestBinningFunctions:
    """Test mass binning functions."""
    
    def test_bin_mass_default(self, xcorr_engine):
        """Test bin_mass with default Comet parameters."""
        # Test with known values
        bin_idx = xcorr_engine.bin_mass(100.0)
        assert isinstance(bin_idx, int)
        assert bin_idx >= 0
        
    def test_bin_mass_consistency(self, xcorr_engine):
        """Test that bin_mass is consistent."""
        mass = 500.5
        bin1 = xcorr_engine.bin_mass(mass)
        bin2 = xcorr_engine.bin_mass(mass)
        assert bin1 == bin2
        
    def test_bin_mass_ordering(self, xcorr_engine):
        """Test that larger masses give larger bin indices."""
        bin1 = xcorr_engine.bin_mass(100.0)
        bin2 = xcorr_engine.bin_mass(200.0)
        bin3 = xcorr_engine.bin_mass(300.0)
        assert bin1 < bin2 < bin3


class TestStaticModifications:
    """Test static modification management."""
    
    def test_add_modification(self, xcorr_engine_no_mods):
        """Test adding a static modification."""
        engine = xcorr_engine_no_mods
        assert len(engine.get_static_modifications()) == 0
        
        engine.add_static_modification('M', 15.994915)
        mods = engine.get_static_modifications()
        assert 'M' in mods
        assert mods['M'] == 15.994915
        
    def test_remove_modification(self, xcorr_engine_with_mods):
        """Test removing a static modification."""
        engine = xcorr_engine_with_mods
        assert 'C' in engine.get_static_modifications()
        
        engine.remove_static_modification('C')
        mods = engine.get_static_modifications()
        assert 'C' not in mods
        
    def test_modification_affects_mass(self, xcorr_engine_no_mods):
        """Test that modifications affect calculated masses."""
        engine = xcorr_engine_no_mods
        
        # Calculate mass without modification
        mass1 = engine.calculate_peptide_mass("ACDEFGHIK")
        
        # Add modification to C
        engine.add_static_modification('C', 57.021464)
        mass2 = engine.calculate_peptide_mass("ACDEFGHIK")
        
        # Mass should increase by modification amount
        assert abs((mass2 - mass1) - 57.021464) < 0.001


class TestPeptideMassCalculation:
    """Test peptide mass calculation."""
    
    def test_simple_peptide_mass(self, xcorr_engine_no_mods):
        """Test mass calculation for simple peptides."""
        engine = xcorr_engine_no_mods
        
        # Single amino acid (A) + water
        mass_a = engine.calculate_peptide_mass("A")
        expected = engine.aa_masses['A'] + engine.h2o_mass
        assert abs(mass_a - expected) < 0.001
        
    def test_known_peptide_mass(self, xcorr_engine_no_mods):
        """Test mass calculation for known peptide YQSHTK."""
        engine = xcorr_engine_no_mods
        mass = engine.calculate_peptide_mass("YQSHTK")
        
        # Calculate expected mass
        expected = (engine.aa_masses['Y'] + engine.aa_masses['Q'] + 
                   engine.aa_masses['S'] + engine.aa_masses['H'] + 
                   engine.aa_masses['T'] + engine.aa_masses['K'] + 
                   engine.h2o_mass)
        assert abs(mass - expected) < 0.001
        
    def test_peptide_mass_with_modifications(self, xcorr_engine_with_mods):
        """Test mass calculation with static modifications."""
        engine = xcorr_engine_with_mods
        
        # Peptide without C
        mass1 = engine.calculate_peptide_mass("YQSHTK")
        
        # Peptide with C
        mass2 = engine.calculate_peptide_mass("CYQSHTK")
        
        # Difference should include the modification
        mass_diff = mass2 - mass1
        expected_diff = engine.base_aa_masses['C'] + 57.021464
        assert abs(mass_diff - expected_diff) < 0.01
