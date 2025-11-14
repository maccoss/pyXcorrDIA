"""
Tests for spectrum library support in pyXcorrDIA.

This module tests the DIA-NN library integration including:
- Library loading and indexing
- UniMod modification parsing
- Decoy fragment generation with intensity remapping
- Fragment matching with ppm tolerance
- Cosine angle scoring with SMZ preprocessing
- MS1 isotope pattern prediction and scoring
- Integration with DIA search pipeline
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
    PeptideCandidate
)


class TestSpectrumLibrary:
    """Tests for SpectrumLibrary class."""

    def setup_method(self):
        """Create a mock library for testing."""
        self.xcorr_engine = FastXCorr()

    def create_mock_library_df(self):
        """Create a minimal mock DIA-NN library dataframe."""
        # Create a simple library with one peptide
        data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2, 2],
            'Proteotypic': [1, 1, 1],
            'Decoy': [0, 0, 0],
            'N.Term': [0, 0, 0],
            'C.Term': [0, 0, 0],
            'RT': [10.5, 10.5, 10.5],
            'IM': [0.0, 0.0, 0.0],
            'Q.Value': [0.001, 0.001, 0.001],
            'Peptidoform.Q.Value': [0.001, 0.001, 0.001],
            'PTM.Site.Confidence': [1.0, 1.0, 1.0],
            'PG.Q.Value': [0.001, 0.001, 0.001],
            'Precursor.Mz': [400.5, 400.5, 400.5],
            'Product.Mz': [500.25, 387.19, 274.15],
            'Relative.Intensity': [1.0, 0.8, 0.6],
            'Fragment.Type': ['y', 'y', 'y'],
            'Fragment.Charge': [1, 1, 1],
            'Fragment.Series.Number': [4, 3, 2],
            'Fragment.Loss.Type': ['noloss', 'noloss', 'noloss'],
            'Exclude.From.Quant': [0, 0, 0],
            'Protein.Ids': ['P12345', 'P12345', 'P12345'],
            'Protein.Group': ['P12345', 'P12345', 'P12345'],
            'Protein.Names': ['TEST_HUMAN', 'TEST_HUMAN', 'TEST_HUMAN'],
            'Genes': ['TEST', 'TEST', 'TEST'],
            'Flags': [0, 0, 0],
        }
        return pd.DataFrame(data)

    def test_library_creation_empty(self):
        """Test creating empty library."""
        library = SpectrumLibrary()
        assert library.library_df is None
        assert len(library.peptide_index) == 0

    def test_library_loading_from_dataframe(self):
        """Test loading library from a mock parquet file."""
        # Create temporary parquet file
        df = self.create_mock_library_df()

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            df.to_parquet(tmp_path)
            library = SpectrumLibrary(tmp_path)

            assert library.library_df is not None
            assert len(library.peptide_index) == 1
            assert ('PEPTIDE', 2) in library.peptide_index

            precursor = library.get_precursor('PEPTIDE', 2)
            assert precursor is not None
            assert precursor['sequence'] == 'PEPTIDE'
            assert precursor['charge'] == 2
            assert len(precursor['fragments']) == 3
            assert precursor['fragments'][0]['mz'] == 500.25
            assert precursor['fragments'][0]['intensity'] == 1.0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_library_invalid_file(self):
        """Test that invalid file raises appropriate error."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Not a parquet file")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="parquet format"):
                library = SpectrumLibrary(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_has_peptide(self):
        """Test checking if peptide exists in library."""
        df = self.create_mock_library_df()

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            df.to_parquet(tmp_path)
            library = SpectrumLibrary(tmp_path)

            assert library.has_peptide('PEPTIDE', 2) is True
            assert library.has_peptide('PEPTIDE', 3) is False
            assert library.has_peptide('NOTHERE', 2) is False
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_all_peptides(self):
        """Test getting all peptides from library."""
        df = self.create_mock_library_df()

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            df.to_parquet(tmp_path)
            library = SpectrumLibrary(tmp_path)

            peptides = library.get_all_peptides()
            assert len(peptides) == 1
            assert ('PEPTIDE', 2) in peptides
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestUniModParsing:
    """Tests for UniMod modification parsing."""

    def test_parse_unmodified_sequence(self):
        """Test parsing sequence without modifications."""
        sequence, mods = SpectrumLibrary.parse_unimod_sequence('PEPTIDE')
        assert sequence == 'PEPTIDE'
        assert len(mods) == 0

    def test_parse_carbamidomethyl(self):
        """Test parsing carbamidomethyl modification."""
        sequence, mods = SpectrumLibrary.parse_unimod_sequence('PEPC(UniMod:4)TIDE')
        assert sequence == 'PEPCTIDE'
        assert 3 in mods  # Position 3 (0-indexed)
        assert abs(mods[3] - 57.021464) < 0.0001

    def test_parse_multiple_modifications(self):
        """Test parsing multiple modifications."""
        sequence, mods = SpectrumLibrary.parse_unimod_sequence('M(UniMod:35)PEPC(UniMod:4)TIDE')
        assert sequence == 'MPEPCTIDE'
        assert 0 in mods  # Oxidation on M
        assert 4 in mods  # Carbamidomethyl on C
        assert abs(mods[0] - 15.994915) < 0.0001
        assert abs(mods[4] - 57.021464) < 0.0001

    def test_parse_unknown_unimod(self):
        """Test parsing unknown UniMod IDs."""
        # Unknown UniMod IDs should be skipped
        sequence, mods = SpectrumLibrary.parse_unimod_sequence('PEPC(UniMod:9999)TIDE')
        assert sequence == 'PEPCTIDE'
        assert len(mods) == 0  # Unknown mod not added


class TestDecoyFragmentGeneration:
    """Tests for decoy fragment generation with intensity remapping."""

    def setup_method(self):
        """Setup test fixtures."""
        self.xcorr_engine = FastXCorr()

    def test_reverse_sequence_with_kr_terminus(self):
        """Test sequence reversal keeping K/R at C-terminus."""
        assert SpectrumLibrary._reverse_sequence('PEPTIDER') == 'EDITPEPR'
        assert SpectrumLibrary._reverse_sequence('PEPTIDEK') == 'EDITPEPK'
        assert SpectrumLibrary._reverse_sequence('PEPTIDE') == 'EDITPEP'

    def test_reverse_sequence_short(self):
        """Test reversing short sequences."""
        assert SpectrumLibrary._reverse_sequence('K') == 'K'
        assert SpectrumLibrary._reverse_sequence('PK') == 'PK'
        assert SpectrumLibrary._reverse_sequence('ABC') == 'CBA'

    def test_fragment_mz_calculation_y_ions(self):
        """Test y-ion m/z calculation."""
        mz = SpectrumLibrary._calculate_fragment_mz('PEPTIDE', 'y', 3, 1, self.xcorr_engine)
        # y3 = IDE + H2O + H = ~376
        assert mz > 0
        assert 370 < mz < 385

    def test_fragment_mz_calculation_b_ions(self):
        """Test b-ion m/z calculation."""
        mz = SpectrumLibrary._calculate_fragment_mz('PEPTIDE', 'b', 3, 1, self.xcorr_engine)
        # b3 = PEP + H = ~324
        assert mz > 0
        assert 300 < mz < 350

    def test_fragment_mz_invalid_inputs(self):
        """Test invalid fragment inputs."""
        # Invalid series number
        assert SpectrumLibrary._calculate_fragment_mz('PEPTIDE', 'y', 0, 1, self.xcorr_engine) == 0
        assert SpectrumLibrary._calculate_fragment_mz('PEPTIDE', 'y', 10, 1, self.xcorr_engine) == 0

        # Invalid fragment type
        assert SpectrumLibrary._calculate_fragment_mz('PEPTIDE', 'z', 3, 1, self.xcorr_engine) == 0


class TestIsotopePatternPrediction:
    """Tests for isotope pattern prediction."""

    def setup_method(self):
        """Setup test fixtures."""
        self.xcorr_engine = FastXCorr()

    def test_isotope_pattern_prediction(self):
        """Test basic isotope pattern prediction."""
        pattern = FastXCorr.predict_isotope_pattern('PEPTIDE', 2, self.xcorr_engine.aa_masses)

        assert len(pattern) == 5
        assert pattern[0] == 0.0  # M-1 is always 0
        assert pattern[1] > 0  # M+0 (monoisotopic)
        assert pattern[2] > 0  # M+1
        assert pattern[3] > 0  # M+2
        assert pattern[4] >= 0  # M+3

        # Check normalization (excluding M-1)
        assert abs(np.sum(pattern[1:]) - 1.0) < 0.001

    def test_isotope_pattern_decreasing(self):
        """Test that isotope intensities generally decrease."""
        pattern = FastXCorr.predict_isotope_pattern('PEPTIDE', 2, self.xcorr_engine.aa_masses)

        # M+0 should be highest for small peptides
        assert pattern[1] >= pattern[2]
        assert pattern[2] >= pattern[3]

    def test_isotope_mz_values(self):
        """Test isotope m/z calculation."""
        precursor_mz = 500.0
        charge = 2

        mz_values = FastXCorr.calculate_isotope_mz_values(precursor_mz, charge)

        assert len(mz_values) == 5
        assert abs(mz_values[0] - (precursor_mz - 1.002868/2)) < 0.001  # M-1
        assert abs(mz_values[1] - precursor_mz) < 0.001  # M+0
        assert abs(mz_values[2] - (precursor_mz + 1.002868/2)) < 0.001  # M+1
        assert abs(mz_values[3] - (precursor_mz + 2*1.002868/2)) < 0.001  # M+2
        assert abs(mz_values[4] - (precursor_mz + 3*1.002868/2)) < 0.001  # M+3


class TestFragmentMatching:
    """Tests for fragment m/z matching with tolerance."""

    def test_match_fragments_ppm_perfect_match(self):
        """Test fragment matching with perfect m/z match."""
        exp_mz = np.array([100.0, 200.0, 300.0])
        exp_intensity = np.array([10.0, 20.0, 30.0])

        lib_fragments = [
            {'mz': 200.0, 'intensity': 1.0},
        ]

        matched_exp, matched_lib = FastXCorr.match_fragments_ppm(
            exp_mz, exp_intensity, lib_fragments, tolerance_ppm=10.0
        )

        assert len(matched_exp) == 1
        assert matched_exp[0] == 20.0
        assert matched_lib[0] == 1.0

    def test_match_fragments_ppm_no_match(self):
        """Test fragment matching with no matches."""
        exp_mz = np.array([100.0, 200.0, 300.0])
        exp_intensity = np.array([10.0, 20.0, 30.0])

        lib_fragments = [
            {'mz': 500.0, 'intensity': 1.0},  # Too far away
        ]

        matched_exp, matched_lib = FastXCorr.match_fragments_ppm(
            exp_mz, exp_intensity, lib_fragments, tolerance_ppm=10.0
        )

        assert len(matched_exp) == 1
        assert matched_exp[0] == 0.0  # No match
        assert matched_lib[0] == 1.0

    def test_match_fragments_ppm_within_tolerance(self):
        """Test fragment matching within ppm tolerance."""
        exp_mz = np.array([100.001, 200.002])  # Slight offset
        exp_intensity = np.array([10.0, 20.0])

        lib_fragments = [
            {'mz': 100.0, 'intensity': 1.0},
            {'mz': 200.0, 'intensity': 2.0},
        ]

        matched_exp, matched_lib = FastXCorr.match_fragments_ppm(
            exp_mz, exp_intensity, lib_fragments, tolerance_ppm=20.0
        )

        assert len(matched_exp) == 2
        assert matched_exp[0] == 10.0
        assert matched_exp[1] == 20.0


class TestCosineAngleScoring:
    """Tests for cosine angle calculation."""

    def test_cosine_angle_identical_vectors(self):
        """Test cosine angle of identical vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        cosine = FastXCorr.calculate_cosine_angle(vec1, vec2)
        assert abs(cosine - 1.0) < 0.001

    def test_cosine_angle_orthogonal_vectors(self):
        """Test cosine angle of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])

        cosine = FastXCorr.calculate_cosine_angle(vec1, vec2)
        assert abs(cosine - 0.0) < 0.001

    def test_cosine_angle_opposite_vectors(self):
        """Test cosine angle of opposite vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])

        cosine = FastXCorr.calculate_cosine_angle(vec1, vec2)
        # Cosine is clamped to [0, 1], so -1 becomes 0
        assert cosine >= 0.0

    def test_cosine_angle_zero_vectors(self):
        """Test cosine angle with zero vectors."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        cosine = FastXCorr.calculate_cosine_angle(vec1, vec2)
        assert cosine == 0.0


class TestLibraryCosineScoring:
    """Tests for library cosine scoring with SMZ preprocessing."""

    def test_library_cosine_score_perfect_match(self):
        """Test library scoring with perfect match."""
        exp_mz = np.array([100.0, 200.0, 300.0])
        exp_intensity = np.array([10.0, 20.0, 30.0])

        lib_fragments = [
            {'mz': 100.0, 'intensity': 10.0},
            {'mz': 200.0, 'intensity': 20.0},
            {'mz': 300.0, 'intensity': 30.0},
        ]

        score = FastXCorr.calculate_library_cosine_score(
            exp_mz, exp_intensity, lib_fragments, tolerance_ppm=10.0
        )

        # Perfect match should give high score
        assert score > 0.9

    def test_library_cosine_score_no_match(self):
        """Test library scoring with no matches."""
        exp_mz = np.array([100.0, 200.0, 300.0])
        exp_intensity = np.array([10.0, 20.0, 30.0])

        lib_fragments = [
            {'mz': 500.0, 'intensity': 10.0},  # No overlap
        ]

        score = FastXCorr.calculate_library_cosine_score(
            exp_mz, exp_intensity, lib_fragments, tolerance_ppm=10.0
        )

        assert score == 0.0


class TestMS1SpectrumManagement:
    """Tests for MS1 spectrum management."""

    def test_find_closest_ms1_exact_match(self):
        """Test finding exact RT match."""
        ms1_spectra = [
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan1', 5.0),
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan2', 10.0),
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan3', 15.0),
        ]

        closest = FastXCorr.find_closest_ms1(ms1_spectra, 10.0)
        assert closest.retention_time == 10.0
        assert closest.scan_id == 'scan2'

    def test_find_closest_ms1_interpolate(self):
        """Test finding closest MS1 between scans."""
        ms1_spectra = [
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan1', 5.0),
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan2', 10.0),
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan3', 15.0),
        ]

        closest = FastXCorr.find_closest_ms1(ms1_spectra, 8.0)
        assert closest.retention_time == 10.0  # Closer to 10 than 5

    def test_find_closest_ms1_boundary_cases(self):
        """Test boundary cases for MS1 finding."""
        ms1_spectra = [
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan1', 5.0),
            MS1Spectrum(np.array([100.0]), np.array([10.0]), 'scan2', 10.0),
        ]

        # Before first
        closest = FastXCorr.find_closest_ms1(ms1_spectra, 1.0)
        assert closest.retention_time == 5.0

        # After last
        closest = FastXCorr.find_closest_ms1(ms1_spectra, 20.0)
        assert closest.retention_time == 10.0

    def test_find_closest_ms1_empty_list(self):
        """Test with empty MS1 list."""
        closest = FastXCorr.find_closest_ms1([], 10.0)
        assert closest is None


class TestIsotopeEnvelopeExtraction:
    """Tests for isotope envelope extraction from MS1."""

    def test_extract_isotope_envelope(self):
        """Test extracting isotope envelope from MS1."""
        # Create mock MS1 with isotope pattern
        mz_array = np.array([499.5, 500.0, 500.5, 501.0, 501.5])  # Charge 2+ pattern
        intensity_array = np.array([0.0, 100.0, 60.0, 20.0, 5.0])

        ms1 = MS1Spectrum(mz_array, intensity_array, 'scan1', 10.0)

        # Extract for charge 2+ at m/z 500.0
        envelope = FastXCorr.extract_isotope_envelope(ms1, 500.0, 2, tolerance_ppm=20.0)

        assert len(envelope) == 5
        assert envelope[1] == 100.0  # M+0
        assert envelope[2] == 60.0   # M+1
        assert envelope[3] == 20.0   # M+2

    def test_extract_isotope_envelope_missing_peaks(self):
        """Test extraction with missing isotope peaks."""
        # Only M+0 present
        mz_array = np.array([500.0])
        intensity_array = np.array([100.0])

        ms1 = MS1Spectrum(mz_array, intensity_array, 'scan1', 10.0)

        envelope = FastXCorr.extract_isotope_envelope(ms1, 500.0, 2, tolerance_ppm=20.0)

        assert len(envelope) == 5
        assert envelope[1] == 100.0  # M+0 found
        assert envelope[0] == 0.0    # M-1 not found
        assert envelope[2] == 0.0    # M+1 not found


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
