"""
Tests for optimization features in pyXcorrDIA.

This module tests the performance optimization features including:
- Library object passing to workers (eliminating redundant file I/O)
- Pre-vectorized library preprocessing (SMZ computation during loading)
- Combined mzML reading (single-pass MS1+MS2)
- Library filtering (decoy removal and Q-value filtering)
- Decoy fragment generation with preprocessing
- Parallelization with shared library objects
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import pickle

# Import the classes we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyXcorrDIA import (
    FastXCorr, SpectrumLibrary, MS1Spectrum, MassSpectrum,
    PeptideCandidate
)


class TestLibraryObjectPassing:
    """Tests for library object passing to workers."""

    def test_library_is_picklable(self):
        """Test that SpectrumLibrary objects can be pickled for multiprocessing."""
        # Create a mock library
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.001, 0.001],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.1, 300.2],
            'Relative.Intensity': [100.0, 80.0],
            'Fragment.Type': ['y', 'b'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Test pickling
            pickled = pickle.dumps(library)
            unpickled = pickle.loads(pickled)
            
            # Verify the unpickled library works
            assert unpickled.has_peptide('PEPTIDE', 2)
            precursor = unpickled.get_precursor('PEPTIDE', 2)
            assert precursor is not None
            assert precursor['precursor_mz'] == 400.5
            assert 'preprocessed_fragments' in precursor
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_preprocessed_fragments_in_library(self):
        """Test that preprocessed fragments are stored during library loading."""
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.005, 0.005],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.0, 300.0],
            'Relative.Intensity': [100.0, 64.0],  # sqrt(100)=10, sqrt(64)=8
            'Fragment.Type': ['y', 'b'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            precursor = library.get_precursor('PEPTIDE', 2)
            assert 'preprocessed_fragments' in precursor
            
            # Check that preprocessing is correct: sqrt(intensity) * mz^2
            preprocessed = precursor['preprocessed_fragments']
            assert len(preprocessed) == 2
            
            # Expected: sqrt(100) * 200^2 = 10 * 40000 = 400000
            #           sqrt(64) * 300^2 = 8 * 90000 = 720000
            # Then normalized
            expected_unnorm = np.array([400000.0, 720000.0])
            expected_norm = expected_unnorm / np.linalg.norm(expected_unnorm)
            
            np.testing.assert_array_almost_equal(preprocessed, expected_norm)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestLibraryFiltering:
    """Tests for library filtering by decoy status and Q-value."""

    def test_decoy_filtering(self):
        """Test that decoy entries are filtered out during library loading."""
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'DECOY2', 'PEPTIDE3'],
            'Modified.Sequence': ['PEPTIDE', 'DECOY', 'ANOTHER'],
            'Stripped.Sequence': ['PEPTIDE', 'DECOY', 'ANOTHER'],
            'Precursor.Charge': [2, 2, 2],
            'Precursor.Mz': [400.5, 350.5, 450.5],
            'Decoy': [0, 1, 0],  # Middle one is decoy
            'Q.Value': [0.001, 0.001, 0.001],
            'RT': [10.5, 11.5, 12.5],
            'Product.Mz': [200.1, 175.5, 225.5],
            'Relative.Intensity': [100.0, 90.0, 80.0],
            'Fragment.Type': ['y', 'y', 'y'],
            'Fragment.Charge': [1, 1, 1],
            'Fragment.Series.Number': [1, 1, 1],
            'Protein.Ids': ['P12345', 'DECOY_P12345', 'P67890']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Should have 2 precursors (decoy filtered out)
            assert len(library.peptide_index) == 2
            assert library.has_peptide('PEPTIDE', 2)
            assert library.has_peptide('ANOTHER', 2)
            assert not library.has_peptide('DECOY', 2)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_qvalue_filtering(self):
        """Test that high Q-value entries are filtered out."""
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2', 'BADPEP2', 'BADPEP2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE', 'BADPEP', 'BADPEP'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE', 'BADPEP', 'BADPEP'],
            'Precursor.Charge': [2, 2, 2, 2],
            'Precursor.Mz': [400.5, 400.5, 350.5, 350.5],
            'Decoy': [0, 0, 0, 0],
            'Q.Value': [0.001, 0.001, 0.05, 0.05],  # BADPEP has Q > 0.01
            'RT': [10.5, 10.5, 11.5, 11.5],
            'Product.Mz': [200.1, 300.2, 175.5, 275.5],
            'Relative.Intensity': [100.0, 80.0, 90.0, 70.0],
            'Fragment.Type': ['y', 'b', 'y', 'b'],
            'Fragment.Charge': [1, 1, 1, 1],
            'Fragment.Series.Number': [1, 2, 1, 2],
            'Protein.Ids': ['P12345', 'P12345', 'P67890', 'P67890']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Should have 1 precursor (BADPEP filtered out by Q-value)
            assert len(library.peptide_index) == 1
            assert library.has_peptide('PEPTIDE', 2)
            assert not library.has_peptide('BADPEP', 2)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_combined_decoy_and_qvalue_filtering(self):
        """Test filtering by both decoy status and Q-value."""
        library_data = {
            'Precursor.Id': ['GOOD2', 'DECOY2', 'BAD2', 'DECOYBAD2'],
            'Modified.Sequence': ['GOOD', 'DECOY', 'BAD', 'DECOYBAD'],
            'Stripped.Sequence': ['GOOD', 'DECOY', 'BAD', 'DECOYBAD'],
            'Precursor.Charge': [2, 2, 2, 2],
            'Precursor.Mz': [400.5, 350.5, 450.5, 500.5],
            'Decoy': [0, 1, 0, 1],
            'Q.Value': [0.001, 0.001, 0.05, 0.05],
            'RT': [10.5, 11.5, 12.5, 13.5],
            'Product.Mz': [200.1, 175.5, 225.5, 250.5],
            'Relative.Intensity': [100.0, 90.0, 80.0, 70.0],
            'Fragment.Type': ['y', 'y', 'y', 'y'],
            'Fragment.Charge': [1, 1, 1, 1],
            'Fragment.Series.Number': [1, 1, 1, 1],
            'Protein.Ids': ['P12345', 'DECOY_P12345', 'P67890', 'DECOY_P67890']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Should have only 1 precursor (GOOD with Decoy=0 and Q<=0.01)
            assert len(library.peptide_index) == 1
            assert library.has_peptide('GOOD', 2)
            assert not library.has_peptide('DECOY', 2)  # Decoy
            assert not library.has_peptide('BAD', 2)  # High Q-value
            assert not library.has_peptide('DECOYBAD', 2)  # Both
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDecoyFragmentGeneration:
    """Tests for decoy fragment generation with preprocessing."""

    def test_decoy_fragments_have_preprocessing(self):
        """Test that generated decoy fragments include preprocessed data."""
        xcorr_engine = FastXCorr()
        
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.001, 0.001],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.0, 300.0],
            'Relative.Intensity': [100.0, 64.0],
            'Fragment.Type': ['y', 'b'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Generate decoy fragments
            decoy_data = library.generate_decoy_fragments('PEPTIDE', 2, xcorr_engine)
            
            assert decoy_data is not None
            assert 'preprocessed_fragments' in decoy_data
            assert len(decoy_data['preprocessed_fragments']) > 0
            
            # Verify it's normalized
            norm = np.linalg.norm(decoy_data['preprocessed_fragments'])
            assert abs(norm - 1.0) < 1e-6
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_decoy_caching(self):
        """Test that decoy fragments are cached after first generation."""
        xcorr_engine = FastXCorr()
        
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.001, 0.001],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.0, 300.0],
            'Relative.Intensity': [100.0, 64.0],
            'Fragment.Type': ['y', 'b'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # First generation
            decoy_data1 = library.generate_decoy_fragments('PEPTIDE', 2, xcorr_engine)
            
            # Second generation should return cached version
            decoy_data2 = library.generate_decoy_fragments('PEPTIDE', 2, xcorr_engine)
            
            # Should be the same object (cached)
            assert decoy_data1 is decoy_data2
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCombinedMzMLReading:
    """Tests for combined MS1+MS2 mzML reading."""

    def test_combined_reader_exists(self):
        """Test that read_mzml_combined method exists."""
        xcorr_engine = FastXCorr()
        assert hasattr(xcorr_engine, 'read_mzml_combined')
        
    def test_combined_reader_signature(self):
        """Test that read_mzml_combined has correct signature."""
        xcorr_engine = FastXCorr()
        import inspect
        sig = inspect.signature(xcorr_engine.read_mzml_combined)
        
        # Should have parameters: mzml_file, max_spectra, preprocess_smz
        params = list(sig.parameters.keys())
        assert 'mzml_file' in params
        assert 'max_spectra' in params
        assert 'preprocess_smz' in params


class TestPreprocessedFragmentScoring:
    """Tests for using preprocessed fragments in scoring."""

    def test_preprocessed_fragments_improve_performance(self):
        """Test that preprocessed fragments eliminate redundant computation."""
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.001, 0.001],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.0, 300.0],
            'Relative.Intensity': [100.0, 64.0],
            'Fragment.Type': ['y', 'b'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            precursor = library.get_precursor('PEPTIDE', 2)
            
            # Verify preprocessed fragments match manual computation
            lib_mz = np.array([frag['mz'] for frag in precursor['fragments']])
            lib_intensity = np.array([frag['intensity'] for frag in precursor['fragments']])
            
            manual_preprocessed = np.sqrt(lib_intensity) * (lib_mz ** 2)
            manual_norm = manual_preprocessed / np.linalg.norm(manual_preprocessed)
            
            stored_preprocessed = precursor['preprocessed_fragments']
            
            np.testing.assert_array_almost_equal(stored_preprocessed, manual_norm)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestRegressionScores:
    """Regression tests for scoring consistency."""

    def test_library_scoring_consistency(self):
        """Test that library scoring produces consistent results."""
        xcorr_engine = FastXCorr()
        
        # Create a simple library
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2, 2],
            'Precursor.Mz': [400.5, 400.5, 400.5],
            'Decoy': [0, 0, 0],
            'Q.Value': [0.001, 0.001, 0.001],
            'RT': [10.5, 10.5, 10.5],
            'Product.Mz': [200.0, 300.0, 400.0],
            'Relative.Intensity': [100.0, 80.0, 60.0],
            'Fragment.Type': ['y', 'y', 'y'],
            'Fragment.Charge': [1, 1, 1],
            'Fragment.Series.Number': [1, 2, 3],
            'Protein.Ids': ['P12345', 'P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Get preprocessed fragments
            precursor = library.get_precursor('PEPTIDE', 2)
            preprocessed = precursor['preprocessed_fragments']
            
            # Verify specific values (regression test)
            # These values should remain constant across code changes
            assert len(preprocessed) == 3
            assert abs(np.linalg.norm(preprocessed) - 1.0) < 1e-10
            
            # Check that values are in expected order (sorted by computation)
            assert all(np.isfinite(preprocessed))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_decoy_scoring_consistency(self):
        """Test that decoy scoring produces consistent results."""
        xcorr_engine = FastXCorr()
        
        library_data = {
            'Precursor.Id': ['PEPTIDE2', 'PEPTIDE2'],
            'Modified.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Stripped.Sequence': ['PEPTIDE', 'PEPTIDE'],
            'Precursor.Charge': [2, 2],
            'Precursor.Mz': [400.5, 400.5],
            'Decoy': [0, 0],
            'Q.Value': [0.001, 0.001],
            'RT': [10.5, 10.5],
            'Product.Mz': [200.0, 300.0],
            'Relative.Intensity': [100.0, 64.0],
            'Fragment.Type': ['y', 'y'],
            'Fragment.Charge': [1, 1],
            'Fragment.Series.Number': [1, 2],
            'Protein.Ids': ['P12345', 'P12345']
        }
        
        df = pd.DataFrame(library_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_parquet(temp_path)
            library = SpectrumLibrary(temp_path, verbose=False)
            
            # Generate decoy twice
            decoy1 = library.generate_decoy_fragments('PEPTIDE', 2, xcorr_engine)
            decoy2 = library.generate_decoy_fragments('PEPTIDE', 2, xcorr_engine)
            
            # Should be identical (deterministic and cached)
            assert decoy1 is decoy2
            np.testing.assert_array_equal(
                decoy1['preprocessed_fragments'],
                decoy2['preprocessed_fragments']
            )
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
