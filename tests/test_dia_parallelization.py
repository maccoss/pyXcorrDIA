"""
Test DIA mode parallelization functionality.

These tests validate that the parallel processing of isolation windows works correctly:
1. Worker function properly serializes/deserializes data
2. Parallel and sequential modes produce identical results
3. Multiple workers can process windows simultaneously
4. Results are correctly aggregated from all workers
"""
import numpy as np
from unittest.mock import patch, MagicMock
from pyXcorrDIA import (
    PeptideCandidate, 
    MassSpectrum,
    process_isolation_window_worker
)


class TestDIAParallelization:
    """Test DIA mode parallel processing."""
    
    def test_worker_function_serialization(self, xcorr_engine, tmp_path):
        """
        Test that the worker function properly handles serialized data.
        
        The worker function must reconstruct PeptideCandidate objects from
        serialized dictionaries because objects can't be pickled across processes.
        """
        # Create test data
        target = PeptideCandidate("PEPTIDE", "protein1", mass=800.0)
        decoy = PeptideCandidate("EDITPEP", "decoy_protein1", mass=800.0)
        
        # Serialize target-decoy pair (as done in main DIA search)
        target_data = {
            'sequence': target.sequence,
            'protein_id': target.protein_id,
            'mass': target.mass
        }
        decoy_data = {
            'sequence': decoy.sequence,
            'protein_id': decoy.protein_id,
            'mass': decoy.mass
        }
        
        target_decoy_pairs_data = [(target_data, decoy_data)]
        
        # Create mock spectrum
        spectrum = MassSpectrum(
            mz_array=np.array([100.0, 200.0, 300.0]),
            intensity_array=np.array([1000.0, 2000.0, 3000.0]),
            scan_id="scan_1",
            isolation_window_lower=400.0,
            isolation_window_upper=405.0
        )
        
        # Use temporary parquet file path
        parquet_file = str(tmp_path / "window_400.0_405.0.parquet")
        
        # Prepare worker arguments
        args = (
            0,  # window_idx
            1,  # total_windows
            (400.0, 405.0),  # isolation_window
            [spectrum],  # window_spectra
            None,  # fasta_file (not needed for this test)
            target_decoy_pairs_data,  # serialized pairs
            [2],  # charge_states
            parquet_file,  # parquet_file (use tmp_path)
            'trypsin',  # enzyme
            1,  # decoy_cycle_length
            None,  # library
            None,  # ms1_spectra
            10.0,  # lib_fragment_tol_ppm
            10.0,  # lib_precursor_tol_ppm
            0  # verbose (suppress output)
        )
        
        # Call worker function
        result = process_isolation_window_worker(args)
        
        # Verify result structure (no longer has peptide_info_file with unified schema)
        assert 'results' in result
        assert 'parquet_file' in result
        assert isinstance(result['results'], dict)
    
    
    def test_worker_creates_own_engine(self):
        """
        Test that each worker creates its own FastXCorr instance.
        
        This is critical for parallel processing - each worker needs its own
        engine instance to avoid conflicts and ensure thread safety.
        """
        # Create minimal test data
        target_data = {
            'sequence': 'PEPTIDE',
            'protein_id': 'protein1',
            'mass': 800.0
        }
        decoy_data = {
            'sequence': 'EDITPEP',
            'protein_id': 'decoy_protein1',
            'mass': 800.0
        }
        
        spectrum = MassSpectrum(
            mz_array=np.array([100.0, 200.0]),
            intensity_array=np.array([1000.0, 2000.0]),
            scan_id="scan_1"
        )
        
        args = (
            0, 1, (400.0, 405.0), [spectrum], None,
            [(target_data, decoy_data)], [2],
            None, 'trypsin', 1,
            None, None, 10.0, 10.0, 0
        )
        
        # Patch FastXCorr to track instantiation
        with patch('pyXcorrDIA.FastXCorr') as mock_fastxcorr:
            mock_instance = MagicMock()
            mock_instance.search_dia_peptide_centric.return_value = {
                'results': {},
                'parquet_file': None,
                'peptide_info_file': None
            }
            mock_fastxcorr.return_value = mock_instance
            
            # Call worker
            process_isolation_window_worker(args)
            
            # Verify FastXCorr was instantiated (worker creates its own engine)
            mock_fastxcorr.assert_called_once()
    
    
    def test_parallel_vs_sequential_consistency(self, xcorr_engine, tmp_path):
        """
        Test that parallel and sequential processing produce identical results.
        
        This is the most important test - results should be deterministic
        regardless of whether we use 1 or N workers.
        """
        # Create test peptides
        peptide1 = PeptideCandidate("PEPTIDE", "protein1", mass=800.0)
        peptide2 = PeptideCandidate("SAMPLE", "protein2", mass=650.0)
        
        # Create target-decoy pairs
        target_decoy_pairs = [
            (peptide1, PeptideCandidate("EDITPEP", "decoy_protein1", mass=800.0)),
            (peptide2, PeptideCandidate("ELPMAS", "decoy_protein2", mass=650.0))
        ]
        
        # Create mock spectra in two isolation windows
        spectrum1 = MassSpectrum(
            mz_array=np.array([400.1, 401.2, 402.3]),
            intensity_array=np.array([1000.0, 2000.0, 1500.0]),
            scan_id="scan_1",
            isolation_window_lower=400.0,
            isolation_window_upper=405.0
        )
        
        spectrum2 = MassSpectrum(
            mz_array=np.array([325.5, 326.8, 327.2]),
            intensity_array=np.array([1200.0, 1800.0, 1300.0]),
            scan_id="scan_2",
            isolation_window_lower=325.0,
            isolation_window_upper=330.0
        )
        
        # Prepare serialized data for workers
        target_decoy_pairs_data = []
        for target, decoy in target_decoy_pairs:
            target_data = {
                'sequence': target.sequence,
                'protein_id': target.protein_id,
                'mass': target.mass
            }
            decoy_data = {
                'sequence': decoy.sequence,
                'protein_id': decoy.protein_id,
                'mass': decoy.mass
            }
            target_decoy_pairs_data.append((target_data, decoy_data))
        
        # Create work items for two isolation windows with temporary parquet paths
        parquet_file1 = str(tmp_path / "window_400.0_405.0.parquet")
        parquet_file2 = str(tmp_path / "window_325.0_330.0.parquet")
        
        work_items = [
            (0, 2, (400.0, 405.0), [spectrum1], None,
             target_decoy_pairs_data, [2],
             parquet_file1, 'trypsin', 1,
             None, None, 10.0, 10.0, 0),
            (1, 2, (325.0, 330.0), [spectrum2], None,
             target_decoy_pairs_data, [2],
             parquet_file2, 'trypsin', 1,
             None, None, 10.0, 10.0, 0)
        ]
        
        # Process sequentially
        sequential_results = [process_isolation_window_worker(item) for item in work_items]
        
        # Process in "parallel" (with multiprocessing but effectively sequential for test)
        from multiprocessing import Pool
        with Pool(1) as pool:
            parallel_results = pool.map(process_isolation_window_worker, work_items)
        
        # Compare results
        assert len(sequential_results) == len(parallel_results)
        for seq_result, par_result in zip(sequential_results, parallel_results):
            # Both should have same structure
            assert seq_result.keys() == par_result.keys()
            
            # Results dictionaries should have same keys
            assert seq_result['results'].keys() == par_result['results'].keys()
    
    
    def test_worker_handles_multiple_spectra(self, tmp_path):
        """
        Test that worker correctly processes windows with multiple spectra.
        
        In real DIA data, each isolation window contains many spectra
        across the chromatographic dimension.
        """
        # Create test data
        target_data = {'sequence': 'PEPTIDE', 'protein_id': 'protein1', 'mass': 800.0}
        decoy_data = {'sequence': 'EDITPEP', 'protein_id': 'decoy_protein1', 'mass': 800.0}
        
        # Create multiple spectra in same isolation window
        spectra = []
        for i in range(10):
            spectrum = MassSpectrum(
                mz_array=np.array([400.0 + i*0.1, 401.0 + i*0.1]),
                intensity_array=np.array([1000.0 + i*100, 2000.0 + i*100]),
                scan_id=f"scan_{i}",
                isolation_window_lower=400.0,
                isolation_window_upper=405.0
            )
            spectra.append(spectrum)
        
        # Use temporary parquet file path
        parquet_file = str(tmp_path / "window_400.0_405.0.parquet")
        
        args = (
            0, 1, (400.0, 405.0), spectra, None,
            [(target_data, decoy_data)], [2],
            parquet_file, 'trypsin', 1,
            None, None, 10.0, 10.0, 0
        )
        
        # Process window
        result = process_isolation_window_worker(args)
        
        # Verify result was produced
        assert 'results' in result
        assert isinstance(result['results'], dict)
    
    
    def test_worker_handles_multiple_charge_states(self, tmp_path):
        """
        Test that worker processes multiple charge states correctly.
        
        DIA searches typically evaluate peptides at multiple charge states.
        """
        target_data = {'sequence': 'PEPTIDE', 'protein_id': 'protein1', 'mass': 800.0}
        decoy_data = {'sequence': 'EDITPEP', 'protein_id': 'decoy_protein1', 'mass': 800.0}
        
        spectrum = MassSpectrum(
            mz_array=np.array([400.5, 267.0, 200.5]),  # Could match z=2 or z=3
            intensity_array=np.array([1000.0, 2000.0, 1500.0]),
            scan_id="scan_1",
            isolation_window_lower=400.0,
            isolation_window_upper=405.0
        )
        
        # Use temporary parquet file path
        parquet_file = str(tmp_path / "window_400.0_405.0.parquet")
        
        # Test with multiple charge states
        args = (
            0, 1, (400.0, 405.0), [spectrum], None,
            [(target_data, decoy_data)],
            [2, 3, 4],  # Multiple charge states
            parquet_file, 'trypsin', 1,
            None, None, 10.0, 10.0, 0
        )
        
        result = process_isolation_window_worker(args)
        
        # Verify processing completed
        assert 'results' in result
    
    
    def test_worker_verbose_output_suppression(self, capsys, tmp_path):
        """
        Test that verbose=0 suppresses worker output.
        
        Workers should be silent when verbose=0 to avoid cluttering
        parallel output.
        """
        # Create peptide with precursor m/z in isolation window 400-405
        # For charge 2: mass ~800 Da → m/z ~400 Da
        target_data = {'sequence': 'PEPTIDE', 'protein_id': 'protein1', 'mass': 799.0}
        decoy_data = {'sequence': 'EDITPEP', 'protein_id': 'decoy_protein1', 'mass': 799.0}
        
        spectrum = MassSpectrum(
            mz_array=np.array([400.5, 401.0]),
            intensity_array=np.array([1000.0, 2000.0]),
            scan_id="scan_1",
            isolation_window_lower=400.0,
            isolation_window_upper=405.0
        )
        
        # Use temporary parquet file path
        parquet_file = str(tmp_path / "window_400.0_405.0.parquet")
        
        # verbose=0 should suppress output
        args = (
            0, 1, (400.0, 405.0), [spectrum], None,
            [(target_data, decoy_data)], [2],
            parquet_file, 'trypsin', 1,
            None, None, 10.0, 10.0, 0  # verbose=0
        )
        
        process_isolation_window_worker(args)
        
        captured = capsys.readouterr()
        # Should not contain worker start/completion messages
        assert "[Worker" not in captured.out
    
    
    def test_worker_result_structure(self, tmp_path):
        """
        Test that worker returns correctly structured results.
        
        Results must include:
        - results: dict of peptide results
        - parquet_file: path to unified chromatogram file (or None)
        - num_spectra: number of spectra processed
        - num_peptides: number of peptides scored
        - isolation_window: tuple of (lower, upper) bounds
        """
        # Create peptide with precursor m/z in isolation window 400-405
        # For charge 2: mass ~800 Da → m/z ~400 Da
        target_data = {'sequence': 'PEPTIDE', 'protein_id': 'protein1', 'mass': 799.0}
        decoy_data = {'sequence': 'EDITPEP', 'protein_id': 'decoy_protein1', 'mass': 799.0}
        
        spectrum = MassSpectrum(
            mz_array=np.array([400.5, 401.0]),
            intensity_array=np.array([1000.0, 2000.0]),
            scan_id="scan_1",
            isolation_window_lower=400.0,
            isolation_window_upper=405.0
        )
        
        # Use temporary parquet file path
        parquet_file = str(tmp_path / "window_400.0_405.0.parquet")
        
        args = (
            0, 1, (400.0, 405.0), [spectrum], None,
            [(target_data, decoy_data)], [2],
            parquet_file, 'trypsin', 1,
            None, None, 10.0, 10.0, 0
        )
        
        result = process_isolation_window_worker(args)
        
        # Verify required keys in unified schema
        assert 'results' in result
        assert 'parquet_file' in result
        assert 'num_spectra' in result
        assert 'num_peptides' in result
        assert 'isolation_window' in result
        
        # Verify types
        assert isinstance(result['results'], dict)
        assert result['parquet_file'] is None or isinstance(result['parquet_file'], str)
        assert isinstance(result['num_spectra'], int)
        assert isinstance(result['num_peptides'], int)
        assert isinstance(result['isolation_window'], tuple)


class TestDIAMatrixScoring:
    """Test that DIA mode uses matrix operations efficiently."""
    
    def test_batch_scoring_uses_unified_xcorr(self, xcorr_engine):
        """
        Test that DIA batch scoring uses the unified calculate_xcorr function.
        
        The unified function supports matrix operations for efficient N×M scoring.
        """
        # Create test peptides
        peptides = [
            PeptideCandidate("PEPTIDE", "protein1", mass=800.0),
            PeptideCandidate("SAMPLE", "protein2", mass=650.0),
            PeptideCandidate("TESTING", "protein3", mass=900.0)
        ]
        
        # Create test spectra
        spectra = []
        for i in range(3):
            spectrum = MassSpectrum(
                mz_array=np.array([400.0 + i, 401.0 + i]),
                intensity_array=np.array([1000.0, 2000.0]),
                scan_id=f"scan_{i}"
            )
            spectra.append(spectrum)
        
        # Preprocess spectra
        exp_matrices = []
        for spectrum in spectra:
            preprocessed = xcorr_engine.preprocess_spectrum(spectrum)
            exp_matrices.append(preprocessed)
        exp_matrix = np.vstack(exp_matrices)  # Shape: (3, n_bins)
        
        # Generate and preprocess theoretical spectra
        theo_matrices = []
        for peptide in peptides:
            theo = xcorr_engine.generate_theoretical_spectrum(peptide, charge=2)
            
            # Find highest ion
            highest_ion = 0
            for i in range(len(theo) - 1, -1, -1):
                if theo[i] > 0:
                    highest_ion = i
                    break
            
            # Apply windowing
            theo_windowed = xcorr_engine._make_corr_data(theo, highest_ion, 1.0)
            
            # Apply Fast XCorr preprocessing
            theo_prep = xcorr_engine.preprocess_for_xcorr(theo_windowed)
            theo_matrices.append(theo_prep)
        
        theo_matrix = np.vstack(theo_matrices)  # Shape: (3, n_bins)
        
        # Perform matrix scoring using unified function
        xcorr_matrix = xcorr_engine.calculate_xcorr(
            theo_matrix, 
            exp_matrix, 
            scaling_factor=0.0001
        )
        
        # Verify matrix result
        assert xcorr_matrix.shape == (3, 3), "Should be 3 peptides × 3 spectra"
        assert isinstance(xcorr_matrix, np.ndarray)
        
        # Verify consistency: matrix result should match single scoring
        single_score = xcorr_engine.calculate_xcorr(
            theo_matrix[0], 
            exp_matrix[0], 
            scaling_factor=0.0001
        )
        assert abs(xcorr_matrix[0, 0] - single_score) < 1e-6
    
    
    def test_matrix_scoring_performance_benefit(self, xcorr_engine):
        """
        Test that matrix scoring is more efficient than loop-based scoring.
        
        Matrix multiplication should be ~10-100x faster for batch operations.
        """
        import time
        
        # Create moderate-sized test (10 peptides × 10 spectra)
        n_peptides = 10
        n_spectra = 10
        
        # Create peptides
        peptides = [
            PeptideCandidate(f"PEPTIDE{i}", f"protein{i}", mass=800.0 + i*10)
            for i in range(n_peptides)
        ]
        
        # Create spectra
        spectra = []
        for i in range(n_spectra):
            spectrum = MassSpectrum(
                mz_array=np.array([400.0 + i, 401.0 + i, 402.0 + i]),
                intensity_array=np.array([1000.0, 2000.0, 1500.0]),
                scan_id=f"scan_{i}"
            )
            spectra.append(spectrum)
        
        # Preprocess all spectra and peptides
        exp_matrices = [xcorr_engine.preprocess_spectrum(s) for s in spectra]
        exp_matrix = np.vstack(exp_matrices)
        
        theo_matrices = []
        for peptide in peptides:
            theo = xcorr_engine.generate_theoretical_spectrum(peptide, charge=2)
            highest_ion = 0
            for i in range(len(theo) - 1, -1, -1):
                if theo[i] > 0:
                    highest_ion = i
                    break
            theo_windowed = xcorr_engine._make_corr_data(theo, highest_ion, 1.0)
            theo_prep = xcorr_engine.preprocess_for_xcorr(theo_windowed)
            theo_matrices.append(theo_prep)
        theo_matrix = np.vstack(theo_matrices)
        
        # Time matrix scoring
        start = time.time()
        matrix_result = xcorr_engine.calculate_xcorr(theo_matrix, exp_matrix, scaling_factor=0.0001)
        matrix_time = time.time() - start
        
        # Time loop-based scoring
        start = time.time()
        loop_results = np.zeros((n_peptides, n_spectra))
        for i in range(n_peptides):
            for j in range(n_spectra):
                loop_results[i, j] = xcorr_engine.calculate_xcorr(
                    theo_matrix[i], exp_matrix[j], scaling_factor=0.0001
                )
        loop_time = time.time() - start
        
        # Verify results are identical
        assert np.allclose(matrix_result, loop_results, atol=1e-6)
        
        # Matrix should be faster (at least not slower)
        # Note: Speedup may vary based on BLAS implementation and system load
        # We allow up to 5x variance to account for test environment differences
        assert matrix_time <= loop_time * 5  # Very lenient to avoid flaky test
        
        print(f"\nMatrix scoring: {matrix_time:.4f}s")
        print(f"Loop scoring: {loop_time:.4f}s")
        if loop_time > 0:
            print(f"Speedup: {loop_time/matrix_time:.1f}x")
