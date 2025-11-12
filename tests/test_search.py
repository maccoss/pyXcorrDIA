"""
Test end-to-end database search functionality.
"""
import pytest


class TestDatabaseSearch:
    """Test complete database search workflow."""
    
    def test_human_proteome_search(self, xcorr_engine_with_mods, large_fasta, ot_centroid_mgf):
        """Test complete search on human proteome data with HGKPTDSTPATWK peptide."""
        # Read FASTA
        proteins = xcorr_engine_with_mods.read_fasta(large_fasta)
        assert len(proteins) > 0
        
        # Digest proteins (using default 7-30 amino acid range)
        all_peptides = []
        for protein_id, sequence in proteins.items():
            peptides = xcorr_engine_with_mods.digest_protein(
                sequence, protein_id, enzyme='trypsin', missed_cleavages=2,
                min_length=7, max_length=30
            )
            all_peptides.extend(peptides)
        
        assert len(all_peptides) > 0
        print(f"Generated {len(all_peptides)} peptides")
        
        # Verify target peptide HGKPTDSTPATWK is present
        target_sequences = [p.sequence for p in all_peptides]
        assert "HGKPTDSTPATWK" in target_sequences, "Target peptide HGKPTDSTPATWK not found in digestion"
        
        # Make non-redundant
        non_redundant = xcorr_engine_with_mods.make_peptides_non_redundant(all_peptides)
        print(f"Non-redundant peptides: {len(non_redundant)}")
        
        # Generate decoys
        target_decoy_pairs = xcorr_engine_with_mods.generate_target_decoy_pairs(
            non_redundant, cycle_length=1
        )
        assert len(target_decoy_pairs) > 0
        print(f"Target-decoy pairs: {len(target_decoy_pairs)}")
        
        # Read spectra from MGF
        spectra = xcorr_engine_with_mods.read_mgf(ot_centroid_mgf)
        assert len(spectra) > 0
        print(f"Read {len(spectra)} spectra")
        
        # Search first spectrum (scan 8340 with HGKPTDSTPATWK)
        if len(spectra) > 0:
            spectrum = spectra[0]
            # Test typical charge states for tryptic peptides
            charge_states = [2, 3]
            
            results = xcorr_engine_with_mods.search_spectrum_target_decoy(
                spectrum, target_decoy_pairs, charge_states
            )
            
            assert results is not None
            assert len(results) > 0
            
            # Check result structure
            for peptide, xcorr, e_value, charge in results:
                assert hasattr(peptide, 'sequence')
                assert isinstance(xcorr, float)
                assert isinstance(e_value, (float, int))
                assert charge in charge_states
            
            # Separate results by charge state
            results_by_charge = {2: [], 3: []}
            for peptide, xcorr, e_value, charge in results:
                if charge in results_by_charge:
                    results_by_charge[charge].append((peptide.sequence, xcorr, e_value))
            
            # Expected top 3 results for charge +2 (these are the reference values)
            print("\nTop 3 results for charge +2:")
            expected_charge2 = [
                ("SLTSSGGGKR", 1.7272, 8.40e-01),
                ("ESSATSSQR", 1.6688, 1.28e+00),
                ("NSVSSSSQR", 1.5252, 3.56e+00),
            ]
            for i, (seq, xcorr, eval) in enumerate(results_by_charge[2][:3]):
                print(f"  {i+1}. {seq} (XCorr={xcorr:.4f}, e-value={eval:.2e})")
                # Check against expected (allow small tolerance for XCorr)
                exp_seq, exp_xcorr, exp_eval = expected_charge2[i]
                assert seq == exp_seq, f"Charge +2 rank {i+1}: expected {exp_seq}, got {seq}"
                assert abs(xcorr - exp_xcorr) < 0.01, f"Charge +2 {seq}: expected XCorr {exp_xcorr:.4f}, got {xcorr:.4f}"
            
            # Expected top result for charge +3 (HGKPTDSTPATWK should be top hit)
            print("\nTop 3 results for charge +3:")
            expected_charge3 = [
                ("HGKPTDSTPATWK", 2.7248, 2.62e-02),
            ]
            for i, (seq, xcorr, eval) in enumerate(results_by_charge[3][:3]):
                print(f"  {i+1}. {seq} (XCorr={xcorr:.4f}, e-value={eval:.2e})")
            
            # Verify HGKPTDSTPATWK is the top charge +3 result
            assert len(results_by_charge[3]) > 0, "No charge +3 results found"
            top_charge3 = results_by_charge[3][0]
            exp_seq, exp_xcorr, exp_eval = expected_charge3[0]
            assert top_charge3[0] == exp_seq, f"Charge +3 top hit: expected {exp_seq}, got {top_charge3[0]}"
            assert abs(top_charge3[1] - exp_xcorr) < 0.01, f"Charge +3 {exp_seq}: expected XCorr {exp_xcorr:.4f}, got {top_charge3[1]:.4f}"
    
    def test_search_with_modifications(self, yqshtk_fasta, yqshtk_mzml):
        """Test search with different static modifications."""
        from pyXcorrDIA import FastXCorr
        
        # Create engine with carbamidomethyl-C
        engine = FastXCorr(static_modifications={'C': 57.021464})
        
        # Quick search
        proteins = engine.read_fasta(yqshtk_fasta)
        all_peptides = []
        for protein_id, sequence in proteins.items():
            peptides = engine.digest_protein(sequence, protein_id)
            all_peptides.extend(peptides)
        
        # Check that C-containing peptides have modified mass
        for peptide in all_peptides:
            if 'C' in peptide.sequence:
                # Mass should include modification
                base_mass = sum(engine.base_aa_masses.get(aa, 0) for aa in peptide.sequence)
                base_mass += engine.h2o_mass
                modified_mass = peptide.mass
                
                # Should have added 57.021464 for each C
                c_count = peptide.sequence.count('C')
                expected_diff = c_count * 57.021464
                assert abs((modified_mass - base_mass) - expected_diff) < 0.01
                break  # Just check one


class TestPeptideIndexing:
    """Test peptide indexing for fast isolation window lookup."""
    
    def test_build_peptide_index(self, xcorr_engine, sample_peptide):
        """Test building peptide m/z index."""
        charge_states = [2, 3]
        
        xcorr_engine.build_peptide_index([sample_peptide], charge_states)
        
        # Check that index was created
        assert len(xcorr_engine.sorted_peptides_by_mz) > 0
        
        for charge in charge_states:
            assert charge in xcorr_engine.sorted_peptides_by_mz
    
    def test_find_peptides_in_window(self, xcorr_engine, sample_peptide):
        """Test finding peptides in isolation window."""
        charge_states = [2, 3]
        
        # Build index
        xcorr_engine.build_peptide_index([sample_peptide], charge_states)
        
        # Calculate expected m/z for charge 2
        mz_charge2 = (sample_peptide.mass + 2 * xcorr_engine.proton_mass) / 2
        
        # Search with wide window around this m/z
        window_lower = mz_charge2 - 5.0
        window_upper = mz_charge2 + 5.0
        
        found = xcorr_engine.find_peptides_in_isolation_window(
            window_lower, window_upper, charge_states
        )
        
        assert len(found) > 0
        
        # Should find our peptide
        found_sequences = [p.sequence for p, c in found]
        assert sample_peptide.sequence in found_sequences


class TestEValueCalculation:
    """Test E-value calculation."""
    
    def test_calculate_e_value(self, xcorr_engine):
        """Test E-value calculation with score distribution."""
        # Create fake score distribution
        scores = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        top_score = 5.0
        
        e_value = xcorr_engine.calculate_e_value(scores, top_score)
        
        assert isinstance(e_value, (float, int))
        assert e_value >= 0
    
    def test_e_value_by_charge(self, xcorr_engine):
        """Test charge-specific E-value calculation."""
        # Create fake score distributions by charge
        score_distributions = {
            2: [1.0, 1.5, 2.0, 2.5, 3.0],
            3: [0.8, 1.2, 1.8, 2.2, 2.8]
        }
        
        xcorr_score = 3.5
        charge = 2
        
        e_value = xcorr_engine.calculate_e_value_by_charge(
            score_distributions, xcorr_score, charge
        )
        
        assert isinstance(e_value, (float, int))
        assert e_value >= 0


class TestIntegrationWorkflow:
    """Test complete integration workflow matching command-line behavior."""
    
    @pytest.mark.slow
    def test_complete_workflow_human_proteome(self, xcorr_engine_with_mods, large_fasta, ot_centroid_mgf):
        """Test complete workflow on human proteome with HGKPTDSTPATWK."""
        print("\n=== Running Complete Workflow Test ===")
        
        # Step 1: Read FASTA
        print("Reading FASTA...")
        proteins = xcorr_engine_with_mods.read_fasta(large_fasta)
        print(f"  Loaded {len(proteins)} proteins")
        
        # Step 2: Digest proteins (using default 7-30 range)
        print("Digesting proteins...")
        all_target_peptides = []
        for protein_id, sequence in proteins.items():
            peptides = xcorr_engine_with_mods.digest_protein(
                sequence, protein_id, enzyme='trypsin', missed_cleavages=2,
                min_length=7, max_length=30
            )
            all_target_peptides.extend(peptides)
        print(f"  Generated {len(all_target_peptides)} target peptides")
        
        # Verify target peptide is present
        target_sequences = [p.sequence for p in all_target_peptides]
        assert "HGKPTDSTPATWK" in target_sequences
        
        # Step 3: Make non-redundant
        print("Making peptide list non-redundant...")
        non_redundant_targets = xcorr_engine_with_mods.make_peptides_non_redundant(
            all_target_peptides
        )
        print(f"  Non-redundant: {len(non_redundant_targets)}")
        
        # Step 4: Generate decoys
        print("Generating target-decoy pairs...")
        target_decoy_pairs = xcorr_engine_with_mods.generate_target_decoy_pairs(
            non_redundant_targets, cycle_length=1
        )
        print(f"  Created {len(target_decoy_pairs)} pairs")
        
        # Step 5: Read spectra
        print("Reading spectra...")
        spectra = xcorr_engine_with_mods.read_mgf(ot_centroid_mgf)
        print(f"  Read {len(spectra)} spectra")
        
        # Step 6: Search spectra
        print("Searching spectra...")
        # Test typical charge states for tryptic peptides
        charge_states = [2, 3]
        total_identifications = 0
        
        for i, spectrum in enumerate(spectra[:3]):  # Test first 3 spectra
            results = xcorr_engine_with_mods.search_spectrum_target_decoy(
                spectrum, target_decoy_pairs, charge_states
            )
            
            if len(results) > 0:
                total_identifications += len(results)
                top_result = results[0]
                print(f"  Spectrum {spectrum.scan_id}: {top_result[0].sequence} "
                      f"(XCorr={top_result[1]:.4f}, charge={top_result[3]})")
        
        print(f"\nTotal identifications: {total_identifications}")
        assert total_identifications > 0
