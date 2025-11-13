"""
Test protein digestion and peptide generation functionality.
"""


class TestProteinDigestion:
    """Test protein digestion into peptides."""
    
    def test_trypsin_digestion_simple(self, xcorr_engine):
        """Test basic trypsin digestion."""
        sequence = "YQSHTKDEFGHR"
        protein_id = "test_protein"
        
        peptides = xcorr_engine.digest_protein(sequence, protein_id, 
                                              enzyme='trypsin', 
                                              missed_cleavages=0,
                                              min_length=6, max_length=30)
        
        assert len(peptides) > 0
        
        # Check peptide properties
        for peptide in peptides:
            assert hasattr(peptide, 'sequence')
            assert hasattr(peptide, 'protein_id')
            assert hasattr(peptide, 'mass')
            assert peptide.protein_id == protein_id
            assert peptide.mass > 0
    
    def test_trypsin_cleavage_sites(self, xcorr_engine):
        """Test that trypsin cleaves at K and R."""
        sequence = "AAAKGGGRGGGKCCC"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='trypsin',
                                              missed_cleavages=0)
        
        # Should get peptides ending in K or R (or at protein terminus)
        for peptide in peptides:
            if peptide.sequence != sequence:  # Not full protein
                last_aa = peptide.sequence[-1]
                # Last amino acid should be K or R for tryptic peptides
                # (except for C-terminal peptide)
                if peptide.sequence != sequence.split('K')[-1].split('R')[-1]:
                    assert last_aa in ['K', 'R'], f"Peptide {peptide.sequence} doesn't end in K/R"
    
    def test_missed_cleavages(self, xcorr_engine):
        """Test missed cleavage handling."""
        sequence = "AAAKGGGKCCCKDDD"
        
        # No missed cleavages
        peptides_0 = xcorr_engine.digest_protein(sequence, "test",
                                                enzyme='trypsin',
                                                missed_cleavages=0)
        
        # One missed cleavage
        peptides_1 = xcorr_engine.digest_protein(sequence, "test",
                                                enzyme='trypsin',
                                                missed_cleavages=1)
        
        # Two missed cleavages
        peptides_2 = xcorr_engine.digest_protein(sequence, "test",
                                                enzyme='trypsin',
                                                missed_cleavages=2)
        
        # More missed cleavages should generate more peptides
        assert len(peptides_1) >= len(peptides_0)
        assert len(peptides_2) >= len(peptides_1)
    
    def test_digest_yqshtk_sequence(self, xcorr_engine, yqshtk_fasta):
        """Test digestion of YQSHTK FASTA file."""
        proteins = xcorr_engine.read_fasta(yqshtk_fasta)
        
        all_peptides = []
        for protein_id, sequence in proteins.items():
            peptides = xcorr_engine.digest_protein(sequence, protein_id,
                                                  enzyme='trypsin',
                                                  missed_cleavages=2,
                                                  min_length=6,  # YQSHTK is 6 amino acids
                                                  max_length=50)
            all_peptides.extend(peptides)
        
        assert len(all_peptides) > 0
        print(f"Generated {len(all_peptides)} peptides from YQSHTK FASTA")
        
        # Check for YQSHTK peptide
        peptide_sequences = [p.sequence for p in all_peptides]
        assert "YQSHTK" in peptide_sequences or any("YQSHTK" in seq for seq in peptide_sequences)


class TestDecoyGeneration:
    """Test decoy peptide generation."""
    
    def test_cycled_decoy_simple(self, xcorr_engine):
        """Test basic decoy generation by cycling."""
        sequence = "YQSHTK"
        decoy = xcorr_engine.generate_decoy_sequence(sequence, cycle_length=1)
        
        assert len(decoy) == len(sequence)
        assert decoy != sequence
        assert decoy[-1] == 'K'  # K should stay at C-terminus
    
    def test_reversed_decoy_simple(self, xcorr_engine):
        """Test basic decoy generation by reversal."""
        sequence = "YQSHTK"
        decoy = xcorr_engine.generate_reversed_decoy_sequence(sequence)
        
        assert len(decoy) == len(sequence)
        assert decoy != sequence
        assert decoy[-1] == 'K'  # K should stay at C-terminus
        
        # Check that sequence is reversed (except last K)
        assert decoy == "THSQYK"
    
    def test_decoy_keeps_terminal_residue(self, xcorr_engine):
        """Test that decoys preserve C-terminal K/R."""
        sequences = ["PEPTIDEK", "PEPTIDER", "PEPTIDE"]
        
        for seq in sequences:
            decoy = xcorr_engine.generate_decoy_sequence(seq, cycle_length=1)
            
            if seq[-1] in ['K', 'R']:
                assert decoy[-1] == seq[-1], f"Terminal {seq[-1]} not preserved in {decoy}"
    
    def test_target_decoy_pairs(self, xcorr_engine, sample_peptide):
        """Test target-decoy pair generation."""
        target_peptides = [sample_peptide]
        
        pairs = xcorr_engine.generate_target_decoy_pairs(target_peptides, cycle_length=1)
        
        assert len(pairs) <= len(target_peptides)
        
        for target, decoy in pairs:
            assert target.sequence != decoy.sequence
            assert len(target.sequence) == len(decoy.sequence)
            # Mass should be the same (same amino acid composition)
            assert abs(target.mass - decoy.mass) < 0.01


class TestPeptideNonRedundancy:
    """Test making peptide lists non-redundant."""
    
    def test_remove_duplicates(self, xcorr_engine):
        """Test that duplicate peptides are properly handled."""
        from pyXcorrDIA import PeptideCandidate
        
        # Create duplicate peptides from different proteins
        peptides = [
            PeptideCandidate("PEPTIDE", "protein1", 100.0),
            PeptideCandidate("PEPTIDE", "protein2", 100.0),
            PeptideCandidate("DIFFERENT", "protein1", 150.0),
        ]
        
        non_redundant = xcorr_engine.make_peptides_non_redundant(peptides)
        
        # Should have only 2 unique sequences
        assert len(non_redundant) == 2
        
        # Check that protein IDs are concatenated
        sequences = {p.sequence: p for p in non_redundant}
        assert "PEPTIDE" in sequences
        assert "DIFFERENT" in sequences
        
        # The PEPTIDE entry should have both protein IDs
        peptide_entry = sequences["PEPTIDE"]
        assert "protein1" in peptide_entry.protein_id
        assert "protein2" in peptide_entry.protein_id


class TestEnzymeSupport:
    """Test all supported enzyme digestion patterns."""
    
    def test_trypsin_digestion(self, xcorr_engine):
        """Test trypsin (cleaves after K, R but NOT before P - with proline suppression)."""
        # Longer sequence with peptides that pass length filter (6-50 aa)
        sequence = "AAAAAAKGGGGGGGRCCCCCCCCKDDDDDDKPEEEEEE"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='trypsin',
                                              missed_cleavages=0)
        
        # Should cleave after K and R, but not before P
        sequences = [p.sequence for p in peptides]
        assert len(sequences) > 0, "Should generate peptides"
        # Most peptides should end at K or R (except last one and those before P)
        has_tryptic = any(seq[-1] in ['K', 'R'] for seq in sequences[:-1])
        assert has_tryptic, "Should have tryptic peptides ending in K or R"
        # Should NOT cleave before P - check that KP stays together in peptide
        kp_combined = any('KP' in seq for seq in sequences)
        assert kp_combined, "KP should stay together (no cleavage before P)"
    
    def test_trypsin_no_proline_digestion(self, xcorr_engine):
        """Test trypsin_no_proline (cleaves after K, R including before P)."""
        # Longer sequence with peptides that pass length filter
        sequence = "AAAAAAKGGGGGGGRCCCCCCCKKDDDDDDKPEEEEEE"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='trypsin_no_proline',
                                              missed_cleavages=0)
        
        # Should cleave after K and R, even if followed by P
        sequences = [p.sequence for p in peptides]
        # Check that we get peptides
        assert len(sequences) > 0, "Should generate peptides"
        # All peptides except last should end at K or R
        for seq in sequences[:-1]:  # All except last peptide
            assert seq[-1] in ['K', 'R'], f"Peptide {seq} should end with K or R"
    
    def test_lysc_digestion(self, xcorr_engine):
        """Test Lys-C (cleaves after K)."""
        sequence = "AAKGGGKCCCRDDD"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='lysc',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with K
        for seq in sequences[:-1]:
            assert seq[-1] == 'K', f"Lys-C peptide {seq} should end with K"
    
    def test_lysn_digestion(self, xcorr_engine):
        """Test Lys-N (cleaves before K)."""
        sequence = "AAAGGGKCCCRDDD"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='lysn',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except first should start with K
        for seq in sequences[1:]:
            assert seq[0] == 'K', f"Lys-N peptide {seq} should start with K"
    
    def test_argc_digestion(self, xcorr_engine):
        """Test Arg-C (cleaves after R)."""
        sequence = "AAARGGGRCCCRDDD"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='argc',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with R
        for seq in sequences[:-1]:
            assert seq[-1] == 'R', f"Arg-C peptide {seq} should end with R"
    
    def test_aspn_digestion(self, xcorr_engine):
        """Test Asp-N (cleaves before D)."""
        sequence = "AAAGGGDCCCDEEEF"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='aspn',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except first should start with D
        for seq in sequences[1:]:
            assert seq[0] == 'D', f"Asp-N peptide {seq} should start with D"
    
    def test_cnbr_digestion(self, xcorr_engine):
        """Test CNBr (cleaves after M)."""
        sequence = "AAAMGGGMCCCDDD"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='cnbr',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with M
        for seq in sequences[:-1]:
            assert seq[-1] == 'M', f"CNBr peptide {seq} should end with M"
    
    def test_gluc_digestion(self, xcorr_engine):
        """Test Glu-C (cleaves after D, E)."""
        sequence = "AAADGGGEFFFGGGD"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='gluc',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with D or E
        for seq in sequences[:-1]:
            assert seq[-1] in ['D', 'E'], f"Glu-C peptide {seq} should end with D or E"
    
    def test_pepsina_digestion(self, xcorr_engine):
        """Test Pepsin A (cleaves after F, L)."""
        sequence = "AAAFGGGFLLLCCC"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='pepsina',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with F or L
        for seq in sequences[:-1]:
            assert seq[-1] in ['F', 'L'], f"Pepsin A peptide {seq} should end with F or L"
    
    def test_chymotrypsin_digestion(self, xcorr_engine):
        """Test Chymotrypsin (cleaves after F, W, Y, L)."""
        sequence = "AAAFGGGWCCCYDDDL"
        peptides = xcorr_engine.digest_protein(sequence, "test", 
                                              enzyme='chymotrypsin',
                                              missed_cleavages=0)
        
        sequences = [p.sequence for p in peptides]
        # All peptides except last should end with F, W, Y, or L
        for seq in sequences[:-1]:
            assert seq[-1] in ['F', 'W', 'Y', 'L'], \
                f"Chymotrypsin peptide {seq} should end with F, W, Y, or L"
    
    def test_enzyme_decoy_preservation_c_terminal(self, xcorr_engine):
        """Test that C-terminal enzymes preserve C-terminal residue in decoys."""
        # Use appropriate terminal residues for each enzyme
        test_cases = [
            ('trypsin', 'PEPTIDEK'),
            ('trypsin_no_proline', 'PEPTIDER'),
            ('lysc', 'PEPTIDEK'),
            ('argc', 'PEPTIDER'),  # R for Arg-C
            ('cnbr', 'PEPTIDEM'),  # M for CNBr
            ('gluc', 'PEPTIDED'),  # D for Glu-C
            ('pepsina', 'PEPTIDEF'),  # F for Pepsin A
            ('chymotrypsin', 'PEPTIDEY'),  # Y for Chymotrypsin
        ]
        
        for enzyme, sequence in test_cases:
            decoy = xcorr_engine.generate_decoy_sequence(sequence, enzyme=enzyme)
            expected_terminal = sequence[-1]
            assert decoy[-1] == expected_terminal, \
                f"Enzyme {enzyme} should preserve C-terminal {expected_terminal} in decoy, got {decoy[-1]}"
    
    def test_enzyme_decoy_preservation_n_terminal(self, xcorr_engine):
        """Test that N-terminal enzymes preserve N-terminal residue in decoys."""
        # Use appropriate cleavage residues for each enzyme
        test_cases = [
            ('lysn', 'KPEPTIDE'),  # K at N-terminus for Lys-N
            ('aspn', 'DPEPTIDE'),  # D at N-terminus for Asp-N
        ]
        
        for enzyme, sequence in test_cases:
            decoy = xcorr_engine.generate_decoy_sequence(sequence, enzyme=enzyme)
            assert decoy[0] == sequence[0], \
                f"Enzyme {enzyme} should preserve N-terminal {sequence[0]} in decoy"
    
    def test_enzyme_reversed_decoy_c_terminal(self, xcorr_engine):
        """Test that C-terminal enzymes preserve C-terminal in reversed decoys."""
        # Use YQSHTK as it ends with K (good for trypsin, lysc)
        # Use YQSHTR for argc (R-specific)
        test_cases = [
            ('trypsin', 'YQSHTK', 'THSQYK'),
            ('trypsin_no_proline', 'YQSHTK', 'THSQYK'),
            ('lysc', 'YQSHTK', 'THSQYK'),
            ('argc', 'YQSHTR', 'THSQYR'),  # R for Arg-C
            ('cnbr', 'YQSHTM', 'THSQYM'),  # M for CNBr
            ('gluc', 'YQSHTD', 'THSQYD'),  # D for Glu-C
            ('pepsina', 'YQSHTF', 'THSQYF'),  # F for Pepsin A
            ('chymotrypsin', 'YQSHTW', 'THSQYW'),  # W for Chymotrypsin
        ]
        
        for enzyme, sequence, expected in test_cases:
            decoy = xcorr_engine.generate_reversed_decoy_sequence(sequence, enzyme=enzyme)
            expected_terminal = sequence[-1]
            assert decoy[-1] == expected_terminal, \
                f"Enzyme {enzyme} should preserve C-terminal {expected_terminal} in reversed decoy"
            # Check expected reversal
            assert decoy == expected, \
                f"Enzyme {enzyme} reversed decoy should be {expected}, got {decoy}"
    
    def test_enzyme_reversed_decoy_n_terminal(self, xcorr_engine):
        """Test that N-terminal enzymes preserve N-terminal in reversed decoys."""
        # Use appropriate cleavage residues for each enzyme
        test_cases = [
            ('lysn', 'KTHSQY', 'KYQSHT'),  # K at N-terminus for Lys-N
            ('aspn', 'DTHSQY', 'DYQSHT'),  # D at N-terminus for Asp-N
        ]
        
        for enzyme, sequence, expected in test_cases:
            decoy = xcorr_engine.generate_reversed_decoy_sequence(sequence, enzyme=enzyme)
            assert decoy[0] == sequence[0], \
                f"Enzyme {enzyme} should preserve N-terminal {sequence[0]} in reversed decoy"
            # Check reversal of internal sequence
            assert decoy == expected, \
                f"Enzyme {enzyme} reversed decoy should be {expected}, got {decoy}"
    
    def test_all_enzymes_generate_peptides(self, xcorr_engine):
        """Test that all enzymes can successfully digest a protein."""
        # Protein with all relevant amino acids
        sequence = "AAKGGRCCDEFGLMWYKDFL"
        
        all_enzymes = ['trypsin', 'trypsin_no_proline', 'lysc', 'lysn', 'argc', 
                      'aspn', 'cnbr', 'gluc', 'pepsina', 'chymotrypsin']
        
        for enzyme in all_enzymes:
            peptides = xcorr_engine.digest_protein(sequence, "test", 
                                                  enzyme=enzyme,
                                                  missed_cleavages=0)
            assert len(peptides) > 0, \
                f"Enzyme {enzyme} should generate at least one peptide"
            
            # Check that all peptides have valid masses
            for peptide in peptides:
                assert peptide.mass > 0, \
                    f"Enzyme {enzyme} generated peptide with invalid mass"
