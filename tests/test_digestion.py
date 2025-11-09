"""
Test protein digestion and peptide generation functionality.
"""
import pytest
from pyXcorrDIA import FastXCorr


class TestProteinDigestion:
    """Test protein digestion into peptides."""
    
    def test_trypsin_digestion_simple(self, xcorr_engine):
        """Test basic trypsin digestion."""
        sequence = "YQSHTKDEFGHR"
        protein_id = "test_protein"
        
        peptides = xcorr_engine.digest_protein(sequence, protein_id, 
                                              enzyme='trypsin', 
                                              missed_cleavages=0)
        
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
        sequence = "AAAKBBBRBBBKCCC"
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
        sequence = "AAAKBBBKCCCKDDD"
        
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
                                                  missed_cleavages=2)
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
