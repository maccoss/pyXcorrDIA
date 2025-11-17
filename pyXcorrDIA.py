#!/usr/bin/env python3

import warnings
# Silence all warnings from third-party libraries before they're imported
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Specifically silence pytz invalid escape sequence warning
warnings.filterwarnings('ignore', message='invalid escape sequence')

"""
Fast SEQUEST Cross-Correlation Implementation
Based on the paper by Eng et al. (2008): "A Fast SEQUEST Cross Correlation Algorithm"

This implementation calculates the cross-correlation score without FFTs,
enabling scoring of all candidate peptides and E-value calculation.

CORRECTED VERSION - Fixed XCorr calculation to match Comet exactly
"""

import numpy as np
from collections import defaultdict
import re
from typing import List, Tuple, Dict, Optional, Union
import argparse
import pymzml
from pyteomics import mgf
import os
import bisect
import sys
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy import stats
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None  # Will fall back to linear regression if statsmodels not available


class MassSpectrum:
    """Represents a mass spectrum with m/z and intensity values."""
    
    def __init__(self, mz_array: np.ndarray, intensity_array: np.ndarray, 
                 scan_id: str = "", precursor_mz: float = 0.0, charge: int = 0,
                 isolation_window_lower: float = 0.0, isolation_window_upper: float = 0.0,
                 retention_time: float = 0.0):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.scan_id = scan_id
        self.precursor_mz = precursor_mz
        self.charge = charge
        self.isolation_window_lower = isolation_window_lower
        self.isolation_window_upper = isolation_window_upper
        self.retention_time = retention_time  # Retention time in minutes
        # Comet-style preprocessing results
        self.processed_spectrum: Optional[np.ndarray] = None  # After MakeCorrData windowing
        self.preprocessed_spectrum: Optional[np.ndarray] = None  # After fast XCorr preprocessing


class PeptideCandidate:
    """Represents a peptide candidate with its theoretical spectrum."""

    def __init__(self, sequence: str, protein_id: str, mass: float, charge: int = None):
        self.sequence = sequence
        self.protein_id = protein_id
        self.mass = mass
        self.charge = charge  # Optional: used for calibration mode with library sampling
        self.theoretical_spectrum = None


class MS1Spectrum:
    """Represents an MS1 spectrum for precursor isotope scoring."""

    def __init__(self, mz_array: np.ndarray, intensity_array: np.ndarray,
                 scan_id: str = "", retention_time: float = 0.0):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.scan_id = scan_id
        self.retention_time = retention_time  # Retention time in minutes


class SpectrumLibrary:
    """
    Represents a spectral library for library-based scoring in DIA searches.

    Supports DIA-NN predicted spectral libraries in parquet format.
    Provides fragment-level library scoring (cosine angle with SMZ preprocessing)
    and MS1 precursor isotope pattern scoring.
    """

    # UniMod to mass mapping for common modifications
    # https://www.unimod.org/modifications_list.php
    UNIMOD_MASSES = {
        '1': 42.010565,    # Acetyl (N-term or K)
        '4': 57.021464,    # Carbamidomethyl (C)
        '5': 0.984016,     # Carbamyl (N-term, K)
        '7': 0.984016,     # Deamidated (N, Q)
        '21': 79.966331,   # Phospho (S, T, Y)
        '35': 15.994915,   # Oxidation (M)
        '36': 25.980265,   # Dimethyl (K, N-term)
        '42': 0.984016,    # Acetyl (N-term)
        '43': 14.015650,   # Trimethyl (K)
        '121': 114.042927, # GlyGly (K) - ubiquitination
        '259': 14.015650,  # Label:13C(6)15N(2) (K)
        '267': 10.008269,  # Label:13C(6)15N(4) (R)
    }

    def __init__(self, library_path: str = None, verbose: bool = True, test_limit_peptides: int = 0):
        """
        Initialize spectrum library.

        Args:
            library_path: Path to DIA-NN parquet library file
            verbose: Print library loading message (default: True)
            test_limit_peptides: For testing, randomly select N precursors (0 = use all, default: 0)
                               Uses fixed seed (42) for reproducibility
        """
        self.library_path = library_path
        self.library_df = None
        self.peptide_index = {}  # (stripped_sequence, charge) -> precursor_data
        self.decoy_fragments = {}  # (stripped_sequence, charge, is_decoy=True) -> decoy fragment data
        self.verbose = verbose
        self.test_limit_peptides = test_limit_peptides

        if library_path:
            self.load_library(library_path)

    def load_library(self, library_path: str):
        """
        Load DIA-NN parquet library file.

        Expected columns:
        - Precursor.Id, Modified.Sequence, Stripped.Sequence
        - Precursor.Charge, Precursor.Mz
        - Product.Mz, Relative.Intensity
        - Fragment.Type, Fragment.Charge, Fragment.Series.Number
        - RT (predicted retention time)
        - Protein.Ids, Decoy
        """
        import pandas as pd
        try:
            self.library_df = pd.read_parquet(library_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load library from {library_path}. "
                f"Error: {e}\n\n"
                f"Note: This tool requires DIA-NN parquet format libraries (*.parquet), "
                f"not the binary .speclib format. Use 'report-lib.parquet' instead of "
                f"'report-lib.predicted.speclib'."
            )

        # Filter library: exclude decoys and only keep high-confidence entries (Q.Value <= 0.01)
        # Count unique precursors at each step
        initial_precursors = self.library_df.groupby(['Stripped.Sequence', 'Precursor.Charge']).ngroups
        
        # Remove decoys
        self.library_df = self.library_df[self.library_df['Decoy'] == 0]
        after_decoy_precursors = self.library_df.groupby(['Stripped.Sequence', 'Precursor.Charge']).ngroups
        
        # Keep only high-confidence entries with Q.Value <= 0.01
        if 'Q.Value' in self.library_df.columns:
            self.library_df = self.library_df[self.library_df['Q.Value'] <= 0.01]
            after_qvalue_precursors = self.library_df.groupby(['Stripped.Sequence', 'Precursor.Charge']).ngroups
            if self.verbose:
                print(f"  Filtered library precursors: {initial_precursors} -> {after_decoy_precursors} (removed decoys) -> {after_qvalue_precursors} (Q.Value <= 0.01)")
        else:
            if self.verbose:
                print(f"  Filtered library precursors: {initial_precursors} -> {after_decoy_precursors} (removed decoys)")
                print("  Warning: 'Q.Value' column not found, skipping Q-value filtering")
        
        # Apply test limit if specified (after filtering for decoys and Q-value)
        if self.test_limit_peptides > 0:
            # Get unique precursors (sequence, charge combinations)
            unique_precursors = self.library_df[['Stripped.Sequence', 'Precursor.Charge']].drop_duplicates()
            
            # Randomly select N precursors with fixed seed for reproducibility
            import random
            random.seed(42)  # Fixed seed for reproducible selection
            if len(unique_precursors) > self.test_limit_peptides:
                limited_precursors = unique_precursors.sample(n=self.test_limit_peptides, random_state=42)
            else:
                limited_precursors = unique_precursors
            
            # Filter library to only include these peptides
            self.library_df = self.library_df.merge(
                limited_precursors,
                on=['Stripped.Sequence', 'Precursor.Charge'],
                how='inner'
            )
            
            final_precursors = self.library_df.groupby(['Stripped.Sequence', 'Precursor.Charge']).ngroups
            if self.verbose:
                print(f"  Test mode: Randomly selected {self.test_limit_peptides} precursors (seed=42) -> {final_precursors} precursors in library")

        # Build index: (stripped_sequence, charge) -> precursor data
        # Group by precursor to collect all fragments
        grouped = self.library_df.groupby(['Stripped.Sequence', 'Precursor.Charge'])

        for (sequence, charge), group in grouped:
            # Extract fragment data
            fragments = []
            for _, row in group.iterrows():
                fragments.append({
                    'mz': row['Product.Mz'],
                    'intensity': row['Relative.Intensity'],
                    'type': row['Fragment.Type'],  # 'y' or 'b'
                    'charge': row['Fragment.Charge'],
                    'series_number': row['Fragment.Series.Number'],
                })

            # Preprocess library fragments: sqrt(intensity) * mz^2
            lib_mz = np.array([frag['mz'] for frag in fragments])
            lib_intensity = np.array([frag['intensity'] for frag in fragments])
            lib_preprocessed = np.sqrt(lib_intensity) * (lib_mz ** 2)
            
            # Normalize library vector once
            lib_norm = np.linalg.norm(lib_preprocessed)
            if lib_norm > 0:
                lib_preprocessed_normalized = lib_preprocessed / lib_norm
            else:
                lib_preprocessed_normalized = lib_preprocessed

            # Store precursor data with preprocessed fragments
            self.peptide_index[(sequence, charge)] = {
                'sequence': sequence,
                'charge': charge,
                'precursor_mz': group['Precursor.Mz'].iloc[0],
                'rt': group['RT'].iloc[0],
                'modified_sequence': group['Modified.Sequence'].iloc[0],
                'protein_ids': group['Protein.Ids'].iloc[0],
                'fragments': fragments,
                'preprocessed_fragments': lib_preprocessed_normalized,  # Pre-computed SMZ
            }

        if self.verbose:
            print(f"Loaded library with {len(self.peptide_index)} precursors from {library_path}")

    def get_precursor(self, sequence: str, charge: int) -> Optional[Dict]:
        """
        Get precursor data from library.

        Returns:
            Dictionary with precursor_mz, rt, fragments list, or None if not found
        """
        return self.peptide_index.get((sequence, charge))

    def has_peptide(self, sequence: str, charge: int) -> bool:
        """Check if peptide exists in library."""
        return (sequence, charge) in self.peptide_index

    def get_all_peptides(self) -> List[Tuple[str, int]]:
        """Get list of all (sequence, charge) tuples in library."""
        return list(self.peptide_index.keys())

    @staticmethod
    def parse_unimod_sequence(modified_sequence: str) -> Tuple[str, Dict[int, float]]:
        """
        Parse DIA-NN modified sequence with UniMod annotations.

        Args:
            modified_sequence: e.g., "AAAPAPVSEAVC(UniMod:4)R"

        Returns:
            (stripped_sequence, modifications_dict)
            modifications_dict: {position: mass_shift}
        """
        import re

        stripped = ""
        modifications = {}
        position = 0

        # Pattern to match amino acid followed by optional (UniMod:N)
        pattern = r'([A-Z])(?:\(UniMod:(\d+)\))?'

        for match in re.finditer(pattern, modified_sequence):
            aa = match.group(1)
            unimod_id = match.group(2)

            stripped += aa

            if unimod_id and unimod_id in SpectrumLibrary.UNIMOD_MASSES:
                modifications[position] = SpectrumLibrary.UNIMOD_MASSES[unimod_id]

            position += 1

        return stripped, modifications

    def generate_decoy_fragments(self, sequence: str, charge: int, xcorr_engine) -> Optional[Dict]:
        """
        Generate decoy fragment spectrum by remapping target fragment intensities.

        Strategy:
        - Reverse the peptide sequence (keeping C-terminal K/R if present)
        - For each target fragment at position N from terminus, assign its intensity
          to the corresponding position N in the decoy sequence
        - Recalculate fragment m/z values for the decoy sequence

        Args:
            sequence: Target peptide sequence (stripped)
            charge: Precursor charge
            xcorr_engine: FastXCorr instance for mass calculations

        Returns:
            Dictionary with decoy fragment data, or None if not in library
        """
        # Check if already generated
        decoy_key = (sequence, charge, True)
        if decoy_key in self.decoy_fragments:
            return self.decoy_fragments[decoy_key]

        # Get target data
        target_data = self.get_precursor(sequence, charge)
        if not target_data:
            return None

        # Generate decoy sequence (reverse, keeping C-term K/R)
        decoy_sequence = self._reverse_sequence(sequence)

        # Build intensity map: (frag_type, series_number, frag_charge) -> intensity
        # This maps target fragment positions to their intensities
        intensity_map = {}
        for frag in target_data['fragments']:
            key = (frag['type'], frag['series_number'], frag['charge'])
            intensity_map[key] = frag['intensity']

        # Generate decoy fragments by mapping intensities to new positions
        decoy_fragments = []
        sequence_length = len(decoy_sequence)

        # For each possible fragment in the decoy
        for frag_type in ['y', 'b']:
            for series_num in range(1, sequence_length):
                for frag_charge in [1, 2]:  # DIA-NN typically uses 1+ and 2+
                    # Check if this position had intensity in the target
                    key = (frag_type, series_num, frag_charge)
                    if key in intensity_map:
                        # Calculate m/z for this decoy fragment
                        frag_mz = self._calculate_fragment_mz(
                            decoy_sequence, frag_type, series_num, frag_charge, xcorr_engine
                        )

                        if frag_mz > 0:  # Valid fragment
                            decoy_fragments.append({
                                'mz': frag_mz,
                                'intensity': intensity_map[key],
                                'type': frag_type,
                                'charge': frag_charge,
                                'series_number': series_num,
                            })

        # Calculate decoy precursor m/z
        decoy_mass = sum(xcorr_engine.aa_masses.get(aa, 0) for aa in decoy_sequence)
        decoy_mass += xcorr_engine.h2o_mass  # Add H2O for neutral mass
        decoy_precursor_mz = (decoy_mass + charge * xcorr_engine.proton_mass) / charge

        # Preprocess decoy fragments: sqrt(intensity) * mz^2
        if decoy_fragments:
            lib_mz_decoy = np.array([frag['mz'] for frag in decoy_fragments])
            lib_intensity_decoy = np.array([frag['intensity'] for frag in decoy_fragments])
            lib_preprocessed_decoy = np.sqrt(lib_intensity_decoy) * (lib_mz_decoy ** 2)
            
            # Normalize library vector once
            lib_norm_decoy = np.linalg.norm(lib_preprocessed_decoy)
            if lib_norm_decoy > 0:
                lib_preprocessed_decoy_normalized = lib_preprocessed_decoy / lib_norm_decoy
            else:
                lib_preprocessed_decoy_normalized = lib_preprocessed_decoy
        else:
            lib_preprocessed_decoy_normalized = np.array([])

        # Store decoy data with preprocessed fragments
        decoy_data = {
            'sequence': decoy_sequence,
            'charge': charge,
            'precursor_mz': decoy_precursor_mz,
            'rt': target_data['rt'],  # Same RT as target
            'fragments': decoy_fragments,
            'preprocessed_fragments': lib_preprocessed_decoy_normalized,  # Pre-computed SMZ
            'is_decoy': True,
        }

        self.decoy_fragments[decoy_key] = decoy_data
        return decoy_data

    @staticmethod
    def _reverse_sequence(sequence: str) -> str:
        """
        Reverse peptide sequence, keeping C-terminal K/R.

        Args:
            sequence: e.g., "PEPTIDER"

        Returns:
            Reversed sequence, e.g., "EDITPEPR"
        """
        if len(sequence) <= 1:
            return sequence

        # Check if C-terminal is K or R
        if sequence[-1] in ['K', 'R']:
            # Reverse all except last residue, keep K/R at end
            return sequence[-2::-1] + sequence[-1]
        else:
            # Full reversal
            return sequence[::-1]

    @staticmethod
    def _calculate_fragment_mz(sequence: str, frag_type: str, series_number: int,
                               frag_charge: int, xcorr_engine) -> float:
        """
        Calculate fragment ion m/z.

        Args:
            sequence: Peptide sequence
            frag_type: 'y' or 'b'
            series_number: Fragment series number (1-indexed)
            frag_charge: Fragment charge state
            xcorr_engine: FastXCorr instance for masses

        Returns:
            Fragment m/z value, or 0 if invalid
        """
        if series_number < 1 or series_number >= len(sequence):
            return 0

        if frag_type == 'y':
            # y-ions: C-terminal fragments (count from C-terminus)
            fragment_seq = sequence[-series_number:]
            mass = sum(xcorr_engine.aa_masses.get(aa, 0) for aa in fragment_seq)
            mass += xcorr_engine.h2o_mass + xcorr_engine.proton_mass  # y-ion includes OH + H
        elif frag_type == 'b':
            # b-ions: N-terminal fragments (count from N-terminus)
            fragment_seq = sequence[:series_number]
            mass = sum(xcorr_engine.aa_masses.get(aa, 0) for aa in fragment_seq)
            mass += xcorr_engine.proton_mass  # b-ion is just protonated
        else:
            return 0

        # Add additional protons for charge and divide
        mz = (mass + (frag_charge - 1) * xcorr_engine.proton_mass) / frag_charge
        return mz

    def sample_precursors(self, n: int, seed: int = 42, max_qvalue: float = 0.01):
        """
        Randomly sample N high-quality precursors from library for calibration.
        
        Args:
            n: Number of precursors to sample
            seed: Random seed for reproducibility
            max_qvalue: Maximum q-value for "high-quality" (default: 0.01)
            
        Returns:
            List of (stripped_sequence, charge) tuples
        """
        import random
        random.seed(seed)
        
        if self.library_df is None:
            raise ValueError("Library not loaded")
        
        # Filter to high quality if Q.Value column exists
        if 'Q.Value' in self.library_df.columns:
            high_quality = self.library_df[self.library_df['Q.Value'] <= max_qvalue]
        else:
            high_quality = self.library_df
        
        # Get unique precursors (Stripped.Sequence + Precursor.Charge)
        precursors = high_quality[['Stripped.Sequence', 'Precursor.Charge']].drop_duplicates()
        
        # Sample N
        if len(precursors) < n:
            print(f"Warning: Only {len(precursors)} high-quality precursors available (requested {n})")
            sampled = precursors
        else:
            sampled = precursors.sample(n=n, random_state=seed)
        
        # Return as list of tuples
        return [(row['Stripped.Sequence'], row['Precursor.Charge']) 
                for _, row in sampled.iterrows()]


class FastXCorr:
    """
    Fast cross-correlation implementation based on Comet's approach.
    
    This class implements Comet's optimized cross-correlation calculation with:
    1. Spectrum binning (1.0005079 Da bins)
    2. MakeCorrData windowing normalization (10 windows, normalize to 50.0)
    3. Fast XCorr preprocessing with sliding window (offset=75)
    4. Unified dot product scoring (supports both single and matrix operations)
    5. Static modifications support (default: carbamidomethylation of cysteine +57.021464)
    
    This implementation closely follows the Comet source code to ensure 
    compatibility and reproducibility with the established search engine.
    """
    
    def __init__(self, bin_width: float = 1.0005079, bin_offset: float = 0.4, static_modifications: Optional[Dict[str, float]] = None):
        self.bin_width = bin_width
        self.mass_range = (0, 2000)  # m/z range
        self.num_bins = int((self.mass_range[1] - self.mass_range[0]) / bin_width) + 1
        
        # Comet BIN macro parameters
        # BIN(dMass) = (int)((dMass)*g_staticParams.dInverseBinWidth + g_staticParams.dOneMinusBinOffset)
        self.inverse_bin_width = 1.0 / bin_width  # g_staticParams.dInverseBinWidth
        self.bin_offset = bin_offset  # g_staticParams.dOneMinusBinOffset (configurable via command line)
        
        # Amino acid masses (monoisotopic, unmodified)
        self.base_aa_masses = {
            'A': 71.037114, 'R': 156.101111, 'N': 114.042927, 'D': 115.026943,
            'C': 103.009185, 'E': 129.042593, 'Q': 128.058578, 'G': 57.021464,
            'H': 137.058912, 'I': 113.084064, 'L': 113.084064, 'K': 128.094963,
            'M': 131.040485, 'F': 147.068414, 'P': 97.052764, 'S': 87.032028,
            'T': 101.047679, 'W': 186.079313, 'Y': 163.063329, 'V': 99.068414
        }
        
        # Static modifications (fixed modifications applied to all instances)
        # Default: Carbamidomethylation of cysteine (+57.021464)
        # To add other static modifications, pass a dictionary like:
        # {'C': 57.021464, 'M': 15.994915}  # Carbamidomethyl-Cys + Oxidation-Met
        # Common examples:
        #   'M': 15.994915   # Oxidation of methionine
        #   'K': 8.014199    # 13C(6)15N(2) lysine (SILAC)
        #   'R': 10.008269   # 13C(6)15N(4) arginine (SILAC)
        #   'S': 79.966331   # Phosphorylation of serine
        #   'T': 79.966331   # Phosphorylation of threonine
        #   'Y': 79.966331   # Phosphorylation of tyrosine
        if static_modifications is None:
            self.static_modifications = {'C': 57.021464}  # Carbamidomethylation
        else:
            self.static_modifications = static_modifications.copy()
        
        # Apply static modifications to create final aa_masses
        self.aa_masses = self.base_aa_masses.copy()
        for aa, mod_mass in self.static_modifications.items():
            if aa in self.aa_masses:
                self.aa_masses[aa] += mod_mass
        
        # Ion type masses
        self.h2o_mass = 18.010565
        self.nh3_mass = 17.026549
        self.proton_mass = 1.007276
        
        # Pre-sorted peptide candidates by m/z for fast lookup
        self.sorted_peptides_by_mz = {}  # Dict[charge_state, List[Tuple[mz, peptide]]]
        
        # Enzyme cleavage patterns and properties
        self.enzymes = {
            'trypsin': {
                'pattern': r'[KR](?!P)',  # Cleaves after K/R, not before P
                'cleavage_type': 'c_terminal',  # Cleaves C-terminal to residue
                'cleavage_residues': ['K', 'R'],  # Residues that define cleavage
                'description': 'Trypsin with proline suppression'
            },
            'trypsin_no_proline': {
                'pattern': r'[KR]',  # Cleaves after K/R
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['K', 'R'],
                'description': 'Trypsin without proline suppression'
            },
            'lysc': {
                'pattern': r'K(?!P)',  # Cleaves after K, not before P
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['K'],
                'description': 'Lys-C with proline suppression'
            },
            'lysn': {
                'pattern': r'.(?=K)',  # Cleaves before K
                'cleavage_type': 'n_terminal',
                'cleavage_residues': ['K'],
                'description': 'Lys-N'
            },
            'argc': {
                'pattern': r'R(?!P)',  # Cleaves after R, not before P
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['R'],
                'description': 'Arg-C with proline suppression'
            },
            'aspn': {
                'pattern': r'.(?=D)',  # Cleaves before D
                'cleavage_type': 'n_terminal',
                'cleavage_residues': ['D'],
                'description': 'Asp-N'
            },
            'cnbr': {
                'pattern': r'M',  # Cleaves after M
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['M'],
                'description': 'CNBr chemical cleavage'
            },
            'gluc': {
                'pattern': r'[DE](?!P)',  # Cleaves after D/E, not before P
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['D', 'E'],
                'description': 'Glu-C with proline suppression'
            },
            'pepsina': {
                'pattern': r'[FL](?!P)',  # Cleaves after F/L, not before P
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['F', 'L'],
                'description': 'Pepsin A with proline suppression'
            },
            'chymotrypsin': {
                'pattern': r'[FWYL](?!P)',  # Cleaves after F/W/Y/L, not before P
                'cleavage_type': 'c_terminal',
                'cleavage_residues': ['F', 'W', 'Y', 'L'],
                'description': 'Chymotrypsin with proline suppression'
            }
        }
    
    def get_static_modifications(self) -> Dict[str, float]:
        """
        Get the current static modifications.
        
        Returns:
            Dictionary mapping amino acid to mass modification
        """
        return self.static_modifications.copy()
    
    def add_static_modification(self, amino_acid: str, mass_delta: float):
        """
        Add or update a static modification for an amino acid.
        
        Args:
            amino_acid: Single letter amino acid code
            mass_delta: Mass delta to add to the amino acid (in Da)
        """
        if amino_acid not in self.base_aa_masses:
            raise ValueError(f"Unknown amino acid: {amino_acid}")
        
        self.static_modifications[amino_acid] = mass_delta
        # Update the working aa_masses dictionary
        self.aa_masses[amino_acid] = self.base_aa_masses[amino_acid] + mass_delta
    
    def remove_static_modification(self, amino_acid: str):
        """
        Remove a static modification for an amino acid.
        
        Args:
            amino_acid: Single letter amino acid code
        """
        if amino_acid in self.static_modifications:
            del self.static_modifications[amino_acid]
            # Reset to base mass
            self.aa_masses[amino_acid] = self.base_aa_masses[amino_acid]
    
    def bin_mass(self, mass: float) -> int:
        """
        Comet's BIN macro implementation with 0.4 offset.
        BIN(dMass) = (int)((dMass)*g_staticParams.dInverseBinWidth + g_staticParams.dOneMinusBinOffset)
        """
        return int(mass * self.inverse_bin_width + self.bin_offset)
        
    def read_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Read protein sequences from FASTA file."""
        proteins = {}
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        proteins[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]  # Take first part of header
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id:
                proteins[current_id] = ''.join(current_seq)
        
        return proteins
    
    def read_mzml_combined(self, mzml_file: str, max_spectra: int = 0, preprocess_smz: bool = False) -> Tuple[List[MassSpectrum], List[MS1Spectrum]]:
        """Read MS1 and MS2 spectra from mzML file in a single pass.
        
        Args:
            mzml_file: Path to mzML file
            max_spectra: Maximum number of MS2 spectra to read (0 = all)
            preprocess_smz: Apply SMZ preprocessing during read (sqrt(intensity) * mz^2)
            
        Returns:
            Tuple of (ms2_spectra, ms1_spectra)
        """
        ms2_spectra = []
        ms1_spectra = []
        
        # Open the mzML file with pymzml
        run = pymzml.run.Reader(mzml_file)
        
        for spectrum_idx, spectrum in enumerate(run):
            if spectrum.ms_level == 2:
                # Process MS2 spectrum
                # Get precursor information
                precursor_mz = 0.0
                charge = 0
                isolation_window_lower = 0.0
                isolation_window_upper = 0.0
                
                # Get precursor m/z and charge using pymzml API
                if hasattr(spectrum, 'selected_precursors') and spectrum.selected_precursors:
                    precursor = spectrum.selected_precursors[0]
                    if 'mz' in precursor:
                        precursor_mz = float(precursor['mz'])
                    
                    # Try to get charge from the precursor element
                    if 'element' in precursor:
                        element = precursor['element']
                        
                        # Look for charge state in selectedIonList/selectedIon/cvParam elements
                        for child in element:
                            if 'selectedIonList' in child.tag:
                                for selected_ion in child:
                                    if 'selectedIon' in selected_ion.tag:
                                        for cv_param in selected_ion:
                                            if 'cvParam' in cv_param.tag:
                                                # MS:1000633 = possible charge state
                                                # MS:1000041 = charge state
                                                accession = cv_param.attrib.get('accession', '')
                                                if accession in ['MS:1000633', 'MS:1000041']:
                                                    value = cv_param.attrib.get('value', '')
                                                    if value:
                                                        charge = int(float(value))
                                                        break
                        
                        # Look for isolation window information in the precursor element
                        for child in element:
                            if 'isolationWindow' in child.tag:
                                for iso_child in child:
                                    if 'lowerOffset' in iso_child.attrib:
                                        isolation_window_lower = precursor_mz - float(iso_child.attrib['lowerOffset'])
                                    elif 'upperOffset' in iso_child.attrib:
                                        isolation_window_upper = precursor_mz + float(iso_child.attrib['upperOffset'])
                                    elif iso_child.get('name') == 'isolation window lower offset':
                                        isolation_window_lower = precursor_mz - float(iso_child.get('value', 0))
                                    elif iso_child.get('name') == 'isolation window upper offset':
                                        isolation_window_upper = precursor_mz + float(iso_child.get('value', 0))
                    
                    # Default charge if still not found (we'll use configurable charge states anyway)
                    if charge == 0:
                        charge = 2  # Common default for MS/MS
                    
                    # If isolation window not found, use default 3 m/z window
                    if isolation_window_lower == 0.0 and isolation_window_upper == 0.0:
                        isolation_window_lower = precursor_mz - 1.5
                        isolation_window_upper = precursor_mz + 1.5
                
                # Get scan ID and retention time
                scan_id = spectrum.ID
                
                # Try to get retention time, fallback to scan index if not available
                try:
                    retention_time_minutes = spectrum.scan_time_in_minutes()
                except (AttributeError, TypeError):
                    # If RT not available, use scan index as a proxy
                    retention_time_minutes = float(spectrum_idx)
                
                if not scan_id:
                    scan_id = f"scan_{retention_time_minutes:.4f}"
                
                # Get m/z and intensity arrays
                if len(spectrum.peaks('centroided')) > 0:
                    peaks = spectrum.peaks('centroided')
                    mz_array = np.array([peak[0] for peak in peaks])
                    intensity_array = np.array([peak[1] for peak in peaks])
                    
                    # Step 4: Apply SMZ preprocessing if requested
                    if preprocess_smz:
                        intensity_array = np.sqrt(intensity_array) * (mz_array ** 2)
                    
                    mass_spectrum = MassSpectrum(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        scan_id=scan_id,
                        precursor_mz=precursor_mz,
                        charge=charge,
                        isolation_window_lower=isolation_window_lower,
                        isolation_window_upper=isolation_window_upper,
                        retention_time=retention_time_minutes
                    )
                    ms2_spectra.append(mass_spectrum)
                    
                    # Stop reading if we've reached the maximum number of MS2 spectra
                    if max_spectra > 0 and len(ms2_spectra) >= max_spectra:
                        break
                        
            elif spectrum.ms_level == 1:
                # Process MS1 spectrum
                # Get scan ID
                scan_id = spectrum.ID
                if not scan_id:
                    scan_id = f"ms1_scan_{spectrum_idx}"

                # Try to get retention time
                try:
                    retention_time_minutes = spectrum.scan_time_in_minutes()
                except (AttributeError, TypeError):
                    retention_time_minutes = float(spectrum_idx)

                # Get m/z and intensity arrays
                if len(spectrum.peaks('centroided')) > 0:
                    peaks = spectrum.peaks('centroided')
                    mz_array = np.array([peak[0] for peak in peaks])
                    intensity_array = np.array([peak[1] for peak in peaks])

                    ms1_spectrum = MS1Spectrum(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        scan_id=scan_id,
                        retention_time=retention_time_minutes
                    )
                    ms1_spectra.append(ms1_spectrum)
        
        # Sort MS1 spectra by retention time for efficient lookup
        ms1_spectra.sort(key=lambda x: x.retention_time)
        
        return ms2_spectra, ms1_spectra

    def read_mzml(self, mzml_file: str, max_spectra: int = 0) -> List[MassSpectrum]:
        """Read mass spectra from mzML file using pymzml."""
        spectra = []
        
        # Open the mzML file with pymzml
        run = pymzml.run.Reader(mzml_file)
        
        for spectrum_idx, spectrum in enumerate(run):
            # Only process MS2 spectra
            if spectrum.ms_level != 2:
                continue
            
            # Get precursor information
            precursor_mz = 0.0
            charge = 0
            isolation_window_lower = 0.0
            isolation_window_upper = 0.0
            
            # Get precursor m/z and charge using pymzml API
            if hasattr(spectrum, 'selected_precursors') and spectrum.selected_precursors:
                precursor = spectrum.selected_precursors[0]
                if 'mz' in precursor:
                    precursor_mz = float(precursor['mz'])
                
                # Try to get charge from the precursor element
                if 'element' in precursor:
                    element = precursor['element']
                    
                    # Look for charge state in selectedIonList/selectedIon/cvParam elements
                    for child in element:
                        if 'selectedIonList' in child.tag:
                            for selected_ion in child:
                                if 'selectedIon' in selected_ion.tag:
                                    for cv_param in selected_ion:
                                        if 'cvParam' in cv_param.tag:
                                            # MS:1000633 = possible charge state
                                            # MS:1000041 = charge state
                                            accession = cv_param.attrib.get('accession', '')
                                            if accession in ['MS:1000633', 'MS:1000041']:
                                                value = cv_param.attrib.get('value', '')
                                                if value:
                                                    charge = int(float(value))
                                                    break
                    
                    # Look for isolation window information in the precursor element
                    for child in element:
                        if 'isolationWindow' in child.tag:
                            for iso_child in child:
                                if 'lowerOffset' in iso_child.attrib:
                                    isolation_window_lower = precursor_mz - float(iso_child.attrib['lowerOffset'])
                                elif 'upperOffset' in iso_child.attrib:
                                    isolation_window_upper = precursor_mz + float(iso_child.attrib['upperOffset'])
                                elif iso_child.get('name') == 'isolation window lower offset':
                                    isolation_window_lower = precursor_mz - float(iso_child.get('value', 0))
                                elif iso_child.get('name') == 'isolation window upper offset':
                                    isolation_window_upper = precursor_mz + float(iso_child.get('value', 0))
                
                # Default charge if still not found (we'll use configurable charge states anyway)
                if charge == 0:
                    charge = 2  # Common default for MS/MS
                
                # If isolation window not found, use default 3 m/z window
                if isolation_window_lower == 0.0 and isolation_window_upper == 0.0:
                    isolation_window_lower = precursor_mz - 1.5
                    isolation_window_upper = precursor_mz + 1.5
            
            # Get scan ID and retention time
            scan_id = spectrum.ID
            
            # Try to get retention time, fallback to scan index if not available
            try:
                retention_time_minutes = spectrum.scan_time_in_minutes()
            except (AttributeError, TypeError):
                # If RT not available, use scan index as a proxy
                retention_time_minutes = float(spectrum_idx)
            
            if not scan_id:
                scan_id = f"scan_{retention_time_minutes:.4f}"
            
            # Get m/z and intensity arrays
            if len(spectrum.peaks('centroided')) > 0:
                peaks = spectrum.peaks('centroided')
                mz_array = np.array([peak[0] for peak in peaks])
                intensity_array = np.array([peak[1] for peak in peaks])
                
                mass_spectrum = MassSpectrum(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    scan_id=scan_id,
                    precursor_mz=precursor_mz,
                    charge=charge,
                    isolation_window_lower=isolation_window_lower,
                    isolation_window_upper=isolation_window_upper,
                    retention_time=retention_time_minutes
                )
                spectra.append(mass_spectrum)
                
                # Stop reading if we've reached the maximum number of spectra
                if max_spectra > 0 and len(spectra) >= max_spectra:
                    break
        
        return spectra

    def read_ms1_spectra(self, mzml_file: str) -> List[MS1Spectrum]:
        """
        Read MS1 spectra from mzML file and index by retention time.

        Args:
            mzml_file: Path to mzML file

        Returns:
            List of MS1Spectrum objects, sorted by retention time
        """
        ms1_spectra = []

        # Open the mzML file with pymzml
        run = pymzml.run.Reader(mzml_file)

        for spectrum_idx, spectrum in enumerate(run):
            # Only process MS1 spectra
            if spectrum.ms_level != 1:
                continue

            # Get scan ID
            scan_id = spectrum.ID
            if not scan_id:
                scan_id = f"ms1_scan_{spectrum_idx}"

            # Try to get retention time
            try:
                retention_time_minutes = spectrum.scan_time_in_minutes()
            except (AttributeError, TypeError):
                retention_time_minutes = float(spectrum_idx)

            # Get m/z and intensity arrays
            if len(spectrum.peaks('centroided')) > 0:
                peaks = spectrum.peaks('centroided')
                mz_array = np.array([peak[0] for peak in peaks])
                intensity_array = np.array([peak[1] for peak in peaks])

                ms1_spectrum = MS1Spectrum(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    scan_id=scan_id,
                    retention_time=retention_time_minutes
                )
                ms1_spectra.append(ms1_spectrum)

        # Sort by retention time for efficient lookup
        ms1_spectra.sort(key=lambda x: x.retention_time)

        return ms1_spectra

    @staticmethod
    def find_closest_ms1(ms1_spectra: List[MS1Spectrum], target_rt: float) -> Optional[MS1Spectrum]:
        """
        Find the MS1 spectrum closest to the target retention time.

        Args:
            ms1_spectra: List of MS1Spectrum objects (sorted by RT)
            target_rt: Target retention time in minutes

        Returns:
            Closest MS1Spectrum, or None if list is empty
        """
        if not ms1_spectra:
            return None

        # Binary search for closest RT
        rts = [ms1.retention_time for ms1 in ms1_spectra]
        idx = bisect.bisect_left(rts, target_rt)

        # Check boundaries
        if idx == 0:
            return ms1_spectra[0]
        elif idx == len(ms1_spectra):
            return ms1_spectra[-1]

        # Compare adjacent entries
        before = ms1_spectra[idx - 1]
        after = ms1_spectra[idx]

        if abs(before.retention_time - target_rt) <= abs(after.retention_time - target_rt):
            return before
        else:
            return after

    def read_mgf(self, mgf_file: str, max_spectra: int = 0) -> List[MassSpectrum]:
        """Read mass spectra from MGF file using pyteomics."""
        spectra = []
        
        with mgf.read(mgf_file) as reader:
            for spectrum_idx, spectrum in enumerate(reader):
                params = spectrum.get('params', {})
                
                # Handle pepmass which can be a float, list, or tuple (mass, intensity)
                pepmass = params.get('pepmass', 0.0)
                if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
                    precursor_mz = float(pepmass[0])  # First element is always the mass
                elif pepmass is not None:
                    try:
                        precursor_mz = float(pepmass)
                    except (ValueError, TypeError):
                        precursor_mz = 0.0
                else:
                    precursor_mz = 0.0
                
                # Handle charge which can be an int, list, or string
                charge_param = params.get('charge', 2)
                if isinstance(charge_param, (list, tuple)) and len(charge_param) > 0:
                    try:
                        charge = int(charge_param[0])
                    except (ValueError, TypeError):
                        charge = 2
                elif charge_param is not None:
                    try:
                        charge = int(charge_param)
                    except (ValueError, TypeError):
                        charge = 2
                else:
                    charge = 2
                
                # MGF typically doesn't have isolation window info, use default
                isolation_window_lower = precursor_mz - 1.5
                isolation_window_upper = precursor_mz + 1.5
                
                # Get scan_id from title or use index
                scan_id = params.get('title', f"scan_{spectrum_idx}")
                
                # Get m/z and intensity arrays
                mz_array = spectrum.get('m/z array', np.array([]))
                intensity_array = spectrum.get('intensity array', np.array([]))
                
                if len(mz_array) > 0 and len(intensity_array) > 0:
                    mass_spectrum = MassSpectrum(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        scan_id=scan_id,
                        precursor_mz=precursor_mz,
                        charge=charge,
                        isolation_window_lower=isolation_window_lower,
                        isolation_window_upper=isolation_window_upper
                    )
                    spectra.append(mass_spectrum)
                    
                    # Stop reading if we've reached the maximum number of spectra
                    if max_spectra > 0 and len(spectra) >= max_spectra:
                        break
        
        return spectra

    def read_single_spectrum(self, spectrum_file: str, scan_id: str) -> MassSpectrum:
        """Read a single spectrum by scan ID from mzML or MGF file."""
        if spectrum_file.lower().endswith('.mzml'):
            return self._read_single_spectrum_mzml(spectrum_file, scan_id)
        elif spectrum_file.lower().endswith('.mgf'):
            return self._read_single_spectrum_mgf(spectrum_file, scan_id)
        else:
            raise ValueError(f"Unsupported spectrum file format: {spectrum_file}")
    
    def _read_single_spectrum_mzml(self, mzml_file: str, scan_id: str) -> MassSpectrum:
        """Read a single spectrum by scan ID using mzML indexing for fast access."""
        try:
            # Use pymzml's indexed access for fast random access
            run = pymzml.run.Reader(mzml_file, build_index_from_scratch=False)
            
            # Try to get the spectrum directly by ID
            spectrum = run[scan_id]
            
            # Only process MS2 spectra
            if spectrum.ms_level != 2:
                raise ValueError(f"Scan {scan_id} is not an MS2 spectrum (ms_level={spectrum.ms_level})")
            
            # Extract spectrum data using the same logic as read_mzml
            precursor_mz = 0.0
            charge = 0
            isolation_window_lower = 0.0
            isolation_window_upper = 0.0
            
            # Get precursor information
            if hasattr(spectrum, 'selected_precursors') and spectrum.selected_precursors:
                precursor = spectrum.selected_precursors[0]
                if 'mz' in precursor:
                    precursor_mz = float(precursor['mz'])
                
                # Try to get charge from the precursor element
                if 'element' in precursor:
                    element = precursor['element']
                    # Look for charge state in child elements
                    for child in element:
                        if 'charge' in child.attrib or 'selectedIonMZ' in child.tag:
                            for sub_child in child:
                                if 'charge' in sub_child.attrib:
                                    charge = int(sub_child.attrib['charge'])
                                    break
                        
                        # Look for isolation window information in the precursor element
                        if 'isolationWindow' in child.tag:
                            for iso_child in child:
                                if 'lowerOffset' in iso_child.attrib:
                                    isolation_window_lower = precursor_mz - float(iso_child.attrib['lowerOffset'])
                                elif 'upperOffset' in iso_child.attrib:
                                    isolation_window_upper = precursor_mz + float(iso_child.attrib['upperOffset'])
                                elif iso_child.get('name') == 'isolation window lower offset':
                                    isolation_window_lower = precursor_mz - float(iso_child.get('value', 0))
                                elif iso_child.get('name') == 'isolation window upper offset':
                                    isolation_window_upper = precursor_mz + float(iso_child.get('value', 0))
                
                # Default charge if still not found
                if charge == 0:
                    charge = 2  # Common default for MS/MS
                
                # If isolation window not found, use default 3 m/z window
                if isolation_window_lower == 0.0 and isolation_window_upper == 0.0:
                    isolation_window_lower = precursor_mz - 1.5
                    isolation_window_upper = precursor_mz + 1.5
            
            # Get m/z and intensity arrays
            if len(spectrum.peaks('centroided')) > 0:
                peaks = spectrum.peaks('centroided')
                mz_array = np.array([peak[0] for peak in peaks])
                intensity_array = np.array([peak[1] for peak in peaks])
                
                mass_spectrum = MassSpectrum(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    scan_id=scan_id,
                    precursor_mz=precursor_mz,
                    charge=charge,
                    isolation_window_lower=isolation_window_lower,
                    isolation_window_upper=isolation_window_upper
                )
                return mass_spectrum
            else:
                raise ValueError(f"No peaks found in spectrum {scan_id}")
                
        except KeyError:
            raise ValueError(f"Scan ID {scan_id} not found in mzML file")
        except Exception as e:
            # Fallback to sequential search if indexing fails
            print(f"Warning: Fast access failed ({e}), falling back to sequential search...")
            return self._read_single_spectrum_sequential(mzml_file, scan_id)
    
    def _read_single_spectrum_sequential(self, mzml_file: str, scan_id: str) -> MassSpectrum:
        """Fallback method to read single spectrum sequentially."""
        run = pymzml.run.Reader(mzml_file)
        
        for spectrum in run:
            if spectrum.ID == scan_id and spectrum.ms_level == 2:
                # Use same extraction logic as read_mzml
                precursor_mz = 0.0
                charge = 0
                isolation_window_lower = 0.0
                isolation_window_upper = 0.0
                
                if hasattr(spectrum, 'selected_precursors') and spectrum.selected_precursors:
                    precursor = spectrum.selected_precursors[0]
                    if 'mz' in precursor:
                        precursor_mz = float(precursor['mz'])
                    
                    if charge == 0:
                        charge = 2
                    
                    if isolation_window_lower == 0.0 and isolation_window_upper == 0.0:
                        isolation_window_lower = precursor_mz - 1.5
                        isolation_window_upper = precursor_mz + 1.5
                
                if len(spectrum.peaks('centroided')) > 0:
                    peaks = spectrum.peaks('centroided')
                    mz_array = np.array([peak[0] for peak in peaks])
                    intensity_array = np.array([peak[1] for peak in peaks])
                    
                    return MassSpectrum(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        scan_id=scan_id,
                        precursor_mz=precursor_mz,
                        charge=charge,
                        isolation_window_lower=isolation_window_lower,
                        isolation_window_upper=isolation_window_upper
                    )
        
        raise ValueError(f"Scan ID {scan_id} not found in mzML file")
    
    def _read_single_spectrum_mgf(self, mgf_file: str, scan_id: str) -> MassSpectrum:
        """Read a single spectrum by scan ID from MGF file using pyteomics."""
        with mgf.read(mgf_file) as reader:
            for spectrum_idx, spectrum in enumerate(reader):
                params = spectrum.get('params', {})
                spectrum_scan_id = params.get('title', f"scan_{spectrum_idx}")
                
                if spectrum_scan_id == scan_id:
                    # Extract spectrum data using the same logic as read_mgf
                    params = spectrum.get('params', {})
                    
                    # Handle pepmass which can be a float, list, or tuple (mass, intensity)
                    pepmass = params.get('pepmass', 0.0)
                    if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
                        precursor_mz = float(pepmass[0])  # First element is always the mass
                    elif pepmass is not None:
                        try:
                            precursor_mz = float(pepmass)
                        except (ValueError, TypeError):
                            precursor_mz = 0.0
                    else:
                        precursor_mz = 0.0
                    
                    # Handle charge which can be an int, list, or string
                    charge_param = params.get('charge', 2)
                    if isinstance(charge_param, (list, tuple)) and len(charge_param) > 0:
                        try:
                            charge = int(charge_param[0])
                        except (ValueError, TypeError):
                            charge = 2
                    elif charge_param is not None:
                        try:
                            charge = int(charge_param)
                        except (ValueError, TypeError):
                            charge = 2
                    else:
                        charge = 2
                    
                    # MGF typically doesn't have isolation window info, use default
                    isolation_window_lower = precursor_mz - 1.5
                    isolation_window_upper = precursor_mz + 1.5
                    
                    # Get m/z and intensity arrays
                    mz_array = spectrum.get('m/z array', np.array([]))
                    intensity_array = spectrum.get('intensity array', np.array([]))
                    
                    if len(mz_array) > 0 and len(intensity_array) > 0:
                        return MassSpectrum(
                            mz_array=mz_array,
                            intensity_array=intensity_array,
                            scan_id=scan_id,
                            precursor_mz=precursor_mz,
                            charge=charge,
                            isolation_window_lower=isolation_window_lower,
                            isolation_window_upper=isolation_window_upper
                        )
        
        raise ValueError(f"Scan ID {scan_id} not found in MGF file")

    def list_ms2_scan_ids(self, mzml_file: str, max_scans: int = 100) -> List[str]:
        """Get a list of available MS2 scan IDs for spectrum selection."""
        scan_ids = []
        run = pymzml.run.Reader(mzml_file)
        
        for spectrum in run:
            if spectrum.ms_level == 2:
                scan_ids.append(spectrum.ID)
                if max_scans > 0 and len(scan_ids) >= max_scans:
                    break
        
        return scan_ids
    
    def generate_decoy_sequence(self, sequence: str, cycle_length: int = 1, enzyme: str = 'trypsin') -> str:
        """
        Generate decoy peptide sequence by cycling N amino acids (default 1).
        Preserve the cleavage site residue based on the enzyme used.
        
        Args:
            sequence: Original peptide sequence
            cycle_length: Number of positions to cycle (default 1)
            enzyme: Enzyme used for digestion (determines which residue to preserve)
            
        Returns:
            Decoy sequence with cycled amino acids
        """
        if len(sequence) <= 1:
            return sequence
        
        # Get enzyme properties
        if enzyme not in self.enzymes:
            raise ValueError(f"Unknown enzyme: {enzyme}")
        
        enzyme_info = self.enzymes[enzyme]
        cleavage_type = enzyme_info['cleavage_type']
        cleavage_residues = enzyme_info['cleavage_residues']
        
        # Determine which residue to preserve based on cleavage type
        if cleavage_type == 'c_terminal':
            # C-terminal cleavage: preserve C-terminal residue if it's a cleavage residue
            if sequence[-1] in cleavage_residues:
                core_sequence = sequence[:-1]
                preserved_residue = sequence[-1]
                preserve_position = 'c_terminal'
            else:
                # No cleavage residue at C-terminus, cycle entire sequence
                core_sequence = sequence
                preserved_residue = ''
                preserve_position = None
        else:  # n_terminal
            # N-terminal cleavage: preserve N-terminal residue if it's a cleavage residue
            if sequence[0] in cleavage_residues:
                core_sequence = sequence[1:]
                preserved_residue = sequence[0]
                preserve_position = 'n_terminal'
            else:
                # No cleavage residue at N-terminus, cycle entire sequence
                core_sequence = sequence
                preserved_residue = ''
                preserve_position = None
        
        # If core sequence too short to cycle meaningfully
        if len(core_sequence) <= cycle_length:
            return sequence  # Return original if can't cycle
        
        # Cycle the sequence by moving first N amino acids to the end
        cycle_length = cycle_length % len(core_sequence)  # Handle cycle_length > sequence length
        if cycle_length == 0:
            cycled_core = core_sequence
        else:
            cycled_core = core_sequence[cycle_length:] + core_sequence[:cycle_length]
        
        # Reconstruct with preserved residue
        if preserve_position == 'c_terminal':
            return cycled_core + preserved_residue
        elif preserve_position == 'n_terminal':
            return preserved_residue + cycled_core
        else:
            return cycled_core
    
    def generate_reversed_decoy_sequence(self, sequence: str, enzyme: str = 'trypsin') -> str:
        """
        Generate decoy peptide sequence by reversing amino acids.
        Preserve the cleavage site residue based on the enzyme used.
        
        Args:
            sequence: Original peptide sequence
            enzyme: Enzyme used for digestion (determines which residue to preserve)
            
        Returns:
            Decoy sequence with reversed amino acids
        """
        if len(sequence) <= 1:
            return sequence
        
        # Get enzyme properties
        if enzyme not in self.enzymes:
            raise ValueError(f"Unknown enzyme: {enzyme}")
        
        enzyme_info = self.enzymes[enzyme]
        cleavage_type = enzyme_info['cleavage_type']
        cleavage_residues = enzyme_info['cleavage_residues']
        
        # Determine which residue to preserve based on cleavage type
        if cleavage_type == 'c_terminal':
            # C-terminal cleavage: preserve C-terminal residue if it's a cleavage residue
            if sequence[-1] in cleavage_residues:
                core_sequence = sequence[:-1]
                preserved_residue = sequence[-1]
                preserve_position = 'c_terminal'
            else:
                # No cleavage residue at C-terminus, reverse entire sequence
                core_sequence = sequence
                preserved_residue = ''
                preserve_position = None
        else:  # n_terminal
            # N-terminal cleavage: preserve N-terminal residue if it's a cleavage residue
            if sequence[0] in cleavage_residues:
                core_sequence = sequence[1:]
                preserved_residue = sequence[0]
                preserve_position = 'n_terminal'
            else:
                # No cleavage residue at N-terminus, reverse entire sequence
                core_sequence = sequence
                preserved_residue = ''
                preserve_position = None
        
        # Reverse the core sequence
        reversed_core = core_sequence[::-1]
        
        # Reconstruct with preserved residue
        if preserve_position == 'c_terminal':
            return reversed_core + preserved_residue
        elif preserve_position == 'n_terminal':
            return preserved_residue + reversed_core
        else:
            return reversed_core
    
    def make_peptides_non_redundant(self, all_peptides: List[PeptideCandidate]) -> List[PeptideCandidate]:
        """
        Make peptide list non-redundant by concatenating protein accessions for duplicate sequences.
        
        Args:
            all_peptides: List of all peptide candidates (may contain duplicates)
            
        Returns:
            List of non-redundant peptides with concatenated protein IDs
        """
        # Dictionary to group peptides by sequence
        peptide_groups = defaultdict(list)
        
        for peptide in all_peptides:
            peptide_groups[peptide.sequence].append(peptide)
        
        # Create non-redundant list
        non_redundant_peptides = []
        for sequence, peptides in peptide_groups.items():
            if len(peptides) == 1:
                # Single occurrence, keep as is
                non_redundant_peptides.append(peptides[0])
            else:
                # Multiple occurrences, concatenate protein IDs
                protein_ids = [p.protein_id for p in peptides]
                concatenated_protein_id = ';'.join(sorted(set(protein_ids)))  # Remove duplicates and sort
                
                # Use first peptide as template, update protein_id
                merged_peptide = PeptideCandidate(sequence, concatenated_protein_id, peptides[0].mass)
                non_redundant_peptides.append(merged_peptide)
        
        return non_redundant_peptides
    
    def generate_target_decoy_pairs(self, target_peptides: List[PeptideCandidate], 
                                  cycle_length: int = 1, enzyme: str = 'trypsin') -> List[Tuple[PeptideCandidate, PeptideCandidate]]:
        """
        Generate target-decoy pairs for proper target-decoy competition.
        
        Uses reversal as the default decoy generation method (keeping cleavage site residue fixed),
        with cycling as a fallback if reversal fails to generate a valid decoy.
        
        Args:
            target_peptides: List of target peptides (should be non-redundant)
            cycle_length: Number of positions to cycle for decoy generation (used in fallback)
            enzyme: Enzyme used for digestion (determines which residue to preserve)
            
        Returns:
            List of (target_peptide, decoy_peptide) tuples
        """
        target_decoy_pairs = []
        
        # Create a set of all target sequences for collision detection
        target_sequences = {peptide.sequence for peptide in target_peptides}
        
        # Track statistics
        pairs_created = 0
        collisions_resolved = 0
        cycling_fallback_used = 0
        max_retries_exceeded = 0
        
        for target_peptide in target_peptides:
            decoy_generated = False
            max_retries = min(10, len(target_peptide.sequence) - 1)
            
            # First, try reversal as the default method
            decoy_sequence = self.generate_reversed_decoy_sequence(target_peptide.sequence, enzyme)
            
            # Check if reversed decoy is valid
            if decoy_sequence != target_peptide.sequence and decoy_sequence not in target_sequences:
                # Create decoy peptide using reversal
                decoy_protein_id = f"decoy_{target_peptide.protein_id}"
                decoy_mass = self.calculate_peptide_mass(decoy_sequence)
                decoy_peptide = PeptideCandidate(decoy_sequence, decoy_protein_id, decoy_mass)
                
                # Create target-decoy pair
                target_decoy_pairs.append((target_peptide, decoy_peptide))
                pairs_created += 1
                decoy_generated = True
            
            # If reversal failed, try cycling as fallback
            if not decoy_generated:
                for retry_cycle in range(cycle_length, cycle_length + max_retries):
                    decoy_sequence = self.generate_decoy_sequence(target_peptide.sequence, retry_cycle, enzyme)
                    
                    # Check if decoy is valid (different from target and not in target database)
                    if decoy_sequence != target_peptide.sequence and decoy_sequence not in target_sequences:
                        # Create decoy peptide
                        decoy_protein_id = f"decoy_{target_peptide.protein_id}"
                        decoy_mass = self.calculate_peptide_mass(decoy_sequence)
                        decoy_peptide = PeptideCandidate(decoy_sequence, decoy_protein_id, decoy_mass)
                        
                        # Create target-decoy pair
                        target_decoy_pairs.append((target_peptide, decoy_peptide))
                        pairs_created += 1
                        cycling_fallback_used += 1
                        decoy_generated = True
                        
                        # Track collision statistics
                        if retry_cycle > cycle_length:
                            collisions_resolved += retry_cycle - cycle_length
                        
                        break
                    elif decoy_sequence in target_sequences:
                        # Collision detected, try next cycle length
                        continue
                    else:
                        # No meaningful decoy could be generated
                        break
            
            if not decoy_generated:
                max_retries_exceeded += 1
        
        # Report statistics
        print("Target-decoy pair generation summary:")
        print(f"  Target peptides: {len(target_peptides)}")
        print(f"  Target-decoy pairs created: {pairs_created}")
        print(f"  Collisions resolved: {collisions_resolved}")
        print(f"  Cycling fallback used: {cycling_fallback_used}")
        print(f"  Peptides without valid decoys: {max_retries_exceeded}")
        
        return target_decoy_pairs
    
    def digest_protein(self, sequence: str, protein_id: str, 
                      enzyme: str = 'trypsin', missed_cleavages: int = 1,
                      min_length: int = 7, max_length: int = 30) -> List[PeptideCandidate]:
        """
        Digest protein sequence into peptides using specified enzyme.
        
        Args:
            sequence: Protein sequence
            protein_id: Protein identifier
            enzyme: Enzyme name (see self.enzymes for options)
            missed_cleavages: Maximum number of missed cleavages
            min_length: Minimum peptide length in amino acids
            max_length: Maximum peptide length in amino acids
            
        Returns:
            List of peptide candidates
        """
        peptides = []
        
        # Validate enzyme
        if enzyme not in self.enzymes:
            available = ', '.join(self.enzymes.keys())
            raise ValueError(f"Enzyme '{enzyme}' not supported. Available enzymes: {available}")
        
        # Get cleavage pattern for the enzyme
        cleavage_pattern = self.enzymes[enzyme]['pattern']
        cleavage_type = self.enzymes[enzyme]['cleavage_type']
        
        # For C-terminal cleavage: split after the cleavage site
        # For N-terminal cleavage: we need to handle differently
        if cleavage_type == 'c_terminal':
            # Pattern matches position after the cleavage residue
            # Use lookahead to insert split marker after the residue
            cleavage_regex = f'(?<={cleavage_pattern})'
            fragments = re.split(cleavage_regex, sequence)
        else:  # n_terminal
            # Pattern matches position before the cleavage residue
            # For N-terminal cleavage, split before the residue
            cleavage_regex = f'(?={cleavage_pattern[4:-1]})'  # Extract residue from .(?=X) pattern
            fragments = re.split(cleavage_regex, sequence)
        
        # Generate peptides with missed cleavages
        for i in range(len(fragments)):
            for j in range(i, min(i + missed_cleavages + 1, len(fragments))):
                peptide_seq = ''.join(fragments[i:j+1])
                
                # Filter by length
                if min_length <= len(peptide_seq) <= max_length:
                    mass = self.calculate_peptide_mass(peptide_seq)
                    peptides.append(PeptideCandidate(peptide_seq, protein_id, mass))
        
        return peptides
    
    def build_peptide_index(self, peptide_candidates: List[PeptideCandidate], charge_states: List[int]):
        """
        Build sorted index of peptides by theoretical m/z for fast isolation window lookup.
        
        Args:
            peptide_candidates: List of all peptides to index
            charge_states: List of charge states to consider
        """
        print("Building peptide m/z index for fast lookup...")
        
        # Clear existing index
        self.sorted_peptides_by_mz = {}
        
        for charge in charge_states:
            peptide_mz_pairs = []
            
            for peptide in peptide_candidates:
                theoretical_mz = (peptide.mass + charge * self.proton_mass) / charge
                peptide_mz_pairs.append((theoretical_mz, peptide))
            
            # Sort by m/z for binary search
            peptide_mz_pairs.sort(key=lambda x: x[0])
            self.sorted_peptides_by_mz[charge] = peptide_mz_pairs
            
            print(f"  Charge +{charge}: {len(peptide_mz_pairs)} peptides indexed, m/z range: {peptide_mz_pairs[0][0]:.3f} - {peptide_mz_pairs[-1][0]:.3f}")
    
    def find_peptides_in_isolation_window(self, isolation_window_lower: float, isolation_window_upper: float, 
                                        charge_states: List[int]) -> List[Tuple[PeptideCandidate, int]]:
        """
        Fast lookup of peptides within an isolation window using binary search.
        
        Args:
            isolation_window_lower: Lower bound of isolation window (m/z)
            isolation_window_upper: Upper bound of isolation window (m/z)
            charge_states: Charge states to search
            
        Returns:
            List of (peptide, charge) tuples for peptides within the window
        """
        
        peptide_charge_pairs = []
        
        for charge in charge_states:
            if charge not in self.sorted_peptides_by_mz:
                continue
                
            sorted_peptides = self.sorted_peptides_by_mz[charge]
            
            # Binary search for lower bound
            left_idx = bisect.bisect_left(sorted_peptides, (isolation_window_lower, None))
            # Binary search for upper bound
            right_idx = bisect.bisect_right(sorted_peptides, (isolation_window_upper, None))
            
            # Extract peptides in the range
            for i in range(left_idx, right_idx):
                theoretical_mz, peptide = sorted_peptides[i]
                peptide_charge_pairs.append((peptide, charge))
        
        return peptide_charge_pairs
    
    def calculate_peptide_mass(self, sequence: str) -> float:
        """Calculate monoisotopic mass of peptide."""
        mass = self.h2o_mass  # Add water for peptide
        
        for aa in sequence:
            if aa in self.aa_masses:
                mass += self.aa_masses[aa]
            else:
                # Handle unknown amino acids
                mass += 100.0  # Approximate average AA mass
        
        return mass
    
    def generate_theoretical_spectrum(self, peptide: PeptideCandidate, charge: int) -> np.ndarray:
        """
        Generate theoretical spectrum for peptide using Comet's exact method.
        
        CORRECTED: Theoretical spectrum uses unit intensities (1.0), not 50.0
        Only the experimental spectrum gets normalized to 50.0 in MakeCorrData.
        """
        spectrum = np.zeros(self.num_bins)
        sequence = peptide.sequence
        
        # Comet's precalculated masses (from g_staticParams.precalcMasses)
        # dNtermProton = PROTON_MASS = 1.007276
        # dCtermOH2Proton = H2O_MASS + PROTON_MASS = 18.010565 + 1.007276 = 19.017841
        nterm_proton = self.proton_mass  # 1.007276
        cterm_oh2_proton = self.h2o_mass + self.proton_mass  # 19.017841
        
        # Generate b ions (N-terminal fragments) - Comet's method
        b_mass = nterm_proton  # Start with N-terminal proton
        for i in range(len(sequence) - 1):  # Exclude last residue (no b_n ion)
            b_mass += self.aa_masses.get(sequence[i], 100.0)
            
            # Generate fragment charges from 1+ up to (precursor_charge - 1)+
            # For +1 precursor: only 1+ fragments
            # For +2 precursor: only 1+ fragments
            # For +3 precursor: 1+ and 2+ fragments  
            # For +4 precursor: 1+, 2+, and 3+ fragments
            max_frag_charge = min(charge - 1, 3)  # Cap at 3+ fragments (like Comet)
            max_frag_charge = max(max_frag_charge, 1)  # Always generate at least 1+ fragments
            for frag_charge in range(1, max_frag_charge + 1):  # 1+ to max_frag_charge
                mz = (b_mass + (frag_charge - 1) * self.proton_mass) / frag_charge
                if self.mass_range[0] <= mz <= self.mass_range[1]:
                    bin_idx = self.bin_mass(mz)
                    relative_bin_idx = bin_idx - self.bin_mass(self.mass_range[0])
                    if 0 <= relative_bin_idx < self.num_bins:
                        # CORRECTED: Use unit intensity (1.0) not 50.0
                        # Only experimental spectrum gets normalized to 50.0 in MakeCorrData
                        spectrum[relative_bin_idx] = 1.0
        
        # Generate y ions (C-terminal fragments) - Comet's method
        y_mass = cterm_oh2_proton  # Start with C-terminal OH2 + proton
        for i in range(len(sequence) - 1, 0, -1):  # Exclude first residue (no y_n ion)
            y_mass += self.aa_masses.get(sequence[i], 100.0)
            
            # Generate fragment charges from 1+ up to (precursor_charge - 1)+
            # For +1 precursor: only 1+ fragments
            # For +2 precursor: only 1+ fragments
            # For +3 precursor: 1+ and 2+ fragments  
            # For +4 precursor: 1+, 2+, and 3+ fragments
            max_frag_charge = min(charge - 1, 3)  # Cap at 3+ fragments (like Comet)
            max_frag_charge = max(max_frag_charge, 1)  # Always generate at least 1+ fragments
            for frag_charge in range(1, max_frag_charge + 1):  # 1+ to max_frag_charge
                mz = (y_mass + (frag_charge - 1) * self.proton_mass) / frag_charge
                if self.mass_range[0] <= mz <= self.mass_range[1]:
                    bin_idx = self.bin_mass(mz)
                    relative_bin_idx = bin_idx - self.bin_mass(self.mass_range[0])
                    if 0 <= relative_bin_idx < self.num_bins:
                        # CORRECTED: Use unit intensity (1.0) not 50.0
                        # Only experimental spectrum gets normalized to 50.0 in MakeCorrData
                        spectrum[relative_bin_idx] = 1.0
        
        return spectrum
    
    def preprocess_spectrum(self, spectrum: MassSpectrum) -> np.ndarray:
        """
        Preprocess experimental spectrum according to Comet's algorithm.
        
        This follows the Comet preprocessing pipeline:
        1. Use all peaks (no filtering by intensity, like Comet)
        2. Bin spectrum into unit mass bins (taking max intensity per bin)
        3. Apply square root transformation to intensities (as SEQUEST does)
        4. Apply Comet's MakeCorrData windowing normalization to 50.0  
        5. Store result for fast XCorr preprocessing
        """
        # Step 1: Use all peaks - no intensity filtering (Comet strategy)
        # Note: Comet's windowing will handle intensity normalization per window
        filtered_mz = spectrum.mz_array
        filtered_intensity = spectrum.intensity_array
        
        # Step 2: Bin spectrum (equivalent to SEQUEST's LoadIons)
        binned_spectrum = np.zeros(self.num_bins)
        highest_intensity = 0.0
        highest_ion_bin = 0
        
        for mz, intensity in zip(filtered_mz, filtered_intensity):
            if self.mass_range[0] <= mz <= self.mass_range[1]:
                # Apply SEQUEST's square root transformation to intensity
                sqrt_intensity = np.sqrt(intensity)
                bin_idx = self.bin_mass(mz)  # Use Comet's BIN macro with 0.4 offset
                # Convert absolute bin to relative bin for our array indexing
                relative_bin_idx = bin_idx - self.bin_mass(self.mass_range[0])
                if 0 <= relative_bin_idx < self.num_bins:
                    binned_spectrum[relative_bin_idx] = max(binned_spectrum[relative_bin_idx], sqrt_intensity)
                    if binned_spectrum[relative_bin_idx] > highest_intensity:
                        highest_intensity = binned_spectrum[relative_bin_idx]
                    # Track the highest bin index that contains any data (not just the highest intensity)
                    if binned_spectrum[relative_bin_idx] > 0:
                        highest_ion_bin = max(highest_ion_bin, relative_bin_idx)
        
        # Step 3: Apply SEQUEST's MakeCorrData windowing normalization
        # This is the key function that makes SEQUEST's preprocessing distinctive
        windowed_spectrum = self._make_corr_data(binned_spectrum, highest_ion_bin, highest_intensity)
        
        # Store both raw and windowed spectra for debugging
        spectrum.processed_spectrum = windowed_spectrum
        return windowed_spectrum
    
    def _make_corr_data(self, raw_spectrum: np.ndarray, highest_ion: int, highest_intensity: float) -> np.ndarray:
        """
        Comet's MakeCorrData function - applies windowing normalization.
        
        This exactly follows Comet's implementation:
        - 10 windows total (iNumWindows = 10)
        - Normalize experimental spectrum to 50.0 within each window (dTmp1 = 50.0 / dMaxWindowInten)
        - Apply 5% of base peak threshold (dTmp2 = 0.05 * dHighestIntensity)
        
        From Comet source: dTmp1 = 50.0 / dMaxWindowInten;
        """
        windowed_spectrum = np.zeros_like(raw_spectrum)
        num_windows = 10
        window_size = (highest_ion // num_windows) + 1
        
        for i in range(num_windows):
            # Find max intensity in this window
            max_window_intensity = 0.0
            
            for ii in range(window_size):
                bin_idx = i * window_size + ii
                if bin_idx <= highest_ion and bin_idx < len(raw_spectrum):
                    if raw_spectrum[bin_idx] > max_window_intensity:
                        max_window_intensity = raw_spectrum[bin_idx]
            
            # Normalize within window if there's signal
            if max_window_intensity > 0.0:
                # Comet's exact implementation: dTmp1 = 50.0 / dMaxWindowInten
                normalization_factor = 50.0 / max_window_intensity
                threshold = 0.05 * highest_intensity
                
                for ii in range(window_size):
                    bin_idx = i * window_size + ii
                    if bin_idx <= highest_ion and bin_idx < len(raw_spectrum):
                        if raw_spectrum[bin_idx] > threshold:
                            windowed_spectrum[bin_idx] = raw_spectrum[bin_idx] * normalization_factor
        
        return windowed_spectrum
    
    def preprocess_for_xcorr(self, windowed_spectrum: np.ndarray) -> np.ndarray:
        """
        Apply Comet's fast XCorr preprocessing - CORRECTED VERSION
        
        This implements Comet's sliding window approach exactly as in the source code:
        1. Calculate sliding window average with offset (default offset = 75)
        2. Subtract from windowed spectrum: final = windowed - sliding_avg
        3. Add flanking peaks contribution (Comet's default behavior)
        """
        # Comet's default XCorr processing offset (g_staticParams.iXcorrProcessingOffset)
        xcorr_offset = 75  # This is Comet's default value
        
        # Initialize arrays for the two-step process
        sliding_window_avg = np.zeros_like(windowed_spectrum)
        
        # Calculate sliding window statistics
        # iTmpRange = 2 * iXcorrProcessingOffset + 1 = 151
        window_range = 2 * xcorr_offset + 1
        normalization_factor = 1.0 / (window_range - 1.0)  # Comet's dTmp = 1.0 / 150.0
        
        # Initialize sliding sum for the first window
        sliding_sum = 0.0
        for i in range(xcorr_offset):
            if i < len(windowed_spectrum):
                sliding_sum += windowed_spectrum[i]
        
        # Apply Comet's exact sliding window algorithm
        for i in range(xcorr_offset, len(windowed_spectrum) + xcorr_offset):
            # Add new element to window if within bounds
            if i < len(windowed_spectrum):
                sliding_sum += windowed_spectrum[i]
            
            # Remove old element from window if within bounds
            if i >= window_range:
                sliding_sum -= windowed_spectrum[i - window_range]
            
            # Calculate sliding window average
            array_idx = i - xcorr_offset
            if array_idx < len(windowed_spectrum):
                # Comet's exact formula: (sliding_sum - current_value) * normalization
                sliding_window_avg[array_idx] = (sliding_sum - windowed_spectrum[array_idx]) * normalization_factor
        
        # CORRECTED: Apply Comet's final preprocessing step
        # pfFastXcorrData[i] = pdTmpCorrelationData[i] - pdTmpFastXcorrData[i]
        final_preprocessed = np.zeros_like(windowed_spectrum)
        final_preprocessed[0] = 0.0  # Comet sets first element to 0
        
        for i in range(1, len(windowed_spectrum)):
            # Core Comet formula: experimental_windowed - sliding_window_average
            final_preprocessed[i] = windowed_spectrum[i] - sliding_window_avg[i]
            
            # Add flanking peaks contribution (Comet's default behavior when iTheoreticalFragmentIons == 0)
            # This is enabled by default in Comet
            if i > 0:
                # Add left neighbor contribution
                final_preprocessed[i] += (windowed_spectrum[i-1] - sliding_window_avg[i-1]) * 0.5
            
            if i < len(windowed_spectrum) - 1:
                # Add right neighbor contribution
                final_preprocessed[i] += (windowed_spectrum[i+1] - sliding_window_avg[i+1]) * 0.5
        
        return final_preprocessed
    
    def calculate_xcorr(self, spectrum_a: np.ndarray, spectrum_b: np.ndarray, 
                       scaling_factor: float = 0.005) -> Union[float, np.ndarray]:
        """
        Unified XCorr calculation supporting both single and batch (matrix) scoring.
        
        This function handles:
        1. Single spectrum scoring: spectrum_a and spectrum_b are 1D arrays
        2. Batch matrix scoring: spectrum_a is 2D (n_spectra, n_bins), spectrum_b is 2D (m_spectra, n_bins)
        
        The core algorithm is identical for both spectrum-centric and peptide-centric searches:
        - XCorr = dot_product(spectrum_a, spectrum_b) * scaling_factor
        
        The only differences are:
        - Which spectrum is preprocessed (experimental vs theoretical)
        - Scaling factor: 0.005 for spectrum-centric, 0.0001 for peptide-centric
        
        Args:
            spectrum_a: First spectrum/spectra (1D array or 2D matrix)
            spectrum_b: Second spectrum/spectra (1D array or 2D matrix)
            scaling_factor: XCorr scaling factor (0.005 for spectrum-centric, 0.0001 for peptide-centric)
            
        Returns:
            XCorr score (float) or XCorr matrix (2D array) for batch scoring
        """
        # Detect if this is matrix multiplication (batch scoring)
        is_matrix = spectrum_a.ndim == 2 or spectrum_b.ndim == 2
        
        if is_matrix:
            # Matrix multiplication: (n, bins) @ (m, bins).T = (n, m)
            # This scores n spectra against m spectra in one optimized operation
            if spectrum_a.ndim == 1:
                spectrum_a = spectrum_a.reshape(1, -1)
            if spectrum_b.ndim == 1:
                spectrum_b = spectrum_b.reshape(1, -1)
            
            # Use @ operator for matrix multiplication (calls optimized BLAS)
            raw_xcorr = spectrum_a @ spectrum_b.T
            
            # Apply scaling and round
            final_xcorr = np.round(raw_xcorr * scaling_factor, 4)
            
            # Return scalar if result is 1x1, otherwise return matrix
            if final_xcorr.shape == (1, 1):
                return float(final_xcorr[0, 0])
            return final_xcorr
        else:
            # Vector dot product: single spectrum vs single spectrum
            # Ensure both spectra have the same length
            min_len = min(len(spectrum_a), len(spectrum_b))
            
            # Calculate dot product
            raw_xcorr = np.dot(spectrum_a[:min_len], spectrum_b[:min_len])
            
            # Apply scaling factor and round
            final_xcorr = round(raw_xcorr * scaling_factor, 4)
            
            return final_xcorr
    
    def calculate_fast_xcorr(self, theoretical_spectrum: np.ndarray, 
                           preprocessed_experimental: np.ndarray) -> float:
        """
        Calculate fast cross-correlation score for spectrum-centric search.
        
        This is a convenience wrapper around calculate_xcorr() for spectrum-centric mode:
        - Theoretical spectrum: raw (unit intensities)
        - Experimental spectrum: fully preprocessed (windowed + Fast XCorr)
        - Scaling: 0.005 (Comet's standard scaling)
        
        Args:
            theoretical_spectrum: Theoretical spectrum (raw, unit intensities)
            preprocessed_experimental: Experimental spectrum (windowed + Fast XCorr preprocessing)
            
        Returns:
            XCorr score scaled by 0.005
        """
        result = self.calculate_xcorr(theoretical_spectrum, preprocessed_experimental, 
                                      scaling_factor=0.005)
        return float(result)  # Guaranteed to be float for 1D inputs
    
    def calculate_e_value(self, xcorr_scores: List[float], top_score: float) -> float:
        """
        Calculate E-value using Comet's LinearRegression approach.
        
        This implements Comet's E-value calculation using:
        1. XCorr score histogram (binned by 0.1 units, scaled by 10)
        2. Cumulative distribution function from right to left
        3. Log transformation of cumulative counts
        4. Linear regression on log-transformed data
        5. Projection of top score to fitted line
        """
        if len(xcorr_scores) < 10:  # Need enough scores for statistics
            return 1.0
        
        # Comet uses bins of 0.1 XCorr units (multiplied by 10 for integer indexing)
        # Note: Since we removed the 0.005 scaling, scores are now in their natural range
        HISTO_SIZE = 1000  # Comet's HISTO_SIZE constant
        histogram = np.zeros(HISTO_SIZE, dtype=int)
        
        # Fill histogram: bin by 0.1 units (multiply by 10)
        for score in xcorr_scores:
            bin_idx = int(score * 10.0 + 0.5)  # Comet's rounding approach
            if bin_idx < 0:
                bin_idx = 0
            if bin_idx >= HISTO_SIZE:
                bin_idx = HISTO_SIZE - 1
            histogram[bin_idx] += 1
        
        # Find maximum non-zero score bin (iMaxCorr)
        max_corr = 0
        for i in range(HISTO_SIZE - 2, -1, -1):
            if histogram[i] > 0:
                max_corr = i
                break
        
        if max_corr < 10:  # Need reasonable score range
            return 1.0
        
        # Find appropriate regression range (iNextCorr)
        next_corr = 0
        found_first_nonzero = False
        
        for i in range(max_corr):
            if histogram[i] == 0 and found_first_nonzero and i >= 10:
                # Register next_corr if there's a consecutive zero
                if i + 1 >= max_corr or histogram[i + 1] == 0:
                    if i > 0:
                        next_corr = i - 1
                    break
            if histogram[i] != 0:
                found_first_nonzero = True
        
        if next_corr == 0:
            next_corr = max_corr
            if max_corr >= 10:
                # Look for zeros in the tail
                for i in range(max_corr, max(max_corr - 5, -1), -1):
                    if histogram[i] == 0:
                        next_corr = i
                        if max_corr <= 20:
                            break
                if next_corr == max_corr:
                    next_corr = max_corr - 1
        
        # Create cumulative distribution function (from right to left)
        cumulative = np.zeros(HISTO_SIZE)
        cumulative[next_corr] = histogram[next_corr]
        
        for i in range(next_corr - 1, -1, -1):
            cumulative[i] = cumulative[i + 1] + histogram[i]
            if histogram[i + 1] == 0:
                cumulative[i + 1] = 0.0
        
        # Log transform cumulative data
        for i in range(next_corr, -1, -1):
            if cumulative[i] > 0.0:
                cumulative[i] = np.log10(cumulative[i])
            else:
                # Handle zeros by interpolation from neighbors
                if i < next_corr and cumulative[i + 1] > 0.0:
                    cumulative[i] = cumulative[i + 1]
                else:
                    cumulative[i] = 0.0
        
        # Linear regression on log-transformed data
        start_corr = next_corr - 5
        if start_corr < 0:
            start_corr = 0
        
        # Count zeros and adjust start
        num_zeros = sum(1 for i in range(start_corr, next_corr + 1) if cumulative[i] == 0)
        start_corr -= num_zeros
        if start_corr < 0:
            start_corr = 0
        
        # Perform regression while start_corr >= 0 and we have enough points
        slope = 0.0
        mean_x = 0.0
        mean_y = 0.0
        
        while start_corr >= 0 and next_corr > start_corr + 2:
            sum_x = sum_y = sum_xy = sum_xx = 0.0
            num_points = 0
            
            # Calculate means
            for i in range(start_corr, next_corr + 1):
                if histogram[i] > 0:
                    sum_x += i
                    sum_y += cumulative[i]
                    num_points += 1
            
            if num_points > 0:
                mean_x = sum_x / num_points
                mean_y = sum_y / num_points
                
                # Calculate slope
                for i in range(start_corr, next_corr + 1):
                    if histogram[i] > 0:
                        dx = i - mean_x
                        dy = cumulative[i] - mean_y
                        sum_xx += dx * dx
                        sum_xy += dx * dy
                
                if sum_xx > 0:
                    slope = sum_xy / sum_xx
                else:
                    slope = 0.0
                
                if slope < 0.0:
                    break
                else:
                    start_corr -= 1
            else:
                break
        
        # Calculate intercept AFTER the loop completes (Comet algorithm)
        intercept = mean_y - slope * mean_x
        
        # Calculate E-value for top score
        if slope < 0.0:  # Valid regression
            # Multiply slope by 10 for final calculation (Comet does this)
            slope *= 10.0
            log_expect = slope * top_score + intercept
            expect_value = 10.0 ** log_expect
            
            # Cap e-value at reasonable bounds
            # E-values should be probabilities in range [0, 1], but allow slightly above 1
            # due to estimation errors, and cap at 1.0
            if expect_value > 1.0:
                expect_value = 1.0
            
            return max(expect_value, 1e-10)
        
        return 1.0
    
    def calculate_e_value_by_charge(self, score_distributions_by_charge: Dict[int, List[float]], xcorr_score: float, charge: int) -> float:
        """
        Calculate E-value for a specific charge state using charge-specific score distribution.
        
        Args:
            score_distributions_by_charge: Dictionary mapping charge state to list of XCorr scores
            xcorr_score: The XCorr score to calculate E-value for
            charge: The charge state for E-value calculation
            
        Returns:
            E-value for the given score and charge state
        """
        if charge not in score_distributions_by_charge:
            return 1.0  # No data for this charge state
            
        xcorr_scores = score_distributions_by_charge[charge]
        
        if len(xcorr_scores) < 10:  # Need minimum scores for statistical modeling
            return 1.0
            
        # Use the existing calculate_e_value method with charge-specific scores
        return self.calculate_e_value(xcorr_scores, xcorr_score)

    def search_spectrum_target_decoy(self, spectrum: MassSpectrum, target_decoy_pairs: List[Tuple[PeptideCandidate, PeptideCandidate]],
                                    charge_states: List[int] = [2, 3]) -> List[Tuple[PeptideCandidate, float, float, int]]:
        """
        Search spectrum with proper target-decoy competition.
        
        For each target-decoy pair that falls within the isolation window:
        1. Score both target and decoy against the spectrum
        2. Keep only the winner (higher XCorr score)
        3. Return the top N winners across all charge states
        
        Args:
            spectrum: The experimental spectrum
            target_decoy_pairs: List of (target_peptide, decoy_peptide) tuples
            charge_states: List of charge states to consider
            
        Returns:
            List of (winning_peptide, xcorr_score, e_value, charge) tuples
        """
        # Apply preprocessing once
        windowed_spectrum = self.preprocess_spectrum(spectrum)
        preprocessed_spectrum = self.preprocess_for_xcorr(windowed_spectrum)
        
        # Get isolation window
        isolation_window_lower = spectrum.isolation_window_lower
        isolation_window_upper = spectrum.isolation_window_upper
        
        if isolation_window_lower == 0.0 or isolation_window_upper == 0.0:
            return []
        
        # Find target-decoy pairs within isolation window and conduct competition
        competition_winners = []
        score_distributions_by_charge = {}  # Track scores separately for each charge state
        
        for target_peptide, decoy_peptide in target_decoy_pairs:
            for charge in charge_states:
                # Check if either target or decoy falls within isolation window
                target_mz = (target_peptide.mass + charge * self.proton_mass) / charge
                decoy_mz = (decoy_peptide.mass + charge * self.proton_mass) / charge
                
                target_in_window = isolation_window_lower <= target_mz <= isolation_window_upper
                decoy_in_window = isolation_window_lower <= decoy_mz <= isolation_window_upper
                
                # For proper target-decoy competition, both should have same mass
                # But check both just in case there are small mass differences
                if target_in_window or decoy_in_window:
                    # Score both target and decoy
                    target_theoretical = self.generate_theoretical_spectrum(target_peptide, charge)
                    target_xcorr = self.calculate_fast_xcorr(target_theoretical, preprocessed_spectrum)
                    
                    decoy_theoretical = self.generate_theoretical_spectrum(decoy_peptide, charge)
                    decoy_xcorr = self.calculate_fast_xcorr(decoy_theoretical, preprocessed_spectrum)
                    
                    # Target-decoy competition: keep the winner
                    if target_xcorr >= decoy_xcorr:
                        winner = target_peptide
                        winning_score = target_xcorr
                    else:
                        winner = decoy_peptide
                        winning_score = decoy_xcorr
                    
                    competition_winners.append((winner, winning_score, charge))
                    
                    # Track scores by charge state for separate E-value calculations
                    if charge not in score_distributions_by_charge:
                        score_distributions_by_charge[charge] = []
                    score_distributions_by_charge[charge].append(winning_score)
        
        # Sort winners by XCorr score (descending) within each charge state
        results_by_charge = {}
        for winner, score, charge in competition_winners:
            if charge not in results_by_charge:
                results_by_charge[charge] = []
            results_by_charge[charge].append((winner, score, charge))
        
        # Sort within each charge state
        for charge in results_by_charge:
            results_by_charge[charge].sort(key=lambda x: x[1], reverse=True)
        
        # Calculate E-values separately for each charge state using charge-specific distributions
        final_results = []
        for charge in charge_states:
            if charge in results_by_charge:
                for winner, xcorr_score, charge in results_by_charge[charge]:
                    # Use charge-specific E-value calculation
                    e_value = self.calculate_e_value_by_charge(score_distributions_by_charge, xcorr_score, charge)
                    final_results.append((winner, xcorr_score, e_value, charge))
        
        # Sort by charge state first, then by XCorr score
        final_results.sort(key=lambda x: (x[3], -x[1]))
        
        return final_results
    
    def group_spectra_by_isolation_window(self, spectra: List[MassSpectrum]) -> Dict[Tuple[float, float], List[MassSpectrum]]:
        """
        Group spectra by their isolation window (precursor m/z window).
        
        For DIA data, multiple spectra will have the same isolation window.
        This groups them together for efficient peptide-centric searching.
        
        Args:
            spectra: List of mass spectra
            
        Returns:
            Dictionary mapping (lower, upper) window bounds to list of spectra
        """
        window_groups = defaultdict(list)
        
        for spectrum in spectra:
            window_key = (spectrum.isolation_window_lower, spectrum.isolation_window_upper)
            window_groups[window_key].append(spectrum)
        
        return dict(window_groups)
    
    def preprocess_theoretical_spectrum(self, theoretical_binned: np.ndarray) -> np.ndarray:
        """
        Preprocess a theoretical spectrum for peptide-centric DIA search.
        
        In peptide-centric mode, we preprocess the THEORETICAL spectrum instead
        of the experimental spectrum. The experimental spectra are only binned,
        sqrt-transformed, and windowed before scoring.
        
        Args:
            theoretical_binned: Binned theoretical spectrum (binary: 0 or 1)
            
        Returns:
            Preprocessed theoretical spectrum ready for scoring
        """
        # Find highest bin with data
        highest_ion_bin = 0
        for i in range(len(theoretical_binned) - 1, -1, -1):
            if theoretical_binned[i] > 0:
                highest_ion_bin = i
                break
        
        # For theoretical spectra, treat as having intensity 1.0 at fragment positions
        highest_intensity = 1.0
        
        # Apply MakeCorrData windowing (though for binary theoretical, this is simpler)
        windowed = self._make_corr_data(theoretical_binned, highest_ion_bin, highest_intensity)
        
        # Apply Fast XCorr preprocessing
        preprocessed = self.preprocess_for_xcorr(windowed)
        
        return preprocessed
    
    def calculate_peptide_centric_xcorr(self, experimental_windowed: np.ndarray, 
                                       theoretical_preprocessed: np.ndarray) -> float:
        """
        Calculate XCorr for peptide-centric search.
        
        This is a convenience wrapper around calculate_xcorr() for peptide-centric mode:
        - Experimental spectrum: windowed only (NOT preprocessed with Fast XCorr)
        - Theoretical spectrum: fully preprocessed (windowed + Fast XCorr)
        - Scaling: 0.0001 (50x smaller than spectrum-centric due to preprocessing asymmetry)
        
        IMPORTANT: The scaling factor is different from spectrum-centric search!
        In peptide-centric mode, preprocessing the theoretical spectrum (instead of experimental)
        produces dot products that are ~50x larger. Therefore we use 0.0001 instead of 0.005.
        
        Args:
            experimental_windowed: Experimental spectrum after MakeCorrData windowing
            theoretical_preprocessed: Theoretical spectrum after full preprocessing
            
        Returns:
            XCorr score (scaled to match typical 0-10 range)
        """
        result = self.calculate_xcorr(experimental_windowed, theoretical_preprocessed, 
                                      scaling_factor=0.0001)
        return float(result)  # Guaranteed to be float for 1D inputs
    
    def search_dia_peptide_centric(self,
                                   spectra: List[MassSpectrum],
                                   target_decoy_pairs: List[Tuple[PeptideCandidate, PeptideCandidate]],
                                   charge_states: List[int] = [2, 3],
                                   parquet_output: str = None,
                                   library: 'SpectrumLibrary' = None,
                                   ms1_spectra: List[MS1Spectrum] = None,
                                   lib_fragment_tol_ppm: float = 10.0,
                                   lib_precursor_tol_ppm: float = 10.0,
                                   lib_fragment_tol_unit: str = 'ppm',
                                   lib_precursor_tol_unit: str = 'ppm',
                                   calibration: Dict = None,
                                   skip_xcorr_matrix: bool = False,
                                   verbose: int = 0) -> Dict:
        """
        Perform comprehensive peptide-centric DIA search.
        
        Key improvements:
        1. Score ALL spectra against ALL peptides in the isolation window
        2. Write XCorr chromatograms to Parquet file incrementally with unified schema
        3. Store all XCorr values for e-value calculation
        4. E-value: best peptide XCorr vs all its XCorr scores across spectra
        5. If library provided: LibCosine-centric scoring with target/decoy competition (report winner only)
        6. If no library: Track target/decoy pairs for downstream competition analysis
        7. If calibration provided: Apply RT filtering and adjusted m/z tolerances, calculate delta columns
        
        Args:
            spectra: List of spectra (should be from same isolation window)
            target_decoy_pairs: List of (target, decoy) peptide pairs
            charge_states: Charge states to search
            parquet_output: Path to write Parquet file (if None, skips parquet writing)
            library: Optional SpectrumLibrary for LibCosine scoring
            ms1_spectra: Optional MS1 spectra for precursor isotope scoring
            calibration: Optional calibration dict with ms1_calibration, ms2_calibration, rt_calibration
            skip_xcorr_matrix: If True and library mode, skip XCorr calculation entirely (for calibration)
            
        Returns:
            Dictionary with results per peptide (and path to Parquet file)
        """
        import time
        window_start_time = time.time()
        
        # QC data collection for quality control plots
        qc_data = {
            'ms1_mass_errors': [],  # List of mass errors (units depend on lib_precursor_tol_unit)
            'ms2_mass_errors': [],  # List of mass errors (units depend on lib_fragment_tol_unit)
            'rt_pairs': [],         # List of (lib_rt, measured_rt, lib_cosine, is_target) tuples
            'ms1_tol_unit': lib_precursor_tol_unit,  # 'ppm' or 'mz'
            'ms2_tol_unit': lib_fragment_tol_unit     # 'ppm' or 'mz'
        }
        
        if not spectra:
            return {'results': {}, 'parquet_file': None, 'qc_data': qc_data}
        
        # Get isolation window (same for all spectra in this group)
        isolation_window = (spectra[0].isolation_window_lower, spectra[0].isolation_window_upper)
        
        # Set up Parquet output file (skip if None - used during calibration)
        skip_parquet = parquet_output is None
        if not skip_parquet:
            # Only create default filename if not skipping
            window_str = f"{isolation_window[0]:.1f}_{isolation_window[1]:.1f}"
            if parquet_output == '':
                parquet_output = f"dia_chromatograms_window_{window_str}.parquet"
        
        # Find all peptides in this isolation window
        # Track target/decoy pairs for later linkage
        peptides_in_window = []  # List of (peptide, charge, is_target, pair_id, lib_rt)
        pair_id = 0
        unique_peptide_sequences = set()  # Track unique sequences for diagnostics
        peptides_filtered_by_rt = 0  # Count peptides filtered by RT calibration
        
        for target_peptide, decoy_peptide in target_decoy_pairs:
            # Check if peptide has a specific charge (calibration mode with library sampling)
            # In this case, use only that charge instead of iterating through charge_states
            if hasattr(target_peptide, 'charge') and target_peptide.charge is not None:
                charges_to_test = [target_peptide.charge]
            else:
                charges_to_test = charge_states
            
            for charge in charges_to_test:
                target_mz = (target_peptide.mass + charge * self.proton_mass) / charge
                
                if isolation_window[0] <= target_mz <= isolation_window[1]:
                    # Get library RT if available for RT filtering
                    lib_rt = None
                    if library is not None:
                        lib_data = library.get_precursor(target_peptide.sequence, charge)
                        if lib_data:
                            lib_rt = lib_data['rt']
                    
                    # Apply RT filtering if calibration available
                    skip_peptide = False
                    if calibration is not None and lib_rt is not None:
                        rt_cal = calibration.get('rt_calibration', {})
                        if rt_cal.get('model_type') is not None:
                            # Predict expected RT from library RT
                            predicted_rt = self.apply_rt_calibration(lib_rt, calibration)
                            rt_window = rt_cal.get('residual_sd', 0) * 3  # 3σ window
                            
                            # Check if any spectrum in this window falls within RT range
                            rt_range_min = predicted_rt - rt_window
                            rt_range_max = predicted_rt + rt_window
                            
                            # Check if any spectrum RT falls in range
                            has_spectrum_in_range = any(
                                rt_range_min <= s.retention_time <= rt_range_max 
                                for s in spectra if hasattr(s, 'retention_time')
                            )
                            
                            if not has_spectrum_in_range:
                                skip_peptide = True
                                peptides_filtered_by_rt += 1
                    
                    if not skip_peptide:
                        # Both target and decoy share the same pair_id for linkage
                        peptides_in_window.append((target_peptide, charge, True, pair_id, lib_rt))
                        peptides_in_window.append((decoy_peptide, charge, False, pair_id, lib_rt))
                        unique_peptide_sequences.add(target_peptide.sequence)
                        pair_id += 1
        
        # Preprocess all theoretical spectra once
        # Count targets and decoys
        n_targets = sum(1 for _, _, is_target, _, _ in peptides_in_window if is_target)
        n_decoys = len(peptides_in_window) - n_targets
        window_str = f"{isolation_window[0]:.1f}-{isolation_window[1]:.1f}"
        if verbose >= 1:
            n_unique_seqs = len(unique_peptide_sequences)
            n_precursors = pair_id
            rt_filter_msg = f" ({peptides_filtered_by_rt} filtered by RT)" if peptides_filtered_by_rt > 0 else ""
            print(f"  DIA: Preprocessing {len(peptides_in_window)} theoretical spectra from {n_unique_seqs} unique peptides ({n_precursors} precursors = peptide+charge combinations) for isolation window {window_str}{rt_filter_msg}")
        peptide_theoretical_preprocessed = {}
        
        for peptide, charge, is_target, pair_id, lib_rt in peptides_in_window:
            theoretical = self.generate_theoretical_spectrum(peptide, charge)
            preprocessed = self.preprocess_theoretical_spectrum(theoretical)
            peptide_theoretical_preprocessed[(peptide, charge)] = preprocessed
        
        # Preprocess all experimental spectra
        if verbose >= 1:
            print(f"  DIA: Preprocessing {len(spectra)} experimental spectra for isolation window {window_str}")
        experimental_preprocessed = []
        spectrum_metadata = []  # (scan_id, rt_minutes, spectrum_idx)
        
        for spectrum_idx, spectrum in enumerate(spectra):
            windowed = self.preprocess_spectrum(spectrum)
            experimental_preprocessed.append(windowed)
            
            # Extract RT in minutes from spectrum object
            rt_minutes = spectrum.retention_time if hasattr(spectrum, 'retention_time') else float(spectrum_idx)
            spectrum_metadata.append((spectrum.scan_id, rt_minutes, spectrum_idx))
        
        # **VECTORIZED SCORING**: Score ALL peptides against ALL spectra using matrix multiplication
        if verbose >= 1:
            print(f"  DIA: Scoring {len(peptides_in_window)} peptides vs {len(spectra)} spectra using vectorized matrix multiplication for isolation window {window_str}")
        
        # Skip if no peptides in window
        if len(peptides_in_window) == 0:
            if verbose >= 1:
                print(f"  DIA: No peptides in isolation window {window_str}, skipping")
            return {
                'results': {},
                'parquet_file': None,
                'qc_data': qc_data,
                'num_results': 0
            }
        
        # **XCORR MATRIX CALCULATION** (skip during calibration for speed)
        xcorr_matrix_raw = None
        if library is None or not skip_xcorr_matrix:
            # Stack all theoretical spectra into a matrix: (n_peptides, n_bins)
            theoretical_matrix = np.vstack([peptide_theoretical_preprocessed[(p, c)] 
                                            for p, c, t, pair_id, lib_rt in peptides_in_window])
            
            # Stack all experimental spectra into a matrix: (n_spectra, n_bins)
            experimental_matrix = np.vstack(experimental_preprocessed)
            
            # Matrix multiply: (n_peptides, n_spectra) in one operation using unified calculate_xcorr()
            # This replaces the nested loops with a single optimized BLAS call
            # Use peptide-centric scaling factor (0.0001 instead of 0.005)
            xcorr_result = self.calculate_xcorr(theoretical_matrix, experimental_matrix, 
                                                 scaling_factor=0.0001)
            # Type assertion: matrix inputs always return ndarray
            assert isinstance(xcorr_result, np.ndarray), "Matrix scoring should return ndarray"
            xcorr_matrix_raw: np.ndarray = xcorr_result
            
            # Show matrix scoring complete if verbose
            if verbose >= 1:
                print("  DIA: Matrix scoring complete, processing results...")
        elif verbose >= 1:
            print("  DIA: Skipping XCorr matrix calculation (calibration mode)")

        # **LIBRARY SCORING**: Score library fragments if library is provided
        lib_cosine_matrix_target = None
        lib_cosine_matrix_decoy = None

        if library is not None:
            if verbose >= 1:
                print(f"  DIA: Calculating library scores for {len(peptides_in_window)} peptides vs {len(spectra)} spectra")

            # Initialize matrices for library scores
            lib_cosine_matrix_target = np.zeros((len(peptides_in_window), len(spectra)))
            lib_cosine_matrix_decoy = np.zeros((len(peptides_in_window), len(spectra)))

            # Preprocess all experimental spectra once for library scoring
            # Store as sorted arrays for fast binary search matching
            experimental_preprocessed_lib = []
            for spectrum in spectra:
                # Sort by m/z for fast searching
                sorted_indices = np.argsort(spectrum.mz_array)
                sorted_mz = spectrum.mz_array[sorted_indices]
                sorted_intensity = spectrum.intensity_array[sorted_indices]
                
                # Apply SMZ preprocessing: sqrt(intensity) * mz^2
                # This is separate from XCorr preprocessing which uses sqrt only
                preprocessed_intensity = np.sqrt(sorted_intensity) * (sorted_mz ** 2)
                
                experimental_preprocessed_lib.append({
                    'mz': sorted_mz,
                    'intensity_preprocessed': preprocessed_intensity
                })

            # Calculate adjusted m/z tolerances if calibration available
            adjusted_precursor_tol_ppm = lib_precursor_tol_ppm
            adjusted_fragment_tol_ppm = lib_fragment_tol_ppm
            if calibration is not None:
                ms1_cal = calibration.get('ms1_calibration', {})
                ms2_cal = calibration.get('ms2_calibration', {})
                # Expand tolerance to mean + 3σ for both precursor and fragments
                if ms1_cal.get('mean_ppm') is not None and ms1_cal.get('sd_ppm') is not None:
                    adjusted_precursor_tol_ppm = abs(ms1_cal['mean_ppm']) + 3 * ms1_cal['sd_ppm']
                if ms2_cal.get('mean_ppm') is not None and ms2_cal.get('sd_ppm') is not None:
                    adjusted_fragment_tol_ppm = abs(ms2_cal['mean_ppm']) + 3 * ms2_cal['sd_ppm']
            
            # Score each peptide's library fragments
            for pep_idx, (peptide, charge, is_target, pair_id, lib_rt) in enumerate(peptides_in_window):
                # Get target library data
                target_lib_data = library.get_precursor(peptide.sequence, charge)

                if target_lib_data:
                    # Use precomputed preprocessed fragments (already normalized)
                    lib_preprocessed_normalized = target_lib_data['preprocessed_fragments']
                    
                    if len(lib_preprocessed_normalized) > 0:
                        # Get fragment m/z for peak matching
                        fragments = target_lib_data['fragments']
                        lib_mz = np.array([frag['mz'] for frag in fragments])
                        
                        # Store MS2 errors temporarily for each spectrum (will only keep best spectrum's errors)
                        temp_ms2_errors = {}  # spectrum_idx -> list of errors
                        
                        # Score against all spectra
                        for spec_idx, exp_data in enumerate(experimental_preprocessed_lib):
                            # Extract matching peaks using binary search with tolerance
                            matched_exp = []
                            matched_lib = []
                            spectrum_ms2_errors = []  # Errors for this spectrum
                            
                            for lib_idx, lib_mz_val in enumerate(lib_mz):
                                # Apply m/z correction if calibration available
                                corrected_lib_mz = lib_mz_val
                                if calibration is not None:
                                    ms2_cal = calibration.get('ms2_calibration', {})
                                    if ms2_cal.get('mean_ppm') is not None:
                                        corrected_lib_mz = lib_mz_val * (1 + ms2_cal['mean_ppm'] / 1e6)
                                
                                # Calculate tolerance window using adjusted tolerance
                                tol_da = corrected_lib_mz * adjusted_fragment_tol_ppm / 1e6
                                mz_min = corrected_lib_mz - tol_da
                                mz_max = corrected_lib_mz + tol_da
                                
                                # Binary search for matching peaks
                                left_idx = np.searchsorted(exp_data['mz'], mz_min, side='left')
                                right_idx = np.searchsorted(exp_data['mz'], mz_max, side='right')
                                
                                # Find best matching peak in tolerance window
                                if left_idx < right_idx:
                                    window_mz = exp_data['mz'][left_idx:right_idx]
                                    window_intensity = exp_data['intensity_preprocessed'][left_idx:right_idx]
                                    
                                    # Use closest m/z match
                                    best_idx = np.argmin(np.abs(window_mz - lib_mz_val))
                                    matched_exp.append(window_intensity[best_idx])
                                    matched_lib.append(lib_preprocessed_normalized[lib_idx])
                                    
                                    # Store MS2 error temporarily (will only add to qc_data at best spectrum)
                                    matched_mz = window_mz[best_idx]
                                    if lib_fragment_tol_unit == 'ppm':
                                        ms2_mass_error = (matched_mz - lib_mz_val) / lib_mz_val * 1e6  # PPM
                                    else:
                                        ms2_mass_error = matched_mz - lib_mz_val  # m/z
                                    spectrum_ms2_errors.append({
                                        'error': ms2_mass_error,
                                        'peptide': peptide.sequence,
                                        'charge': charge,
                                        'is_target': is_target
                                    })
                            
                            # Store errors for this spectrum
                            if len(spectrum_ms2_errors) > 0:
                                temp_ms2_errors[spec_idx] = spectrum_ms2_errors
                            
                            # Calculate cosine similarity
                            if len(matched_exp) > 0:
                                matched_exp = np.array(matched_exp)
                                matched_lib = np.array(matched_lib)
                                
                                exp_norm = np.linalg.norm(matched_exp)
                                if exp_norm > 0:
                                    matched_exp_normalized = matched_exp / exp_norm
                                    lib_cosine_matrix_target[pep_idx, spec_idx] = np.dot(matched_exp_normalized, matched_lib)
                        
                        # Store temp errors for later (will only add best spectrum's errors after finding best LibCosine)
                        # Use pep_idx as key to retrieve later
                        if not hasattr(self, '_temp_ms2_errors_target'):
                            self._temp_ms2_errors_target = {}
                        self._temp_ms2_errors_target[pep_idx] = temp_ms2_errors

                    # Generate and score decoy fragments
                    decoy_lib_data = library.generate_decoy_fragments(peptide.sequence, charge, self)

                    if decoy_lib_data:
                        # Use precomputed preprocessed fragments (already normalized)
                        lib_preprocessed_decoy_normalized = decoy_lib_data['preprocessed_fragments']
                        
                        if len(lib_preprocessed_decoy_normalized) > 0:
                            # Get fragment m/z for peak matching
                            fragments_decoy = decoy_lib_data['fragments']
                            lib_mz_decoy = np.array([frag['mz'] for frag in fragments_decoy])
                            
                            # Store MS2 errors temporarily for each spectrum (will only keep best spectrum's errors)
                            temp_ms2_errors_decoy = {}  # spectrum_idx -> list of errors
                            
                            # Score against all spectra
                            for spec_idx, exp_data in enumerate(experimental_preprocessed_lib):
                                # Extract matching peaks
                                matched_exp = []
                                matched_lib = []
                                spectrum_ms2_errors = []  # Errors for this spectrum
                                
                                for lib_idx, lib_mz_val in enumerate(lib_mz_decoy):
                                    # Apply m/z correction if calibration available
                                    corrected_lib_mz = lib_mz_val
                                    if calibration is not None:
                                        ms2_cal = calibration.get('ms2_calibration', {})
                                        if ms2_cal.get('mean_ppm') is not None:
                                            corrected_lib_mz = lib_mz_val * (1 + ms2_cal['mean_ppm'] / 1e6)
                                    
                                    tol_da = corrected_lib_mz * adjusted_fragment_tol_ppm / 1e6
                                    mz_min = corrected_lib_mz - tol_da
                                    mz_max = corrected_lib_mz + tol_da
                                    
                                    left_idx = np.searchsorted(exp_data['mz'], mz_min, side='left')
                                    right_idx = np.searchsorted(exp_data['mz'], mz_max, side='right')
                                    
                                    if left_idx < right_idx:
                                        window_mz = exp_data['mz'][left_idx:right_idx]
                                        window_intensity = exp_data['intensity_preprocessed'][left_idx:right_idx]
                                        
                                        best_idx = np.argmin(np.abs(window_mz - lib_mz_val))
                                        matched_exp.append(window_intensity[best_idx])
                                        matched_lib.append(lib_preprocessed_decoy_normalized[lib_idx])
                                        
                                        # Store MS2 error temporarily (will only add to qc_data at best spectrum)
                                        matched_mz = window_mz[best_idx]
                                        if lib_fragment_tol_unit == 'ppm':
                                            ms2_mass_error = (matched_mz - lib_mz_val) / lib_mz_val * 1e6  # PPM
                                        else:
                                            ms2_mass_error = matched_mz - lib_mz_val  # m/z
                                        spectrum_ms2_errors.append({
                                            'error': ms2_mass_error,
                                            'peptide': peptide.sequence,
                                            'charge': charge,
                                            'is_target': False
                                        })
                                
                                # Store errors for this spectrum
                                if len(spectrum_ms2_errors) > 0:
                                    temp_ms2_errors_decoy[spec_idx] = spectrum_ms2_errors
                                
                                if len(matched_exp) > 0:
                                    matched_exp = np.array(matched_exp)
                                    matched_lib = np.array(matched_lib)
                                    
                                    exp_norm = np.linalg.norm(matched_exp)
                                    if exp_norm > 0:
                                        matched_exp_normalized = matched_exp / exp_norm
                                        lib_cosine_matrix_decoy[pep_idx, spec_idx] = np.dot(matched_exp_normalized, matched_lib)
                            
                            # Store temp errors for later (will only add best spectrum's errors after finding best LibCosine)
                            if not hasattr(self, '_temp_ms2_errors_decoy'):
                                self._temp_ms2_errors_decoy = {}
                            self._temp_ms2_errors_decoy[pep_idx] = temp_ms2_errors_decoy

            if verbose >= 1:
                print("  DIA: Library scoring complete")

        peptide_results = {}  # Track results by pair_id for paired output
        
        # PAIRED TARGET/DECOY OUTPUT (no competition during search):
        # - With library: LibCosine determines scan for target and decoy separately
        # - Without library: XCorr determines scan for target and decoy separately
        # Competition will be performed later during analysis
        
        if verbose >= 1:
            print("  DIA: Storing paired target/decoy results (competition deferred to analysis)")
        
        # Process pairs (every two peptides form a target/decoy pair)
        for pair_idx in range(0, len(peptides_in_window), 2):
            if pair_idx + 1 >= len(peptides_in_window):
                break  # Should not happen with proper pairing
            
            # Get target and decoy from pair
            target_peptide, target_charge, target_is_target, target_pair_id, target_lib_rt = peptides_in_window[pair_idx]
            decoy_peptide, decoy_charge, decoy_is_target, decoy_pair_id, decoy_lib_rt = peptides_in_window[pair_idx + 1]
            
            # Verify this is a proper target/decoy pair
            if not target_is_target or decoy_is_target or target_pair_id != decoy_pair_id:
                print(f"Warning: Improper pairing at index {pair_idx}")
                continue
            
            # Get XCorr series for both (if calculated)
            if xcorr_matrix_raw is not None:
                target_xcorr = xcorr_matrix_raw[pair_idx, :].tolist()
                decoy_xcorr = xcorr_matrix_raw[pair_idx + 1, :].tolist()
            else:
                # Library-only mode (calibration): no XCorr calculated
                target_xcorr = [0.0] * len(spectra)
                decoy_xcorr = [0.0] * len(spectra)
            
            # Process target and decoy independently (no competition)
            if library is not None:
                # Library mode: LibCosine determines scan for each peptide
                target_lib = lib_cosine_matrix_target[pair_idx, :].tolist()
                decoy_lib = lib_cosine_matrix_decoy[pair_idx, :].tolist()
                
                # Find max LibCosine for target
                target_lib_cosine = max(target_lib)
                target_primary_idx = target_lib.index(target_lib_cosine)
                
                # Find max LibCosine for decoy
                decoy_lib_cosine = max(decoy_lib)
                decoy_primary_idx = decoy_lib.index(decoy_lib_cosine)
            else:
                # No library: XCorr determines scan for each peptide
                target_primary_score = max(target_xcorr)
                target_primary_idx = target_xcorr.index(target_primary_score)
                
                decoy_primary_score = max(decoy_xcorr)
                decoy_primary_idx = decoy_xcorr.index(decoy_primary_score)
                
                target_lib = None
                decoy_lib = None
            
            # **PROCESS TARGET**
            target_scan_id, target_rt, target_spec_idx = spectrum_metadata[target_primary_idx]
            target_xcorr_at_peak = target_xcorr[target_primary_idx]
            
            # Add MS2 errors from best spectrum only (target)
            if library is not None and hasattr(self, '_temp_ms2_errors_target'):
                if pair_idx in self._temp_ms2_errors_target:
                    temp_errors = self._temp_ms2_errors_target[pair_idx]
                    if target_spec_idx in temp_errors:
                        qc_data['ms2_mass_errors'].extend(temp_errors[target_spec_idx])
            
            # Calculate target e-value (non-library mode only)
            if library is None:
                temp_engine = FastXCorr()
                target_e_value = temp_engine.calculate_e_value(target_xcorr, target_xcorr_at_peak)
            else:
                target_e_value = 0.0
            
            # Calculate delta_rt if library RT available
            target_delta_rt = None
            if target_lib_rt is not None:
                target_delta_rt = target_rt - target_lib_rt
            
            # Calculate target precursor cosine (library mode with MS1)
            target_precursor_cosine = 0.0
            target_delta_mz_ppm_precursor = None
            if library is not None and ms1_spectra is not None:
                closest_ms1 = FastXCorr.find_closest_ms1(ms1_spectra, target_rt)
                if closest_ms1 is not None:
                    lib_data = library.get_precursor(target_peptide.sequence, target_charge)
                    if lib_data:
                        # Apply m/z correction if calibration available
                        corrected_precursor_mz = lib_data['precursor_mz']
                        if calibration is not None:
                            ms1_cal = calibration.get('ms1_calibration', {})
                            if ms1_cal.get('mean_ppm') is not None:
                                corrected_precursor_mz = lib_data['precursor_mz'] * (1 + ms1_cal['mean_ppm'] / 1e6)
                        
                        # Collect QC data: MS1 M+0 mass error
                        experimental_isotopes, m0_mass_error = FastXCorr.extract_isotope_envelope(
                            closest_ms1,
                            corrected_precursor_mz,
                            target_charge,
                            adjusted_precursor_tol_ppm,
                            collect_qc=True
                        )
                        if m0_mass_error is not None:
                            # Calculate delta_mz_ppm_precursor (always in PPM)
                            target_delta_mz_ppm_precursor = m0_mass_error / lib_data['precursor_mz'] * 1e6
                            
                            # Convert to appropriate unit for QC data
                            if lib_precursor_tol_unit == 'ppm':
                                m0_mass_error_final = target_delta_mz_ppm_precursor  # PPM
                            else:
                                m0_mass_error_final = m0_mass_error  # m/z
                            qc_data['ms1_mass_errors'].append({
                                'error': m0_mass_error_final,
                                'peptide': target_peptide.sequence,
                                'charge': target_charge,
                                'is_target': True
                            })
                        
                        theoretical_isotopes = FastXCorr.predict_isotope_pattern(
                            target_peptide.sequence, target_charge, self.aa_masses
                        )
                        target_precursor_cosine = FastXCorr.calculate_cosine_angle(
                            experimental_isotopes, theoretical_isotopes
                        )
            
            # Calculate target Z-scores
            target_xcorr_zscore = 0.0
            if len(target_xcorr) > 1:
                xcorr_mean = np.mean(target_xcorr)
                xcorr_std = np.std(target_xcorr, ddof=1)
                if xcorr_std > 0:
                    target_xcorr_zscore = (target_xcorr_at_peak - xcorr_mean) / xcorr_std
            
            target_lib_zscore = 0.0
            if target_lib is not None and len(target_lib) > 1:
                lib_mean = np.mean(target_lib)
                lib_std = np.std(target_lib, ddof=1)
                if lib_std > 0:
                    target_lib_zscore = (target_lib_cosine - lib_mean) / lib_std
            
            # **PROCESS DECOY**
            decoy_scan_id, decoy_rt, decoy_spec_idx = spectrum_metadata[decoy_primary_idx]
            decoy_xcorr_at_peak = decoy_xcorr[decoy_primary_idx]
            
            # Calculate delta_rt for decoy if library RT available
            decoy_delta_rt = None
            if decoy_lib_rt is not None:
                decoy_delta_rt = decoy_rt - decoy_lib_rt
            
            # Add MS2 errors from best spectrum only (decoy)
            if library is not None and hasattr(self, '_temp_ms2_errors_decoy'):
                if pair_idx + 1 in self._temp_ms2_errors_decoy:
                    temp_errors = self._temp_ms2_errors_decoy[pair_idx + 1]
                    if decoy_spec_idx in temp_errors:
                        qc_data['ms2_mass_errors'].extend(temp_errors[decoy_spec_idx])
            
            # Calculate decoy e-value (non-library mode only)
            if library is None:
                decoy_e_value = temp_engine.calculate_e_value(decoy_xcorr, decoy_xcorr_at_peak)
            else:
                decoy_e_value = 0.0
            
            # Calculate decoy precursor cosine (library mode with MS1)
            decoy_precursor_cosine = 0.0
            if library is not None and ms1_spectra is not None:
                closest_ms1 = FastXCorr.find_closest_ms1(ms1_spectra, decoy_rt)
                if closest_ms1 is not None:
                    # Use target sequence for isotope pattern (same mass)
                    lib_data = library.get_precursor(target_peptide.sequence, target_charge)
                    if lib_data:
                        # Collect QC data: MS1 M+0 mass error
                        experimental_isotopes, m0_mass_error = FastXCorr.extract_isotope_envelope(
                            closest_ms1,
                            lib_data['precursor_mz'],
                            target_charge,
                            lib_precursor_tol_ppm,
                            collect_qc=True
                        )
                        if m0_mass_error is not None:
                            # Convert to PPM if needed
                            if lib_precursor_tol_unit == 'ppm':
                                m0_mass_error_final = m0_mass_error / lib_data['precursor_mz'] * 1e6  # PPM
                            else:
                                m0_mass_error_final = m0_mass_error  # m/z
                            qc_data['ms1_mass_errors'].append({
                                'error': m0_mass_error_final,
                                'peptide': target_peptide.sequence,
                                'charge': target_charge,
                                'is_target': False
                            })
                        
                        theoretical_isotopes = FastXCorr.predict_isotope_pattern(
                            target_peptide.sequence, target_charge, self.aa_masses
                        )
                        decoy_precursor_cosine = FastXCorr.calculate_cosine_angle(
                            experimental_isotopes, theoretical_isotopes
                        )
            
            # Calculate decoy Z-scores
            decoy_xcorr_zscore = 0.0
            if len(decoy_xcorr) > 1:
                xcorr_mean = np.mean(decoy_xcorr)
                xcorr_std = np.std(decoy_xcorr, ddof=1)
                if xcorr_std > 0:
                    decoy_xcorr_zscore = (decoy_xcorr_at_peak - xcorr_mean) / xcorr_std
            
            decoy_lib_zscore = 0.0
            if decoy_lib is not None and len(decoy_lib) > 1:
                lib_mean = np.mean(decoy_lib)
                lib_std = np.std(decoy_lib, ddof=1)
                if lib_std > 0:
                    decoy_lib_zscore = (decoy_lib_cosine - lib_mean) / lib_std
            
            # Calculate delta_mz_ppm_fragments for target (average MS2 error from best spectrum)
            target_delta_mz_ppm_fragments = None
            if library is not None and hasattr(self, '_temp_ms2_errors_target'):
                if pair_idx in self._temp_ms2_errors_target:
                    temp_errors = self._temp_ms2_errors_target[pair_idx]
                    if target_spec_idx in temp_errors:
                        # Average of all fragment errors (already in PPM)
                        errors = [e['error'] for e in temp_errors[target_spec_idx]]
                        if len(errors) > 0:
                            target_delta_mz_ppm_fragments = np.mean(errors)
            
            # Calculate delta_mz_ppm_fragments for decoy (average MS2 error from best spectrum)
            decoy_delta_mz_ppm_fragments = None
            if library is not None and hasattr(self, '_temp_ms2_errors_decoy'):
                if pair_idx + 1 in self._temp_ms2_errors_decoy:
                    temp_errors = self._temp_ms2_errors_decoy[pair_idx + 1]
                    if decoy_spec_idx in temp_errors:
                        # Average of all fragment errors (already in PPM)
                        errors = [e['error'] for e in temp_errors[decoy_spec_idx]]
                        if len(errors) > 0:
                            decoy_delta_mz_ppm_fragments = np.mean(errors)
            
            # Calculate delta_mz_ppm_precursor for decoy
            decoy_delta_mz_ppm_precursor = None
            if library is not None and ms1_spectra is not None:
                closest_ms1_decoy = FastXCorr.find_closest_ms1(ms1_spectra, decoy_rt)
                if closest_ms1_decoy is not None:
                    lib_data_decoy = library.get_precursor(target_peptide.sequence, target_charge)
                    if lib_data_decoy:
                        # Apply m/z correction if calibration available
                        corrected_precursor_mz_decoy = lib_data_decoy['precursor_mz']
                        if calibration is not None:
                            ms1_cal = calibration.get('ms1_calibration', {})
                            if ms1_cal.get('mean_ppm') is not None:
                                corrected_precursor_mz_decoy = lib_data_decoy['precursor_mz'] * (1 + ms1_cal['mean_ppm'] / 1e6)
                        
                        experimental_isotopes_decoy, m0_mass_error_decoy = FastXCorr.extract_isotope_envelope(
                            closest_ms1_decoy,
                            corrected_precursor_mz_decoy,
                            target_charge,
                            adjusted_precursor_tol_ppm,
                            collect_qc=True
                        )
                        
                        if m0_mass_error_decoy is not None:
                            decoy_delta_mz_ppm_precursor = m0_mass_error_decoy / lib_data_decoy['precursor_mz'] * 1e6
            
            # Store paired results
            target_result_dict = {
                'peptide': target_peptide,
                'charge': target_charge,
                'best_xcorr': target_xcorr_at_peak,
                'best_rt': target_rt,
                'best_scan': target_scan_id,
                'isolation_window': isolation_window,
                'e_value': target_e_value,
                'num_spectra_scored': len(target_xcorr),
                'xcorr_zscore': target_xcorr_zscore,
                'delta_rt': target_delta_rt,
                'delta_mz_ppm_precursor': target_delta_mz_ppm_precursor,
                'delta_mz_ppm_fragments': target_delta_mz_ppm_fragments,
            }
            
            decoy_result_dict = {
                'peptide': decoy_peptide,
                'charge': target_charge,
                'best_xcorr': decoy_xcorr_at_peak,
                'best_rt': decoy_rt,
                'best_scan': decoy_scan_id,
                'isolation_window': isolation_window,
                'e_value': decoy_e_value,
                'num_spectra_scored': len(decoy_xcorr),
                'xcorr_zscore': decoy_xcorr_zscore,
                'delta_rt': decoy_delta_rt,
                'delta_mz_ppm_precursor': decoy_delta_mz_ppm_precursor,
                'delta_mz_ppm_fragments': decoy_delta_mz_ppm_fragments,
            }
            
            # Add library-specific fields
            if library is not None:
                target_result_dict['best_lib_cosine_target'] = target_lib_cosine
                target_result_dict['lib_cosine_target_zscore'] = target_lib_zscore
                target_result_dict['precursor_cosine_target'] = target_precursor_cosine
                
                decoy_result_dict['best_lib_cosine_decoy'] = decoy_lib_cosine
                decoy_result_dict['lib_cosine_decoy_zscore'] = decoy_lib_zscore
                decoy_result_dict['precursor_cosine_decoy'] = decoy_precursor_cosine
                
                # Collect QC data: RT pairs for library mode (targets only)
                # Get library RT for this peptide from the library
                lib_data = library.get_precursor(target_peptide.sequence, target_charge)
                if lib_data and 'rt' in lib_data:
                    library_rt = lib_data['rt']
                    # Store RT pair for target only (decoys don't have meaningful library RTs)
                    qc_data['rt_pairs'].append({
                        'library_rt': library_rt,
                        'measured_rt': target_rt,
                        'lib_cosine': target_lib_cosine,
                        'is_target': True,
                        'peptide': target_peptide.sequence,
                        'charge': target_charge
                    })
            
            # Store paired results by pair_id
            peptide_results[target_pair_id] = {
                'target': target_result_dict,
                'decoy': decoy_result_dict
            }
        
        # Show completion at verbose=0 (always show, once per window)
        window_elapsed = time.time() - window_start_time
        pairs_or_peptides = len(peptides_in_window) // 2 if library is not None else len(peptides_in_window)
        print(f"  DIA: Completed window {window_str}: {pairs_or_peptides} {'pairs' if library is not None else 'peptides'} processed in {window_elapsed/60:.2f} min")

        return {
            'results': peptide_results,
            'parquet_file': None if skip_parquet else parquet_output,
            'num_spectra': len(spectra),
            'num_peptides': len(peptides_in_window),
            'isolation_window': isolation_window,
            'qc_data': qc_data
        }

    @staticmethod
    def predict_isotope_pattern(sequence: str, charge: int, aa_masses: Dict[str, float]) -> np.ndarray:
        """
        Predict isotope pattern intensities for M-1, M+0, M+1, M+2, M+3.

        Uses averagine model for rapid isotope distribution estimation.

        Args:
            sequence: Peptide sequence
            charge: Precursor charge state
            aa_masses: Amino acid masses dict (with modifications applied)

        Returns:
            np.array of 5 intensities [M-1, M+0, M+1, M+2, M+3], normalized to sum=1
            M-1 is always 0.0 (theoretical)
        """
        # Calculate peptide mass
        mass = sum(aa_masses.get(aa, 0) for aa in sequence)
        mass += 18.010565  # Add H2O for neutral mass

        # Simple averagine-based isotope distribution
        # Averagine approximation: C4.9384 H7.7583 N1.3577 O1.4773 S0.0417 (110.5 Da)
        # For most peptides, use empirical model based on mass

        # Calculate number of carbon atoms (approximate)
        num_carbons = mass / 110.5 * 4.9384

        # Natural isotope abundance: C13 = 1.07%, N15 = 0.37%
        # For simplicity, use carbon contribution (dominant)
        p_c13 = 0.0107  # Probability of C13

        # Binomial distribution for isotope peaks
        # M+0: all C12
        # M+1: one C13
        # M+2: two C13
        # M+3: three C13

        from scipy.special import comb

        intensities = np.zeros(5)
        intensities[0] = 0.0  # M-1 is always 0

        # Calculate probabilities using binomial distribution
        p_c12 = 1 - p_c13
        for k in range(4):  # M+0, M+1, M+2, M+3
            intensities[k + 1] = comb(num_carbons, k, exact=False) * (p_c13 ** k) * (p_c12 ** (num_carbons - k))

        # Normalize (excluding M-1 which is 0)
        total = intensities[1:].sum()
        if total > 0:
            intensities[1:] /= total

        return intensities

    @staticmethod
    def calculate_isotope_mz_values(precursor_mz: float, charge: int) -> np.ndarray:
        """
        Calculate m/z values for M-1, M+0, M+1, M+2, M+3 isotope peaks.

        Args:
            precursor_mz: Monoisotopic precursor m/z (M+0)
            charge: Precursor charge state

        Returns:
            np.array of 5 m/z values [M-1, M+0, M+1, M+2, M+3]
        """
        neutron_mass = 1.002868  # Mass difference between isotopes. This came from Devin Schweppe.
        isotope_gap = neutron_mass / charge

        mz_values = np.array([
            precursor_mz - isotope_gap,    # M-1
            precursor_mz,                  # M+0 (monoisotopic)
            precursor_mz + isotope_gap,    # M+1
            precursor_mz + 2 * isotope_gap, # M+2
            precursor_mz + 3 * isotope_gap  # M+3
        ])

        return mz_values

    @staticmethod
    def extract_isotope_envelope(ms1_spectrum: MS1Spectrum, precursor_mz: float,
                                 charge: int, tolerance_ppm: float = 10.0, 
                                 collect_qc: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[float]]]:
        """
        Extract isotope envelope intensities from MS1 spectrum.

        Args:
            ms1_spectrum: MS1Spectrum object
            precursor_mz: Monoisotopic precursor m/z
            charge: Precursor charge state
            tolerance_ppm: m/z matching tolerance in ppm
            collect_qc: If True, return (intensities, m0_mass_error) for QC plots

        Returns:
            If collect_qc=False: np.array of 5 extracted intensities [M-1, M+0, M+1, M+2, M+3]
            If collect_qc=True: (intensities, m0_mass_error) where m0_mass_error is observed - theoretical for M+0 peak with highest intensity
            Returns 0 for unmatched peaks, None for m0_mass_error if no match
        """
        # Calculate expected m/z values
        expected_mz = FastXCorr.calculate_isotope_mz_values(precursor_mz, charge)

        # Extract intensities
        extracted_intensities = np.zeros(5)
        m0_mass_error = None
        m0_best_mz = None
        m0_best_intensity = 0.0

        for i, target_mz in enumerate(expected_mz):
            # Find matching peak in MS1
            matched_intensity = 0.0
            matched_mz = None

            for j, obs_mz in enumerate(ms1_spectrum.mz_array):
                # Calculate ppm error
                ppm_error = abs(obs_mz - target_mz) / target_mz * 1e6

                if ppm_error <= tolerance_ppm:
                    # Match found
                    if ms1_spectrum.intensity_array[j] > matched_intensity:
                        matched_intensity = ms1_spectrum.intensity_array[j]
                        matched_mz = obs_mz

            extracted_intensities[i] = matched_intensity
            
            # Track M+0 peak (index 1) for QC
            if collect_qc and i == 1 and matched_mz is not None and matched_intensity > m0_best_intensity:
                m0_best_intensity = matched_intensity
                m0_best_mz = matched_mz
                m0_mass_error = matched_mz - target_mz  # Delta m/z

        if collect_qc:
            return extracted_intensities, m0_mass_error
        return extracted_intensities

    @staticmethod
    def calculate_cosine_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine angle between two vectors.

        Formula: cos(θ) = dot(v1, v2) / (||v1|| * ||v2||)

        Args:
            vec1: First vector (e.g., experimental intensities)
            vec2: Second vector (e.g., library intensities)

        Returns:
            Cosine angle (0 to 1), or 0 if either vector has zero norm
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)

        # Clamp to [0, 1] to handle numerical errors
        return max(0.0, min(1.0, cos_angle))

    @staticmethod
    def match_fragments_ppm(experimental_mz: np.ndarray, experimental_intensity: np.ndarray,
                           library_fragments: List[Dict], tolerance_ppm: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match experimental fragments to library fragments using ppm tolerance.

        Args:
            experimental_mz: Experimental m/z array
            experimental_intensity: Experimental intensity array
            library_fragments: List of library fragment dicts with 'mz' and 'intensity'
            tolerance_ppm: m/z matching tolerance in ppm

        Returns:
            (matched_exp_intensities, matched_lib_intensities)
            Arrays of matched intensities in corresponding order
        """
        matched_exp = []
        matched_lib = []

        for lib_frag in library_fragments:
            lib_mz = lib_frag['mz']
            lib_intensity = lib_frag['intensity']

            # Find best matching experimental peak
            best_intensity = 0.0

            for exp_mz, exp_intensity in zip(experimental_mz, experimental_intensity):
                ppm_error = abs(exp_mz - lib_mz) / lib_mz * 1e6

                if ppm_error <= tolerance_ppm:
                    best_intensity = max(best_intensity, exp_intensity)

            # Include this fragment (0 intensity if not matched)
            matched_exp.append(best_intensity)
            matched_lib.append(lib_intensity)

        return np.array(matched_exp), np.array(matched_lib)

    @staticmethod
    def calculate_library_cosine_score(experimental_mz: np.ndarray, experimental_intensity: np.ndarray,
                                      library_fragments: List[Dict], tolerance_ppm: float = 10.0) -> float:
        """
        Calculate library cosine score with SMZ preprocessing.

        SMZ preprocessing: preprocessed_intensity = sqrt(intensity) * mz^2

        Args:
            experimental_mz: Experimental m/z array
            experimental_intensity: Experimental intensity array
            library_fragments: List of library fragment dicts
            tolerance_ppm: Fragment m/z tolerance in ppm

        Returns:
            Cosine angle score (0 to 1)
        """
        # Match fragments
        matched_exp, matched_lib = FastXCorr.match_fragments_ppm(
            experimental_mz, experimental_intensity, library_fragments, tolerance_ppm
        )

        if len(matched_exp) == 0:
            return 0.0

        # Get corresponding m/z values for matched fragments
        matched_mz = np.array([frag['mz'] for frag in library_fragments])

        # Apply SMZ preprocessing: sqrt(intensity) * mz^2
        exp_preprocessed = np.sqrt(matched_exp) * (matched_mz ** 2)
        lib_preprocessed = np.sqrt(matched_lib) * (matched_mz ** 2)

        # Calculate cosine angle
        return FastXCorr.calculate_cosine_angle(exp_preprocessed, lib_preprocessed)

    def calculate_mz_calibration(self, qc_data: Dict) -> Dict:
        """
        Calculate m/z calibration parameters from QC data.
        
        Args:
            qc_data: QC data dictionary with ms1_mass_errors and ms2_mass_errors
                     (can be either list of dicts with 'error' key or list of scalar values)
            
        Returns:
            Dictionary with MS1 and MS2 calibration parameters
        """
        # Handle both formats: list of dicts or list of scalars
        ms1_list = qc_data['ms1_mass_errors']
        ms2_list = qc_data['ms2_mass_errors']
        
        # Extract errors if they're in dict format, otherwise use directly
        # Check for dict by trying to access 'error' key rather than isinstance
        try:
            if len(ms1_list) > 0:
                # Try to access as dict
                _ = ms1_list[0]['error']
                ms1_errors = np.array([e['error'] for e in ms1_list])
            else:
                ms1_errors = np.array([])
        except (TypeError, KeyError, IndexError):
            # Not a dict or not indexable, use directly
            ms1_errors = np.array(ms1_list)
            
        try:
            if len(ms2_list) > 0:
                # Try to access as dict
                _ = ms2_list[0]['error']
                ms2_errors = np.array([e['error'] for e in ms2_list])
            else:
                ms2_errors = np.array([])
        except (TypeError, KeyError, IndexError):
            # Not a dict or not indexable, use directly
            ms2_errors = np.array(ms2_list)
        
        ms1_unit = qc_data['ms1_tol_unit']
        ms2_unit = qc_data['ms2_tol_unit']
        
        # Calculate statistics
        ms1_mean = np.mean(ms1_errors) if len(ms1_errors) > 0 else 0.0
        ms1_sd = np.std(ms1_errors, ddof=1) if len(ms1_errors) > 1 else 0.0
        
        ms2_mean = np.mean(ms2_errors) if len(ms2_errors) > 0 else 0.0
        ms2_sd = np.std(ms2_errors, ddof=1) if len(ms2_errors) > 1 else 0.0
        
        return {
            'ms1_mean': ms1_mean,
            'ms1_sd': ms1_sd,
            'ms1_unit': ms1_unit,
            'ms2_mean': ms2_mean,
            'ms2_sd': ms2_sd,
            'ms2_unit': ms2_unit
        }
    
    def fit_rt_calibration(self, rt_pairs: List[Dict]) -> Dict:
        """
        Fit RT calibration model using LOESS regression with fallback to linear.
        
        Args:
            rt_pairs: List of {library_rt, measured_rt, lib_cosine, is_target, ...} dicts
            
        Returns:
            Dictionary with RT calibration model and residual SD
        """
        if len(rt_pairs) < 50:
            print(f"Warning: Only {len(rt_pairs)} RT pairs for calibration. Using linear regression.")
            method = 'linear'
        elif lowess is None:
            print("Warning: statsmodels not available. Using linear regression for RT calibration.")
            method = 'linear'
        else:
            method = 'loess'
        
        library_rts = np.array([p['library_rt'] for p in rt_pairs])
        measured_rts = np.array([p['measured_rt'] for p in rt_pairs])
        
        if method == 'loess':
            try:
                # LOESS smoothing (frac=0.3 for local smoothing)
                smoothed = lowess(measured_rts, library_rts, frac=0.3, return_sorted=True)
                
                # Calculate R² and RMSE
                predicted_rts = np.interp(library_rts, smoothed[:, 0], smoothed[:, 1])
                residuals = measured_rts - predicted_rts
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((measured_rts - np.mean(measured_rts)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                rmse = np.sqrt(np.mean(residuals ** 2))
                
                # Store LOESS model as interpolation points
                model_params = {
                    'library_rts': smoothed[:, 0].tolist(),
                    'predicted_rts': smoothed[:, 1].tolist()
                }
                
                # Check if LOESS fit is acceptable (R² >= 0.7)
                if r_squared < 0.7:
                    print(f"Warning: LOESS fit R²={r_squared:.3f} < 0.7. Falling back to linear regression.")
                    method = 'linear'
                else:
                    residual_sd = np.std(residuals, ddof=1)
                    
                    return {
                        'method': method,
                        'r_squared': r_squared,
                        'rmse': rmse,
                        'residual_sd': residual_sd,
                        'model_params': model_params
                    }
            except Exception as e:
                print(f"Warning: LOESS fitting failed: {e}. Falling back to linear regression.")
                method = 'linear'
        
        # Linear regression fallback
        if method == 'linear':
            slope, intercept, r_value, p_value, std_err = stats.linregress(library_rts, measured_rts)
            predicted_rts = slope * library_rts + intercept
            residuals = measured_rts - predicted_rts
            rmse = np.sqrt(np.mean(residuals ** 2))
            residual_sd = np.std(residuals, ddof=1)
            
            return {
                'method': 'linear',
                'r_squared': r_value ** 2,
                'rmse': rmse,
                'residual_sd': residual_sd,
                'model_params': {
                    'slope': slope,
                    'intercept': intercept
                }
            }
    
    def apply_rt_calibration(self, library_rt: float, calibration: Dict) -> float:
        """
        Apply RT calibration model to predict expected RT.
        
        Args:
            library_rt: Library retention time
            calibration: RT calibration dictionary from fit_rt_calibration()
            
        Returns:
            Predicted RT in data
        """
        method = calibration['method']
        params = calibration['model_params']
        
        if method == 'loess':
            # Interpolate using LOESS model
            library_rts = np.array(params['library_rts'])
            predicted_rts = np.array(params['predicted_rts'])
            predicted_rt = np.interp(library_rt, library_rts, predicted_rts)
        else:  # linear
            predicted_rt = params['slope'] * library_rt + params['intercept']
        
        return predicted_rt
    
    @staticmethod
    def save_calibration_json(calibration_params: Dict, output_path: str):
        """
        Save calibration parameters to JSON file.
        
        Args:
            calibration_params: Complete calibration dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(calibration_params, f, indent=2)
    
    @staticmethod
    def load_calibration_json(input_path: str) -> Dict:
        """
        Load calibration parameters from JSON file.
        
        Args:
            input_path: Path to calibration JSON file
            
        Returns:
            Calibration parameters dictionary
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_calibration_filename(output_path: str) -> str:
        """Generate calibration JSON filename from output path."""
        base = os.path.splitext(output_path)[0]
        return f"{base}.calibration.json"


class PepXMLWriter:
    """Class to write results in pepXML format."""
    
    def __init__(self, output_file: str, mzml_file: str, fasta_file: str):
        self.output_file = output_file
        self.mzml_file = mzml_file
        self.fasta_file = fasta_file
        self.file_handle = None
        self.spectrum_counter = 0
        
    def __enter__(self):
        self.file_handle = open(self.output_file, 'w')
        self._write_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write_footer()
        if self.file_handle:
            self.file_handle.close()
    
    def _write_header(self):
        """Write pepXML header."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        header = f'''<?xml version="1.0" encoding="UTF-8"?>
<msms_pipeline_analysis date="{timestamp}" xmlns="http://regis-web.systemsbiology.net/pepXML" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/pepXML/pepXML_v117.xsd" summary_xml="{self.output_file}">
<msms_run_summary base_name="{os.path.splitext(os.path.basename(self.mzml_file))[0]}" raw_data_type="raw" raw_data=".mzML" search_engine="SimpleSearch">
<sample_enzyme name="trypsin">
<specificity cut="KR" no_cut="P" sense="C"/>
</sample_enzyme>
<search_summary base_name="{os.path.splitext(os.path.basename(self.mzml_file))[0]}" search_engine="SimpleSearch" precursor_mass_type="monoisotopic" fragment_mass_type="monoisotopic" out_data_type="" out_data=".pepXML" search_id="1">
<search_database local_path="{self.fasta_file}" type="AA"/>
<enzymatic_search_constraint enzyme="trypsin" max_num_internal_cleavages="2" min_number_termini="2"/>
<aminoacid_modification aminoacid="C" massdiff="+57.021464" mass="160.030649" variable="N" symbol="^"/>
<parameter name="fragment_mass_tolerance" value="1.0"/>
<parameter name="parent_mass_tolerance" value="3.0"/>
<parameter name="parent_mass_type" value="monoisotopic"/>
<parameter name="fragment_mass_type" value="monoisotopic"/>
</search_summary>
'''
        self.file_handle.write(header)
    
    def _write_footer(self):
        """Write pepXML footer."""
        footer = '''</msms_run_summary>
</msms_pipeline_analysis>
'''
        self.file_handle.write(footer)
    
    def write_spectrum_query(self, spectrum: 'MassSpectrum', search_results: List[Tuple['PeptideCandidate', float, float, int]], top_hits_per_charge: int = 3):
        """
        Write a spectrum query with its search results, grouped by charge state.
        
        Args:
            spectrum: The experimental spectrum
            search_results: List of (peptide, xcorr_score, e_value, charge) tuples
            top_hits_per_charge: Number of top hits to report per charge state (default: 3)
        """
        self.spectrum_counter += 1
        
        # Group results by charge state
        results_by_charge = {}
        for peptide, xcorr_score, e_value, charge in search_results:
            if charge not in results_by_charge:
                results_by_charge[charge] = []
            results_by_charge[charge].append((peptide, xcorr_score, e_value, charge))
        
        # For each charge state, determine the neutral mass from the best hit
        # and write a separate spectrum_query entry
        charge_states = sorted(results_by_charge.keys())
        
        for charge_idx, charge in enumerate(charge_states):
            charge_results = results_by_charge[charge][:top_hits_per_charge]
            
            if not charge_results:
                continue
                
            # Calculate neutral mass using this charge state
            assumed_charge = charge
            proton_mass = 1.007276
            neutral_mass = (spectrum.precursor_mz * assumed_charge) - (assumed_charge * proton_mass)
            
            # Use standard pepXML format - spectrum should be scan ID, not include charge state
            spectrum_id = spectrum.scan_id
            
            spectrum_query = f'''<spectrum_query spectrum="{spectrum_id}" start_scan="{self._extract_scan_number(spectrum.scan_id)}" end_scan="{self._extract_scan_number(spectrum.scan_id)}" precursor_neutral_mass="{neutral_mass:.6f}" assumed_charge="{assumed_charge}" index="{self.spectrum_counter + charge_idx}">
'''
            self.file_handle.write(spectrum_query)
            
            # Write search results for this charge state
            search_result = '<search_result>\n'
            self.file_handle.write(search_result)
            
            # Write search hits for this charge state
            for hit_rank, (peptide, xcorr_score, e_value, peptide_charge) in enumerate(charge_results, 1):
                # Calculate peptide properties
                peptide_mass = peptide.mass
                # Calculate mass difference: (observed_neutral_mass - theoretical_peptide_mass)
                mass_diff = neutral_mass - peptide_mass
                
                # Count missed cleavages
                num_missed_cleavages = peptide.sequence.count('K') + peptide.sequence.count('R') - 1
                if peptide.sequence.endswith('K') or peptide.sequence.endswith('R'):
                    num_missed_cleavages -= 1
                num_missed_cleavages = max(0, num_missed_cleavages)
                
                # Determine termini
                tot_num_proteins = 1  # Simplified
                num_tol_term = 2  # Assuming fully tryptic
                
                search_hit = f'''<search_hit hit_rank="{hit_rank}" peptide="{peptide.sequence}" peptide_prev_aa="-" peptide_next_aa="-" protein="{peptide.protein_id}" num_tot_proteins="{tot_num_proteins}" num_matched_ions="0" tot_num_ions="0" calc_neutral_pep_mass="{peptide_mass:.6f}" massdiff="{mass_diff:.6f}" num_tol_term="{num_tol_term}" num_missed_cleavages="{num_missed_cleavages}" is_rejected="0">
<search_score name="xcorr" value="{xcorr_score:.4f}"/>
<search_score name="expect" value="{e_value:.2e}"/>
</search_hit>
'''
                self.file_handle.write(search_hit)
            
            # Close search_result and spectrum_query for this charge state
            self.file_handle.write('</search_result>\n')
            self.file_handle.write('</spectrum_query>\n')
        
        # Update spectrum counter to account for multiple charge states
        self.spectrum_counter += len(charge_states) - 1
        self.file_handle.flush()  # Ensure data is written immediately
    
    def _extract_scan_number(self, scan_id: str) -> str:
        """Extract scan number from scan ID."""
        # Convert to string if it's not already
        scan_id_str = str(scan_id)
        
        # Try to extract number from scan ID
        import re
        match = re.search(r'scan[=\s]*(\d+)', scan_id_str, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try to find any number in the scan ID
        match = re.search(r'(\d+)', scan_id_str)
        if match:
            return match.group(1)
        
        return str(self.spectrum_counter)


class PINWriter:
    """Class to write results in Percolator Input (PIN) format."""
    
    def __init__(self, output_file: str, mzml_file: str):
        self.output_file = output_file
        self.mzml_file = mzml_file
        self.file_handle = None
        self.spectrum_counter = 0
        # Extract base filename without extension for SpecId generation
        self.base_filename = os.path.splitext(os.path.basename(mzml_file))[0]
        
    def __enter__(self):
        self.file_handle = open(self.output_file, 'w')
        self._write_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def _write_header(self):
        """Write PIN header."""
        # PIN format: tab-delimited with specific columns matching Percolator input format
        header = "SpecId\tLabel\tScanNr\tExpMass\tCalcMass\te-value\tXcorr\tIonFrac\tPepLen\tCharge1\tCharge2\tCharge3\tdM\tabsdM\tPeptide\tProteins\n"
        self.file_handle.write(header)
    
    def write_spectrum_results(self, spectrum: 'MassSpectrum', search_results: List[Tuple['PeptideCandidate', float, float, int]]):
        """
        Write best peptide results for each charge state to PIN format.
        
        For each spectrum, write only the best peptide for each charge state (target and decoy).
        
        Args:
            spectrum: The experimental spectrum
            search_results: List of (peptide, xcorr_score, e_value, charge) tuples
        """
        self.spectrum_counter += 1
        
        if not search_results:
            return
        
        # Extract scan number from spectrum scan_id
        scan_nr = self._extract_scan_number(spectrum.scan_id)
        
        # Calculate experimental mass (center of precursor isolation window)
        exp_mass = (spectrum.isolation_window_lower + spectrum.isolation_window_upper) / 2.0
        
        # Group results by charge state and keep only the best (highest XCorr) for each charge
        best_by_charge = {}
        for peptide, xcorr_score, e_value, charge in search_results:
            if charge not in best_by_charge or xcorr_score > best_by_charge[charge][1]:
                best_by_charge[charge] = (peptide, xcorr_score, e_value, charge)
        
        # Write the best peptide for each charge state
        for peptide, xcorr_score, e_value, charge in best_by_charge.values():
            # Generate SpecId: filename_scannr_scannr_charge
            spec_id = f"{self.base_filename}_{scan_nr}_{scan_nr}_{charge}"
            
            # Calculate theoretical m/z
            calc_mass = (peptide.mass + charge * 1.007276) / charge  # Using proton mass
            
            # Determine label (1 for target, -1 for decoy)
            label = -1 if peptide.protein_id.startswith('decoy_') else 1
            
            # Calculate mass difference (dM = ExpMass - CalcMass)
            dm = exp_mass - calc_mass
            abs_dm = abs(dm)
            
            # Ion fraction (placeholder - we don't calculate this yet)
            ion_frac = 0.0
            
            # Peptide length
            pep_len = len(peptide.sequence)
            
            # Charge state booleans
            charge1 = 1 if charge == 1 else 0
            charge2 = 1 if charge == 2 else 0  
            charge3 = 1 if charge == 3 else 0
            
            # Format peptide with flanking amino acids (using placeholder)
            peptide_formatted = f"-.{peptide.sequence}.-"
            
            # Extract protein identifier (first part before any description)
            proteins = peptide.protein_id.split(';')[0]  # Take first protein if multiple
            
            # Format the PIN line according to the new specification
            pin_line = f"{spec_id}\t{label}\t{scan_nr}\t{exp_mass:.6f}\t{calc_mass:.6f}\t{e_value:.6f}\t{xcorr_score:.3f}\t{ion_frac:.5f}\t{pep_len}\t{charge1}\t{charge2}\t{charge3}\t{dm:.6f}\t{abs_dm:.6f}\t{peptide_formatted}\t{proteins}\n"
            self.file_handle.write(pin_line)
        
        self.file_handle.flush()  # Ensure data is written immediately
    
    def _extract_scan_number(self, scan_id: str) -> str:
        """Extract scan number from scan ID."""
        # Convert to string if it's not already
        scan_id_str = str(scan_id)
        
        # Try to extract number from scan ID
        import re
        match = re.search(r'scan[=\s]*(\d+)', scan_id_str, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try to find any number in the scan ID
        match = re.search(r'(\d+)', scan_id_str)
        if match:
            return match.group(1)
        
        return str(self.spectrum_counter)


class DIAResultsWriter:
    """Class to write DIA peptide-centric search results."""
    
    def __init__(self, output_file: str, mzml_file: str, write_lock=None, library_mode=False):
        self.output_file = output_file
        self.mzml_file = mzml_file
        self.file_handle = None
        self.write_lock = write_lock  # Optional lock for thread-safe writing
        self.library_mode = library_mode  # Whether using library search
        self.header_written = False
        
    def __enter__(self):
        self.file_handle = open(self.output_file, 'w')
        self._write_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def open_for_append(self):
        """Open file for appending (for incremental writes from multiple workers)."""
        self.file_handle = open(self.output_file, 'a')
    
    def close(self):
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def _write_header(self):
        """Write DIA results header."""
        if self.library_mode:
            # Library mode: paired target/decoy output with delta columns
            header = "Peptide\tCharge\tProteinID\tMass\tIsolationWindow\tNumSpectraScored\t"
            header += "LibCosine\tLibCosineZScore\tXCorr\tRT\tScanID\tPrecursorCosine\t"
            header += "delta_rt\tdelta_mz_ppm_precursor\tdelta_mz_ppm_fragments\t"
            header += "decoy_Peptide\tdecoy_LibCosine\tdecoy_LibCosineZScore\tdecoy_XCorr\tdecoy_RT\tdecoy_ScanID\tdecoy_PrecursorCosine\t"
            header += "decoy_delta_rt\tdecoy_delta_mz_ppm_precursor\tdecoy_delta_mz_ppm_fragments\n"
        else:
            # Non-library mode: paired target/decoy output with XCorr (no delta columns without library)
            header = "Peptide\tCharge\tProteinID\tMass\tIsolationWindow\tNumSpectraScored\t"
            header += "BestXCorr\tBestRT\tBestScan\tEValue\tXCorrZScore\t"
            header += "decoy_Peptide\tdecoy_BestXCorr\tdecoy_BestRT\tdecoy_BestScan\tdecoy_EValue\tdecoy_XCorrZScore\n"
        
        self.file_handle.write(header)
    
    def write_dia_results(self, results: Dict):
        """
        Write DIA peptide-centric results in paired target/decoy format.
        
        Args:
            results: Dictionary from search_dia_peptide_centric
                     Keys: pair_id (unique identifier for target/decoy pair)
                     Values: dict with 'target' and 'decoy' subdicts containing peptide info and scores
        """
        # Debug: Check the structure of results
        if results:
            first_key = next(iter(results))
            first_value = results[first_key]
            if not isinstance(first_value, dict) or 'target' not in first_value:
                raise ValueError(f"ERROR: Results format is incorrect. Expected paired format with 'target'/'decoy' keys, got: {type(first_value)}, keys: {first_value.keys() if isinstance(first_value, dict) else 'N/A'}")
        
        for pair_id, pair_data in results.items():
            target_result = pair_data['target']
            decoy_result = pair_data['decoy']
            
            # Basic peptide info from target
            target_peptide = target_result['peptide']
            decoy_peptide = decoy_result['peptide']
            charge = target_result['charge']
            isolation_window = target_result['isolation_window']
            window_str = f"[{isolation_window[0]:.4f}-{isolation_window[1]:.4f}]"
            num_spectra = target_result['num_spectra_scored']
            
            if self.library_mode:
                # Library mode: LibCosine determines which scan to use for XCorr/PrecursorCosine
                target_lib_cosine = target_result['best_lib_cosine_target']
                target_lib_zscore = target_result['lib_cosine_target_zscore']
                target_xcorr = target_result['best_xcorr']  # XCorr at LibCosine peak
                target_rt = target_result['best_rt']
                target_scan = target_result['best_scan']
                target_precursor_cosine = target_result['precursor_cosine_target']
                target_delta_rt = target_result.get('delta_rt')
                target_delta_mz_precursor = target_result.get('delta_mz_ppm_precursor')
                target_delta_mz_fragments = target_result.get('delta_mz_ppm_fragments')
                
                decoy_lib_cosine = decoy_result['best_lib_cosine_decoy']
                decoy_lib_zscore = decoy_result['lib_cosine_decoy_zscore']
                decoy_xcorr = decoy_result['best_xcorr']  # XCorr at decoy LibCosine peak
                decoy_rt = decoy_result['best_rt']
                decoy_scan = decoy_result['best_scan']
                decoy_precursor_cosine = decoy_result['precursor_cosine_decoy']
                decoy_delta_rt = decoy_result.get('delta_rt')
                decoy_delta_mz_precursor = decoy_result.get('delta_mz_ppm_precursor')
                decoy_delta_mz_fragments = decoy_result.get('delta_mz_ppm_fragments')
                
                # Format delta columns (use empty string if None)
                target_delta_rt_str = f"{target_delta_rt:.4f}" if target_delta_rt is not None else ""
                target_delta_mz_precursor_str = f"{target_delta_mz_precursor:.4f}" if target_delta_mz_precursor is not None else ""
                target_delta_mz_fragments_str = f"{target_delta_mz_fragments:.4f}" if target_delta_mz_fragments is not None else ""
                decoy_delta_rt_str = f"{decoy_delta_rt:.4f}" if decoy_delta_rt is not None else ""
                decoy_delta_mz_precursor_str = f"{decoy_delta_mz_precursor:.4f}" if decoy_delta_mz_precursor is not None else ""
                decoy_delta_mz_fragments_str = f"{decoy_delta_mz_fragments:.4f}" if decoy_delta_mz_fragments is not None else ""
                
                line = f"{target_peptide.sequence}\t{charge}\t{target_peptide.protein_id}\t{target_peptide.mass:.6f}\t{window_str}\t{num_spectra}\t"
                line += f"{target_lib_cosine:.4f}\t{target_lib_zscore:.4f}\t{target_xcorr:.4f}\t{target_rt:.2f}\t{target_scan}\t{target_precursor_cosine:.4f}\t"
                line += f"{target_delta_rt_str}\t{target_delta_mz_precursor_str}\t{target_delta_mz_fragments_str}\t"
                line += f"{decoy_peptide.sequence}\t{decoy_lib_cosine:.4f}\t{decoy_lib_zscore:.4f}\t{decoy_xcorr:.4f}\t{decoy_rt:.2f}\t{decoy_scan}\t{decoy_precursor_cosine:.4f}\t"
                line += f"{decoy_delta_rt_str}\t{decoy_delta_mz_precursor_str}\t{decoy_delta_mz_fragments_str}\n"
                
            else:
                # Non-library mode: XCorr-based with e-value
                target_xcorr = target_result['best_xcorr']
                target_rt = target_result['best_rt']
                target_scan = target_result['best_scan']
                target_evalue = target_result['e_value']
                target_zscore = target_result['xcorr_zscore']
                
                decoy_xcorr = decoy_result['best_xcorr']
                decoy_rt = decoy_result['best_rt']
                decoy_scan = decoy_result['best_scan']
                decoy_evalue = decoy_result['e_value']
                decoy_zscore = decoy_result['xcorr_zscore']
                
                line = f"{target_peptide.sequence}\t{charge}\t{target_peptide.protein_id}\t{target_peptide.mass:.6f}\t{window_str}\t{num_spectra}\t"
                line += f"{target_xcorr:.4f}\t{target_rt:.2f}\t{target_scan}\t{target_evalue:.6e}\t{target_zscore:.4f}\t"
                line += f"{decoy_peptide.sequence}\t{decoy_xcorr:.4f}\t{decoy_rt:.2f}\t{decoy_scan}\t{decoy_evalue:.6e}\t{decoy_zscore:.4f}\n"

            self.file_handle.write(line)
    
    def write_dia_results_synchronized(self, results: Dict):
        """
        Write DIA results with thread-safe locking for parallel workers.
        Opens file in append mode, acquires lock, writes, then closes.
        """
        if self.write_lock:
            with self.write_lock:
                self.open_for_append()
                self.write_dia_results(results)
                self.file_handle.flush()  # Ensure data is written
                self.close()
        else:
            # No lock provided, just write directly
            self.write_dia_results(results)
            if self.file_handle:
                self.file_handle.flush()


# QC Data Filtering Functions
def filter_qc_data_by_fdr(qc_data, winners_df, fdr_threshold=0.01):
    """
    Filter QC data to only include precursors at or below the specified FDR threshold.
    
    Args:
        qc_data: Dictionary with 'ms1_mass_errors', 'ms2_mass_errors', 'rt_pairs'
        winners_df: DataFrame from competition with columns ['Peptide', 'Charge', 'IsTarget', 'LibCosine']
        fdr_threshold: FDR threshold (default 0.01 for 1% FDR)
    
    Returns:
        Filtered qc_data dictionary with same structure but only high-confidence precursors
    """
    # Calculate FDR for each precursor based on LibCosine score
    winners_sorted = winners_df.sort_values('LibCosine', ascending=False).copy()
    winners_sorted['cumulative_targets'] = (winners_sorted['IsTarget'] == 'Target').cumsum()
    winners_sorted['cumulative_decoys'] = (winners_sorted['IsTarget'] == 'Decoy').cumsum()
    winners_sorted['fdr'] = winners_sorted['cumulative_decoys'] / winners_sorted['cumulative_targets'].replace(0, 1)
    
    # Get precursors (peptide+charge) at or below FDR threshold (targets only)
    high_conf = winners_sorted[
        (winners_sorted['fdr'] <= fdr_threshold) & 
        (winners_sorted['IsTarget'] == 'Target')
    ]
    valid_precursors = set(zip(high_conf['Peptide'].values, high_conf['Charge'].values))
    
    print(f"\nFiltering QC data to precursors at {fdr_threshold*100:.1f}% FDR...")
    print(f"  High-confidence precursors: {len(valid_precursors):,}")
    
    # Filter MS1 mass errors - match by precursor (peptide+charge)
    filtered_ms1 = [
        entry['error'] for entry in qc_data['ms1_mass_errors']
        if (entry['peptide'], entry['charge']) in valid_precursors and entry['is_target']
    ]
    
    # Filter MS2 mass errors - match by precursor (peptide+charge)
    filtered_ms2 = [
        entry['error'] for entry in qc_data['ms2_mass_errors']
        if (entry['peptide'], entry['charge']) in valid_precursors and entry['is_target']
    ]
    
    # Filter RT pairs - match by precursor (peptide+charge)
    filtered_rt = [
        entry for entry in qc_data['rt_pairs']
        if (entry['peptide'], entry['charge']) in valid_precursors and entry['is_target']
    ]
    
    print(f"  MS1 mass accuracy: {len(qc_data['ms1_mass_errors']):,} total → {len(filtered_ms1):,} at <{fdr_threshold*100:.0f}% FDR")
    if len(filtered_ms1) < len(valid_precursors):
        print(f"    ({len(valid_precursors) - len(filtered_ms1)} precursors have no detectable M+0 signal in MS1)")
    print(f"  MS2 mass accuracy: {len(qc_data['ms2_mass_errors']):,} total → {len(filtered_ms2):,} at <{fdr_threshold*100:.0f}% FDR")
    print(f"  RT pairs: {len(qc_data['rt_pairs']):,} total → {len(filtered_rt):,} at <{fdr_threshold*100:.0f}% FDR")
    
    return {
        'ms1_mass_errors': filtered_ms1,
        'ms2_mass_errors': filtered_ms2,
        'rt_pairs': filtered_rt,
        'ms1_tol_unit': qc_data['ms1_tol_unit'],
        'ms2_tol_unit': qc_data['ms2_tol_unit'],
        'num_precursors': len(valid_precursors)
    }


# QC Plotting Functions
def plot_mass_accuracy_histograms(ms1_errors, ms2_errors, output_prefix, ms1_unit='ppm', ms2_unit='ppm', num_precursors=None):
    """
    Generate MS1 and MS2 mass accuracy histogram plots.
    
    Args:
        ms1_errors: List of MS1 M+0 mass errors (filtered to <1% FDR targets)
        ms2_errors: List of MS2 fragment mass errors (filtered to <1% FDR targets)
        output_prefix: Output file prefix (will add '_ms1_accuracy.png' and '_ms2_accuracy.png')
        ms1_unit: Unit for MS1 errors ('ppm' or 'mz')
        ms2_unit: Unit for MS2 errors ('ppm' or 'mz')
        num_precursors: Number of high-confidence precursors (for plot title)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # MS1 mass accuracy histogram
    if len(ms1_errors) > 0:
        ms1_array = np.array(ms1_errors)
        mean_ms1 = np.mean(ms1_array)
        std_ms1 = np.std(ms1_array)
        
        unit_label_ms1 = 'PPM' if ms1_unit == 'ppm' else 'm/z'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ms1_array, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(mean_ms1, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_ms1:.6f} {unit_label_ms1}')
        ax.axvline(mean_ms1 + std_ms1, color='orange', linestyle=':', linewidth=2, label=f'+SD = {mean_ms1 + std_ms1:.6f} {unit_label_ms1}')
        ax.axvline(mean_ms1 - std_ms1, color='orange', linestyle=':', linewidth=2, label=f'-SD = {mean_ms1 - std_ms1:.6f} {unit_label_ms1}')
        
        ax.set_xlabel(f'Mass Error (observed - theoretical, {unit_label_ms1})', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = f'MS1 Mass Accuracy (M+0 Peak)\n{len(ms1_errors):,} measurements'
        if num_precursors:
            title += f' from {num_precursors:,} precursors (<1% FDR)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_ms1_accuracy.png', dpi=300)
        plt.close()
        print(f"\nMS1 mass accuracy plot saved: {output_prefix}_ms1_accuracy.png")
        print(f"  Mean = {mean_ms1:.6f} {unit_label_ms1}, SD = {std_ms1:.6f} {unit_label_ms1}")
    else:
        print("\nNo MS1 mass errors collected (MS1 data may not be available)")
    
    # MS2 mass accuracy histogram
    if len(ms2_errors) > 0:
        ms2_array = np.array(ms2_errors)
        mean_ms2 = np.mean(ms2_array)
        std_ms2 = np.std(ms2_array)
        
        unit_label_ms2 = 'PPM' if ms2_unit == 'ppm' else 'm/z'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ms2_array, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(mean_ms2, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_ms2:.6f} {unit_label_ms2}')
        ax.axvline(mean_ms2 + std_ms2, color='orange', linestyle=':', linewidth=2, label=f'+SD = {mean_ms2 + std_ms2:.6f} {unit_label_ms2}')
        ax.axvline(mean_ms2 - std_ms2, color='orange', linestyle=':', linewidth=2, label=f'-SD = {mean_ms2 - std_ms2:.6f} {unit_label_ms2}')
        
        ax.set_xlabel(f'Mass Error (observed - theoretical, {unit_label_ms2})', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = f'MS2 Fragment Mass Accuracy\n{len(ms2_errors):,} measurements'
        if num_precursors:
            title += f' from {num_precursors:,} precursors (<1% FDR)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_ms2_accuracy.png', dpi=300)
        plt.close()
        print(f"MS2 mass accuracy plot saved: {output_prefix}_ms2_accuracy.png")
        print(f"  Mean = {mean_ms2:.6f} {unit_label_ms2}, SD = {std_ms2:.6f} {unit_label_ms2}")
    else:
        print("\nNo MS2 mass errors collected (library mode may not be active)")


def plot_rt_correlation(rt_pairs, output_prefix):
    """
    Generate RT correlation plot with LOESS fit.
    
    Args:
        rt_pairs: List of dicts with 'library_rt', 'measured_rt', 'lib_cosine', 'is_target' (filtered to <1% FDR)
        output_prefix: Output file prefix (will add '_rt_correlation.png')
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    from scipy.interpolate import UnivariateSpline
    
    if len(rt_pairs) == 0:
        print("\nNo RT pairs available for plotting (library mode may not be active or filtering removed all data)")
        return
    
    # Convert to DataFrame for easier manipulation
    rt_df = pd.DataFrame(rt_pairs)
    
    # Should already be filtered to targets only, but double-check
    target_df = rt_df[rt_df['is_target']].copy()
    if len(target_df) == 0:
        print("\nNo target peptides found in RT pairs")
        return
    
    print(f"\nRT correlation plot: Using {len(target_df)} target peptides at <1% FDR")
    
    # Extract arrays
    lib_rt = target_df['library_rt'].values
    meas_rt = target_df['measured_rt'].values
    
    # Fit LOESS using cubic spline as approximation
    # Sort by library RT for smooth fit
    sort_idx = np.argsort(lib_rt)
    lib_rt_sorted = lib_rt[sort_idx]
    meas_rt_sorted = meas_rt[sort_idx]
    
    try:
        # Use UnivariateSpline with smoothing
        spl = UnivariateSpline(lib_rt_sorted, meas_rt_sorted, s=len(lib_rt_sorted)*0.1)
        lib_rt_fit = np.linspace(lib_rt_sorted.min(), lib_rt_sorted.max(), 500)
        meas_rt_fit = spl(lib_rt_fit)
        
        # Calculate residuals
        meas_rt_pred = spl(lib_rt_sorted)
        residuals = meas_rt_sorted - meas_rt_pred
        sd_residuals = np.std(residuals)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use hexbin for density visualization if many points, otherwise scatter
        if len(target_df) > 1000:
            # Hexbin plot for high-density visualization
            hexbin = ax.hexbin(lib_rt, meas_rt, gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
            plt.colorbar(hexbin, ax=ax, label='Peptide Count')
            plot_type = 'hexbin'
        else:
            # Scatter plot for smaller datasets
            ax.scatter(lib_rt, meas_rt, alpha=0.3, s=20, c='blue', label='Target peptides')
            plot_type = 'scatter'
        
        # LOESS fit
        ax.plot(lib_rt_fit, meas_rt_fit, 'r-', linewidth=2, label='LOESS fit', zorder=10)
        ax.plot(lib_rt_fit, meas_rt_fit + sd_residuals, 'orange', linestyle='--', linewidth=1.5, 
                label=f'+SD ({sd_residuals:.2f})', zorder=10)
        ax.plot(lib_rt_fit, meas_rt_fit - sd_residuals, 'orange', linestyle='--', linewidth=1.5, 
                label=f'-SD ({sd_residuals:.2f})', zorder=10)
        
        # Note: Library RT may be in iRT units (unitless), not minutes
        # Don't assume perfect correlation (slope=1) - removed diagonal line
        
        ax.set_xlabel('Library RT (may be iRT units)', fontsize=12)
        ax.set_ylabel('Measured RT (minutes)', fontsize=12)
        ax.set_title(f'Retention Time Correlation (<1% FDR targets)\nN = {len(target_df):,} peptides', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_rt_correlation.png', dpi=300)
        plt.close()
        print(f"RT correlation plot saved: {output_prefix}_rt_correlation.png")
        print(f"  Visualization type: {plot_type}")
        print(f"  SD of residuals = {sd_residuals:.2f}")
        
    except Exception as e:
        print(f"\nError generating RT correlation plot: {e}")
        print("  This may occur with insufficient data points or RT range")


def run_calibration_workflow(xcorr_engine, library, spectra, ms1_spectra, charge_states,
                             cal_library_peptides, lib_fragment_tol, lib_precursor_tol,
                             lib_fragment_tol_unit, lib_precursor_tol_unit, output_file,
                             fasta_file, enzyme, decoy_cycle_length, verbose=0):
    """
    Run calibration search workflow with escalation strategy.
    
    Args:
        xcorr_engine: FastXCorr instance
        library: SpectrumLibrary instance
        spectra: All MS2 spectra from mzML
        ms1_spectra: All MS1 spectra from mzML
        charge_states: List of charge states to search
        cal_library_peptides: Initial number of peptides to sample
        lib_fragment_tol, lib_precursor_tol: Tolerances
        lib_fragment_tol_unit, lib_precursor_tol_unit: Tolerance units
        output_file: Output file path (for auto-naming calibration JSON)
        fasta_file: FASTA file path (for generating decoys)
        enzyme: Enzyme name
        decoy_cycle_length: Decoy generation cycle length
        verbose: Verbosity level
        
    Returns:
        Dictionary with calibration parameters
        
    Raises:
        RuntimeError if calibration fails after escalation
    """
    import time
    calibration_start = time.time()
    
    print("\n" + "="*80)
    print("CALIBRATION SEARCH")
    print("="*80)
    
    # Try with initial number, then double once if needed
    for attempt, num_peptides in enumerate([cal_library_peptides, cal_library_peptides * 2]):
        print(f"\nCalibration attempt {attempt + 1}: Sampling {num_peptides} high-quality precursors...")
        
        # Sample precursors from library
        sampled_precursors = library.sample_precursors(num_peptides, seed=42, max_qvalue=0.01)
        print(f"  Sampled {len(sampled_precursors)} precursors from library")
        
        # Create PeptideCandidate objects for sampled precursors
        # Include charge state info to avoid testing each precursor with multiple charges
        from collections import namedtuple
        PeptideCand = namedtuple('PeptideCandidate', ['sequence', 'protein_id', 'mass', 'charge'])
        
        target_decoy_pairs = []
        for sequence, charge in sampled_precursors:
            # Calculate mass
            mass = sum(xcorr_engine.aa_masses.get(aa, 0) for aa in sequence)
            mass += xcorr_engine.h2o_mass  # Add water for peptide mass
            
            # Create target with charge info
            target = PeptideCand(sequence=sequence, protein_id='CALIBRATION', mass=mass, charge=charge)
            
            # Generate decoy
            decoy_seq = xcorr_engine.generate_decoy_sequence(sequence, decoy_cycle_length)
            decoy_mass = sum(xcorr_engine.aa_masses.get(aa, 0) for aa in decoy_seq)
            decoy_mass += xcorr_engine.h2o_mass
            decoy = PeptideCand(sequence=decoy_seq, protein_id='DECOY_CALIBRATION', mass=decoy_mass, charge=charge)
            
            target_decoy_pairs.append((target, decoy))
        
        print(f"  Created {len(target_decoy_pairs)} target-decoy pairs for calibration search")
        
        # Group spectra by isolation window
        print("  Grouping spectra by isolation window...")
        window_groups = xcorr_engine.group_spectra_by_isolation_window(spectra)
        print(f"  Found {len(window_groups)} isolation windows")
        
        # Run DIA search on all isolation windows for calibration
        print(f"  Searching {len(window_groups)} isolation windows with {len(target_decoy_pairs)} peptides...")
        
        # Use parallel processing for calibration search
        from multiprocessing import Pool, cpu_count
        
        # Determine number of workers (same logic as full search)
        n_workers = max(1, cpu_count() - 1)
        print(f"  Using {n_workers} parallel workers for calibration search")
        
        # Serialize target_decoy_pairs for parallel processing
        # Include charge in serialized data for calibration
        target_decoy_pairs_data = []
        for target, decoy in target_decoy_pairs:
            target_data = {
                'sequence': target.sequence,
                'protein_id': target.protein_id,
                'mass': target.mass,
                'charge': target.charge
            }
            decoy_data = {
                'sequence': decoy.sequence,
                'protein_id': decoy.protein_id,
                'mass': decoy.mass,
                'charge': decoy.charge
            }
            target_decoy_pairs_data.append((target_data, decoy_data))
        
        # Prepare work items for parallel processing
        work_items = []
        for window_idx, (isolation_window, window_spectra) in enumerate(window_groups.items()):
            work_items.append((
                window_idx,
                len(window_groups),
                isolation_window,
                window_spectra,
                fasta_file,
                target_decoy_pairs_data,
                [],  # Empty charge_states list signals worker to use charge from peptide data
                None,  # No parquet output for calibration
                enzyme,
                decoy_cycle_length,
                library,
                ms1_spectra,
                lib_fragment_tol if lib_fragment_tol_unit == 'ppm' else lib_fragment_tol,
                lib_precursor_tol if lib_precursor_tol_unit == 'ppm' else lib_precursor_tol,
                lib_fragment_tol_unit,
                lib_precursor_tol_unit,
                None,  # No calibration during calibration search
                0  # Suppress verbose output per worker
            ))
        
        all_qc_data = {
            'ms1_mass_errors': [],
            'ms2_mass_errors': [],
            'rt_pairs': [],
            'ms1_tol_unit': lib_precursor_tol_unit,
            'ms2_tol_unit': lib_fragment_tol_unit
        }
        
        all_results = []
        
        # Process windows in parallel
        if n_workers == 1:
            # Sequential for debugging
            for item in work_items:
                search_result = process_isolation_window_worker(item)
                
                # Collect QC data
                qc_data = search_result.get('qc_data', {})
                all_qc_data['ms1_mass_errors'].extend(qc_data.get('ms1_mass_errors', []))
                all_qc_data['ms2_mass_errors'].extend(qc_data.get('ms2_mass_errors', []))
                all_qc_data['rt_pairs'].extend(qc_data.get('rt_pairs', []))
                
                # Collect results for FDR filtering
                for pair_id, pair_data in search_result.get('results', {}).items():
                    all_results.append(pair_data)
        else:
            # Parallel processing
            with Pool(n_workers) as pool:
                for window_idx, search_result in enumerate(pool.imap_unordered(process_isolation_window_worker, work_items)):
                    # Progress indicator with timing
                    if verbose >= 1 or (window_idx + 1) % 10 == 0:
                        elapsed = time.time() - calibration_start
                        print(f"  Calibration: {window_idx + 1}/{len(window_groups)} windows processed ({elapsed:.1f}s elapsed)")
                    
                    # Collect QC data
                    qc_data = search_result.get('qc_data', {})
                    all_qc_data['ms1_mass_errors'].extend(qc_data.get('ms1_mass_errors', []))
                    all_qc_data['ms2_mass_errors'].extend(qc_data.get('ms2_mass_errors', []))
                    all_qc_data['rt_pairs'].extend(qc_data.get('rt_pairs', []))
                    
                    # Collect results for FDR filtering
                    for pair_id, pair_data in search_result.get('results', {}).items():
                        all_results.append(pair_data)
        
        calibration_search_time = time.time() - calibration_start
        print(f"\n  Calibration search complete in {calibration_search_time:.1f}s: Collected QC data")
        print(f"    MS1 mass errors: {len(all_qc_data['ms1_mass_errors'])}")
        print(f"    MS2 mass errors: {len(all_qc_data['ms2_mass_errors'])}")
        print(f"    RT pairs: {len(all_qc_data['rt_pairs'])}")
        
        # Convert results to DataFrame for FDR filtering
        winners_data = []
        for pair_data in all_results:
            target = pair_data['target']
            decoy = pair_data['decoy']
            
            # Determine winner (higher LibCosine wins)
            target_score = target.get('best_lib_cosine_target', 0)
            decoy_score = decoy.get('best_lib_cosine_decoy', 0)
            
            if target_score >= decoy_score:
                winners_data.append({
                    'Peptide': target['peptide'].sequence,
                    'Charge': target['charge'],
                    'LibCosine': target_score,
                    'IsTarget': 'Target'
                })
            else:
                winners_data.append({
                    'Peptide': decoy['peptide'].sequence,
                    'Charge': decoy['charge'],
                    'LibCosine': decoy_score,
                    'IsTarget': 'Decoy'
                })
        
        winners_df = pd.DataFrame(winners_data)
        
        # Filter to 1% FDR
        print("\n  Filtering calibration results to 1% FDR...")
        filtered_qc = filter_qc_data_by_fdr(all_qc_data, winners_df, fdr_threshold=0.01)
        num_confident = filtered_qc['num_precursors']
        
        print(f"  High-confidence target precursors at 1% FDR: {num_confident}")
        
        if num_confident >= 100:
            total_calibration_time = time.time() - calibration_start
            print(f"✓ Calibration successful: {num_confident} confident peptides ({total_calibration_time:.1f}s total)")
            
            # Calculate calibration parameters
            print("\n  Calculating m/z calibration...")
            mz_cal = xcorr_engine.calculate_mz_calibration(filtered_qc)
            
            print(f"    MS1: mean = {mz_cal['ms1_mean']:.4f} {mz_cal['ms1_unit']}, "
                  f"SD = {mz_cal['ms1_sd']:.4f} {mz_cal['ms1_unit']}")
            print(f"    MS2: mean = {mz_cal['ms2_mean']:.4f} {mz_cal['ms2_unit']}, "
                  f"SD = {mz_cal['ms2_sd']:.4f} {mz_cal['ms2_unit']}")
            
            print("\n  Fitting RT calibration...")
            rt_cal = xcorr_engine.fit_rt_calibration(filtered_qc['rt_pairs'])
            
            print(f"    Method: {rt_cal['method']}")
            print(f"    R² = {rt_cal['r_squared']:.4f}, RMSE = {rt_cal['rmse']:.4f} min")
            print(f"    Residual SD = {rt_cal['residual_sd']:.4f} min")
            print(f"    RT window (3σ) = ±{3 * rt_cal['residual_sd']:.2f} min")
            
            # Create calibration dictionary
            calibration = {
                'calibration_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_library_peptides_sampled': num_peptides,
                    'num_passing_fdr': num_confident,
                    'fdr_threshold': 0.01,
                    'calibration_successful': True
                },
                'ms1_calibration': {
                    'mean': mz_cal['ms1_mean'],
                    'sd': mz_cal['ms1_sd'],
                    'unit': mz_cal['ms1_unit'],
                    'adjusted_tolerance': mz_cal['ms1_mean'] + 3 * mz_cal['ms1_sd'],
                    'window_halfwidth_multiplier': 3.0
                },
                'ms2_calibration': {
                    'mean': mz_cal['ms2_mean'],
                    'sd': mz_cal['ms2_sd'],
                    'unit': mz_cal['ms2_unit'],
                    'adjusted_tolerance': mz_cal['ms2_mean'] + 3 * mz_cal['ms2_sd'],
                    'window_halfwidth_multiplier': 3.0
                },
                'rt_calibration': rt_cal
            }
            
            # Save calibration JSON
            cal_file = FastXCorr.get_calibration_filename(output_file)
            FastXCorr.save_calibration_json(calibration, cal_file)
            print(f"\n  Calibration parameters saved: {cal_file}")
            
            # Generate QC plots
            print("\n  Generating QC plots...")
            qc_output_prefix = os.path.splitext(output_file)[0] + '.calibration'
            
            plot_mass_accuracy_histograms(
                filtered_qc['ms1_mass_errors'],
                filtered_qc['ms2_mass_errors'],
                qc_output_prefix,
                ms1_unit=mz_cal['ms1_unit'],
                ms2_unit=mz_cal['ms2_unit'],
                num_precursors=num_confident
            )
            
            plot_rt_correlation(filtered_qc['rt_pairs'], qc_output_prefix)
            
            print("\n" + "="*80)
            print("CALIBRATION COMPLETE")
            print("="*80)
            
            return calibration
        else:
            print(f"✗ Only {num_confident} confident peptides (need ≥100)")
            
            if attempt == 0:
                print(f"  Will retry with {cal_library_peptides * 2} peptides...")
            else:
                # Failed after doubling
                raise RuntimeError(
                    f"\nCalibration failed: Only {num_confident} peptides passed 1% FDR after searching "
                    f"{num_peptides} library peptides.\n\n"
                    f"This suggests:\n"
                    f"  - Library may not match sample\n"
                    f"  - Sample quality issues\n"
                    f"  - Incorrect search parameters\n\n"
                    f"Please review your library and sample, or try a larger calibration set."
                )


# Worker function for parallel processing of isolation windows
def process_isolation_window_worker(args):
    """
    Worker function to process one isolation window in parallel.

    This function is designed to be called by multiprocessing.Pool.
    Each worker creates its own FastXCorr instance to avoid pickling issues.

    Args:
        args: Tuple of (window_idx, total_windows, isolation_window, window_spectra,
                       fasta_file, target_decoy_pairs, charge_states,
                       parquet_file, enzyme, decoy_cycle_length,
                       library_path, ms1_spectra, lib_fragment_tol_ppm, lib_precursor_tol_ppm,
                       lib_fragment_tol_unit, lib_precursor_tol_unit, calibration, verbose)

    Returns:
        Dictionary with search results and metadata
    """
    (window_idx, total_windows, isolation_window, window_spectra,
     fasta_file, target_decoy_pairs_data, charge_states,
     parquet_file, enzyme, decoy_cycle_length,
     library, ms1_spectra, lib_fragment_tol_ppm, lib_precursor_tol_ppm,
     lib_fragment_tol_unit, lib_precursor_tol_unit, calibration, verbose) = args

    # Create a fresh FastXCorr instance for this worker process
    xcorr_engine = FastXCorr()

    # Reconstruct target_decoy_pairs from serialized data
    # (PeptideCandidate objects need to be reconstructed)
    # Check if charge is included in data (for calibration mode with library sampling)
    has_charge = len(target_decoy_pairs_data) > 0 and 'charge' in target_decoy_pairs_data[0][0]
    
    target_decoy_pairs = []
    for target_data, decoy_data in target_decoy_pairs_data:
        if has_charge:
            # Calibration mode: charge is specified per peptide
            target = PeptideCandidate(
                sequence=target_data['sequence'],
                protein_id=target_data['protein_id'],
                mass=target_data['mass'],
                charge=target_data['charge']
            )
            decoy = PeptideCandidate(
                sequence=decoy_data['sequence'],
                protein_id=decoy_data['protein_id'],
                mass=decoy_data['mass'],
                charge=decoy_data['charge']
            )
        else:
            # Normal mode: charge will be iterated in search function
            target = PeptideCandidate(
                sequence=target_data['sequence'],
                protein_id=target_data['protein_id'],
                mass=target_data['mass']
            )
            decoy = PeptideCandidate(
                sequence=decoy_data['sequence'],
                protein_id=decoy_data['protein_id'],
                mass=decoy_data['mass']
            )
        target_decoy_pairs.append((target, decoy))

    # Library object is passed directly (no loading needed per worker)
    # This eliminates redundant parquet file reads across all workers

    # Only print worker start message if verbose >= 1
    if verbose >= 1:
        print(f"\n[Worker {window_idx+1}/{total_windows}] Processing isolation window: "
              f"[{isolation_window[0]:.4f}-{isolation_window[1]:.4f}] m/z, "
              f"{len(window_spectra)} spectra")

    # Determine if this is calibration mode: no calibration input + no parquet output
    is_calibration_mode = (calibration is None and parquet_file is None and library is not None)

    # Perform DIA peptide-centric search
    search_result = xcorr_engine.search_dia_peptide_centric(
        window_spectra,
        target_decoy_pairs,
        charge_states,
        parquet_output=parquet_file,
        library=library,
        ms1_spectra=ms1_spectra,
        lib_fragment_tol_ppm=lib_fragment_tol_ppm,
        lib_precursor_tol_ppm=lib_precursor_tol_ppm,
        lib_fragment_tol_unit=lib_fragment_tol_unit,
        lib_precursor_tol_unit=lib_precursor_tol_unit,
        calibration=calibration,  # Pass calibration for RT filtering and adjusted tolerances
        skip_xcorr_matrix=is_calibration_mode,  # Skip XCorr during calibration, calculate during full search
        verbose=verbose
    )

    # Only print completion message if verbose >= 1
    if verbose >= 1:
        print(f"[Worker {window_idx+1}/{total_windows}] Completed: "
              f"{len(search_result['results'])} peptide-charge combinations")

    return search_result


def write_pin_file(results_df: pd.DataFrame, output_path: str, library_mode: bool = True):
    """
    Write results in Percolator/Mokapot PIN format.
    
    Args:
        results_df: DataFrame with DIA search results
        output_path: Path to output PIN file
        library_mode: Whether this is library-based search
    """
    print(f"\nWriting Mokapot PIN file: {output_path}")
    
    pin_data = []
    for _, row in results_df.iterrows():
        # Determine if target or decoy
        is_target = not row['Peptide'].startswith('DECOY_')
        
        # Create unique PSM ID
        psm_id = f"{row['Peptide']}_{row['Charge']}_{row['ScanID']}"
        
        pin_row = {
            'PSMId': psm_id,
            'Label': 1 if is_target else -1,
            'ScanNr': row['ScanID'],
            'ExpMass': row['Mass'],
            'CalcMass': row['Mass'],  # For DIA, these are the same
            'Peptide': row['Peptide'],
            'Proteins': row['ProteinID'],
        }
        
        # Add feature columns based on library mode
        if library_mode:
            pin_row['LibCosine'] = row.get('LibCosine', 0.0)
            pin_row['LibCosineZScore'] = row.get('LibCosineZScore', 0.0)
            pin_row['XCorr'] = row.get('XCorr', 0.0)
            pin_row['PrecursorCosine'] = row.get('PrecursorCosine', 0.0)
            
            # Add delta columns if available
            if 'delta_mz_ppm_precursor' in row:
                pin_row['absDeltaMzPpmPrecursor'] = abs(row['delta_mz_ppm_precursor'])
            if 'delta_mz_ppm_fragments' in row:
                pin_row['absDeltaMzPpmFragments'] = abs(row['delta_mz_ppm_fragments'])
            if 'delta_rt' in row:
                pin_row['absDeltaRT'] = abs(row['delta_rt'])
        else:
            pin_row['XCorr'] = row.get('BestXCorr', 0.0)
            pin_row['XCorrZScore'] = row.get('XCorrZScore', 0.0)
        
        pin_data.append(pin_row)
    
    # Write to file
    pin_df = pd.DataFrame(pin_data)
    pin_df.to_csv(output_path, sep='\t', index=False)
    print(f"  Wrote {len(pin_df)} PSMs to PIN file")


def run_mokapot(results_df: pd.DataFrame, library_mode: bool = True, n_workers: int = 4) -> pd.DataFrame:
    """
    Run Mokapot rescoring on search results.
    
    Args:
        results_df: DataFrame with DIA search results (paired target/decoy format)
        library_mode: Whether this is library-based search
        n_workers: Number of parallel workers for Mokapot
        
    Returns:
        DataFrame with added Mokapot q-value columns
    """
    print("\nRunning Mokapot for peptide-level FDR estimation...")
    
    try:
        import mokapot
    except ImportError:
        print("ERROR: mokapot not installed. Install with: pip install mokapot")
        print("Skipping Mokapot rescoring.")
        return results_df
    
    # Unroll paired target/decoy format into separate rows for Mokapot
    psm_data = []
    for idx, row in results_df.iterrows():
        # Create unique ID for this precursor (peptide+charge)
        precursor_id = f"{row['Peptide']}_{row['Charge']}"
        
        # Target PSM
        target_psm = {
            'PSMId': f"{precursor_id}_target",
            'PrecursorId': precursor_id,
            'Label': 1,  # Target
            'ScanNr': row['ScanID'],
            'Peptide': row['Peptide'],
            'Proteins': row['ProteinID'],
            'RowIndex': idx  # Track original row for merging back
        }
        
        # Decoy PSM
        decoy_psm = {
            'PSMId': f"{precursor_id}_decoy",
            'PrecursorId': precursor_id,
            'Label': -1,  # Decoy
            'ScanNr': row.get('DecoyScanID', row['ScanID']),  # Use decoy scan if available
            'Peptide': row.get('DecoyPeptide', f"DECOY_{row['Peptide']}"),
            'Proteins': row['ProteinID'],
            'RowIndex': idx
        }
        
        # Add features for both target and decoy
        if library_mode:
            # Target features
            target_psm['LibCosine'] = row.get('LibCosine', 0.0)
            target_psm['LibCosineZScore'] = row.get('LibCosineZScore', 0.0)
            target_psm['XCorr'] = row.get('XCorr', 0.0)
            target_psm['PrecursorCosine'] = row.get('PrecursorCosine', 0.0)
            
            # Decoy features
            decoy_psm['LibCosine'] = row.get('DecoyLibCosine', 0.0)
            decoy_psm['LibCosineZScore'] = row.get('DecoyLibCosineZScore', 0.0)
            decoy_psm['XCorr'] = row.get('DecoyXCorr', 0.0)
            decoy_psm['PrecursorCosine'] = row.get('DecoyPrecursorCosine', 0.0)
            
            feature_cols = ['LibCosine', 'LibCosineZScore', 'XCorr', 'PrecursorCosine']
            
            # Add delta features if present (only for targets with calibration)
            if pd.notna(row.get('delta_rt')):
                target_psm['absDeltaRT'] = abs(row['delta_rt'])
                feature_cols.append('absDeltaRT')
            if pd.notna(row.get('delta_mz_ppm_precursor')):
                target_psm['absDeltaMzPpmPrecursor'] = abs(row['delta_mz_ppm_precursor'])
                feature_cols.append('absDeltaMzPpmPrecursor')
            if pd.notna(row.get('delta_mz_ppm_fragments')):
                target_psm['absDeltaMzPpmFragments'] = abs(row['delta_mz_ppm_fragments'])
                feature_cols.append('absDeltaMzPpmFragments')
        else:
            target_psm['XCorr'] = row.get('BestXCorr', 0.0)
            target_psm['XCorrZScore'] = row.get('XCorrZScore', 0.0)
            decoy_psm['XCorr'] = row.get('DecoyXCorr', 0.0)
            decoy_psm['XCorrZScore'] = row.get('DecoyXCorrZScore', 0.0)
            feature_cols = ['XCorr', 'XCorrZScore']
        
        psm_data.append(target_psm)
        psm_data.append(decoy_psm)
    
    psm_df = pd.DataFrame(psm_data)
    
    # Check for missing values in features
    missing_features = []
    for col in feature_cols:
        if col in psm_df.columns and psm_df[col].isna().any():
            missing_features.append(col)
    
    if missing_features:
        print(f"Missing values detected in the following features:")
        for feat in missing_features:
            print(f"  - {feat}")
        print("Dropping features with missing values...")
        feature_cols = [f for f in feature_cols if f not in missing_features]
    
    if not feature_cols:
        print("ERROR: No valid features remaining after dropping missing values.")
        print("Returning results without Mokapot scores.")
        return results_df
    
    # Create Mokapot dataset
    try:
        psms = mokapot.LinearPsmDataset(
            psm_df,
            target_column='Label',
            spectrum_columns='ScanNr',
            peptide_column='Peptide',
            feature_columns=feature_cols
        )
        
        # Run Mokapot
        n_targets = (psm_df['Label'] == 1).sum()
        n_decoys = (psm_df['Label'] == -1).sum()
        print(f"  Training Mokapot model: {n_targets} targets, {n_decoys} decoys, {len(feature_cols)} features")
        
        results, models = mokapot.brew(psms, max_workers=n_workers, test_fdr=0.01)
        
        # Get results
        psm_results_df = results.psms
        peptide_df = results.peptides
        
        num_precursors_01fdr = len(psm_results_df[psm_results_df['mokapot q-value'] <= 0.01])
        num_peptides_01fdr = len(peptide_df[peptide_df['mokapot q-value'] <= 0.01])
        print(f"  Mokapot complete: {num_precursors_01fdr} precursors, {num_peptides_01fdr} peptides at 1% FDR")
        
        # Merge Mokapot precursor-level scores back (use 'PrecursorId' to map)
        precursor_qvalues = {}
        precursor_scores = {}
        for _, psm_row in psm_results_df.iterrows():
            prec_id = psm_row['PSMId'].rsplit('_', 1)[0]  # Remove _target or _decoy suffix
            qval = psm_row['mokapot q-value']
            score = psm_row['mokapot score']
            
            # Keep best (lowest q-value) for each precursor
            if prec_id not in precursor_qvalues or qval < precursor_qvalues[prec_id]:
                precursor_qvalues[prec_id] = qval
                precursor_scores[prec_id] = score
        
        results_df['mokapot_precursor_qvalue'] = results_df.apply(
            lambda row: precursor_qvalues.get(f"{row['Peptide']}_{row['Charge']}", None),
            axis=1
        )
        results_df['mokapot_precursor_score'] = results_df.apply(
            lambda row: precursor_scores.get(f"{row['Peptide']}_{row['Charge']}", None),
            axis=1
        )
        
        # Map peptide-level scores
        peptide_qvalues = peptide_df.set_index('mokapot peptide')['mokapot q-value'].to_dict()
        peptide_scores = peptide_df.set_index('mokapot peptide')['mokapot score'].to_dict()
        
        results_df['mokapot_peptide_qvalue'] = results_df['Peptide'].map(peptide_qvalues)
        results_df['mokapot_peptide_score'] = results_df['Peptide'].map(peptide_scores)
        
        return results_df
        
    except Exception as e:
        print(f"ERROR running Mokapot: {e}")
        print("Returning results without Mokapot scores.")
        return results_df


def main():
    """Main function to run the Comet-style fast XCorr search."""
    import time
    workflow_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Comet-style Fast XCorr Database Search with Target-Decoy Competition')
    parser.add_argument('fasta_file', help='FASTA file containing protein sequences')
    parser.add_argument('mzml_file', help='mzML file containing mass spectra')
    parser.add_argument('--output', '-o', default='', help='Output file (pepXML format). If not specified, uses mzML filename with .pepXML extension')
    parser.add_argument('--pin_output', '-p', default='', help='Percolator Input (PIN) output file. If not specified, uses mzML filename with .pin extension')
    parser.add_argument('--dia_mode', action='store_true', 
                       help='Enable DIA peptide-centric search mode (experimental)')
    parser.add_argument('--dia_output', default='', 
                       help='DIA results output file. If not specified, uses mzML filename with .dia.tsv extension')
    parser.add_argument('--threads', '-t', type=int, default=0,
                       help='Number of threads for parallel processing in DIA mode. 0 = auto-detect (use all available cores - 1). Default: 0')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase output verbosity. Default: show once-per-window messages. Use -v to show all progress including batch writes.')
    parser.add_argument('--top_hits', '-n', type=int, default=10, 
                       help='Number of top hits to report per spectrum (distributed across charge states)')
    parser.add_argument('--max_spectra', '-m', type=int, default=0, 
                       help='Maximum number of MS2 spectra to process (0 = process all)')
    parser.add_argument('--charge_states', '-c', type=str, default='2,3',
                       help='Comma-separated list of charge states to consider (default: 2,3)')
    parser.add_argument('--enzyme', '-e', type=str, default='trypsin',
                       help='Enzyme for protein digestion (default: trypsin). Options: trypsin, trypsin_no_proline, lysc, lysn, argc, aspn, cnbr, gluc, pepsina, chymotrypsin')
    parser.add_argument('--missed_cleavages', type=int, default=1,
                       help='Maximum number of missed cleavages (default: 1)')
    parser.add_argument('--min_peptide_length', type=int, default=7,
                       help='Minimum peptide length in amino acids (default: 7)')
    parser.add_argument('--max_peptide_length', type=int, default=30,
                       help='Maximum peptide length in amino acids (default: 30)')
    parser.add_argument('--decoy_cycle_length', '-d', type=int, default=1,
                       help='Number of amino acids to cycle for decoy generation (default: 1)')
    parser.add_argument('--static_mods', '-s', type=str, default='C:57.021464',
                       help='Static modifications as AA:mass pairs separated by commas (default: C:57.021464 for carbamidomethylation). Use "none" for no modifications.')
    parser.add_argument('--bin_width', '-bw', type=float, default=1.0005079,
                       help='Mass bin width in Th for spectrum binning (default: 1.0005079, Comet default)')
    parser.add_argument('--bin_offset', '-bo', type=float, default=0.4,
                       help='Bin offset for mass binning calculation (default: 0.4, Comet default)')
    # Library-related arguments
    parser.add_argument('--speclib', type=str, default='',
                       help='DIA-NN spectrum library file (parquet format) for library-based scoring')
    parser.add_argument('--lib_fragment_tol', type=float, default=10.0,
                       help='Fragment m/z tolerance for library matching (default: 10.0 ppm)')
    parser.add_argument('--lib_fragment_tol_unit', type=str, default='ppm', choices=['ppm', 'mz'],
                       help='Fragment tolerance unit: ppm or mz (default: ppm)')
    parser.add_argument('--lib_precursor_tol', type=float, default=10.0,
                       help='Precursor m/z tolerance for MS1 isotope matching (default: 10.0 ppm)')
    parser.add_argument('--lib_precursor_tol_unit', type=str, default='ppm', choices=['ppm', 'mz'],
                       help='Precursor tolerance unit: ppm or mz (default: ppm)')
    parser.add_argument('--test_library_peptides', type=int, default=0,
                       help='For testing: randomly select N precursors from library (target, Q<=0.01) with fixed seed (default: 0 = all)')
    
    # Calibration arguments
    parser.add_argument('--auto_calibrate', action='store_true',
                       help='Automatically calibrate m/z and RT windows using library subset (recommended for DIA)')
    parser.add_argument('--cal_library_peptides', type=int, default=2000,
                       help='Number of high-quality library peptides to use for calibration (default: 2000)')
    parser.add_argument('--use_calibration', type=str, default='',
                       help='Use existing calibration JSON file from previous run')
    parser.add_argument('--calibration_only', action='store_true',
                       help='Only perform calibration, do not run full search')
    
    # Mokapot integration arguments
    parser.add_argument('--output_pin', type=str, default='',
                       help='Save Mokapot PIN file for external processing (optional)')
    parser.add_argument('--skip_mokapot', action='store_true',
                       help='Skip Mokapot rescoring (not recommended, default: False)')

    args = parser.parse_args()
    
    # Print command line for reproducibility
    import sys
    print(f"Command: {' '.join(sys.argv)}")
    print()
    
    # Parse static modifications
    static_modifications = {}
    if args.static_mods.lower() != 'none':
        try:
            for mod_str in args.static_mods.split(','):
                mod_str = mod_str.strip()
                if ':' in mod_str:
                    aa, mass_str = mod_str.split(':', 1)
                    aa = aa.strip().upper()
                    mass = float(mass_str.strip())
                    static_modifications[aa] = mass
        except ValueError as e:
            print(f"Error parsing static modifications '{args.static_mods}': {e}")
            print("Format should be: AA:mass,AA:mass (e.g., C:57.021464,M:15.994915)")
            sys.exit(1)
    
    # Parse charge states
    charge_states = [int(c.strip()) for c in args.charge_states.split(',')]
    
    # Validate enzyme
    if args.enzyme not in ['trypsin', 'trypsin_no_proline', 'lysc', 'lysn', 'argc', 'aspn', 'cnbr', 'gluc', 'pepsina', 'chymotrypsin']:
        print(f"Error: Unknown enzyme '{args.enzyme}'")
        print("Available enzymes: trypsin, trypsin_no_proline, lysc, lysn, argc, aspn, cnbr, gluc, pepsina, chymotrypsin")
        sys.exit(1)
    
    # Start total analysis timer
    import time
    total_start_time = time.time()
    
    print(f"Using charge states: {charge_states}")
    print("pyXcorrDIA: A simple python search tool for DIA data")
    
    # Initialize engine to get enzyme description
    xcorr_engine = FastXCorr(bin_width=args.bin_width, bin_offset=args.bin_offset, static_modifications=static_modifications)
    enzyme_desc = xcorr_engine.enzymes[args.enzyme]['description']
    
    print(f"- Enzyme: {args.enzyme} ({enzyme_desc})")
    print(f"- Missed cleavages: {args.missed_cleavages}")
    print(f"- Decoy generation: cycling {args.decoy_cycle_length} amino acid(s)")
    print(f"- Bin width: {args.bin_width:.7f} Th")
    print(f"- Bin offset: {args.bin_offset:.1f}")
    
    # Display static modifications
    if static_modifications:
        print("- Static modifications:")
        for aa, mass in static_modifications.items():
            print(f"    {aa}: +{mass:.6f} Th")
    else:
        print("- Static modifications: None")
    print(f"- Peptide length range: {args.min_peptide_length}-{args.max_peptide_length} amino acids")
    
    # Determine output filename
    if not args.output:
        base_name = os.path.splitext(args.mzml_file)[0]
        args.output = base_name + '.pepXML'
    
    # Determine PIN output filename
    if not args.pin_output:
        base_name = os.path.splitext(args.mzml_file)[0]
        args.pin_output = base_name + '.pin'
    
    # Check if using spectrum library for DIA mode - load it early to set defaults
    library_mode = args.dia_mode and args.speclib
    
    if library_mode:
        lib_start = time.time()
        print(f"\nLoading spectrum library: {args.speclib}")
        if args.test_library_peptides > 0:
            print(f"  Test mode: Randomly selecting {args.test_library_peptides} precursors (target, Q.Value<=0.01, seed=42)")
        library = SpectrumLibrary(args.speclib, test_limit_peptides=args.test_library_peptides)
        
        # Extract unique peptide sequences and metadata from library
        lib_elapsed = time.time() - lib_start
        print(f"  Library loaded in {lib_elapsed:.1f}s")

        print("Extracting peptides from spectrum library...")
        library_sequences = set()
        library_charge_states = set()
        library_peptide_lengths = []
        
        for (sequence, charge), precursor_data in library.peptide_index.items():
            library_sequences.add(sequence)
            library_charge_states.add(int(charge))  # Convert numpy int64 to Python int
            library_peptide_lengths.append(len(sequence))
        
        # Show relationship between precursors and unique peptides
        n_precursors = len(library.peptide_index)
        n_unique_peptides = len(library_sequences)
        print(f"  {n_precursors} precursors = {n_unique_peptides} unique peptides selected from library")
        
        # Set peptide length range from library if not explicitly set by user
        # Check if user provided non-default values
        user_set_min = '--min_peptide_length' in sys.argv
        user_set_max = '--max_peptide_length' in sys.argv
        
        if library_peptide_lengths:
            lib_min_length = min(library_peptide_lengths)
            lib_max_length = max(library_peptide_lengths)
            
            if not user_set_min:
                args.min_peptide_length = lib_min_length
            if not user_set_max:
                args.max_peptide_length = lib_max_length
            
            print(f"  Library peptide length range: {lib_min_length}-{lib_max_length}")
            if user_set_min or user_set_max:
                print(f"  Using command-line length range: {args.min_peptide_length}-{args.max_peptide_length}")
        
        # Handle charge states: use command-line if specified, otherwise use library defaults
        if args.charge_states == '2,3':  # Default value - use library charges
            charge_states = sorted(library_charge_states)
            print(f"  Using library charge states: {charge_states}")
        else:
            # Command-line override - filter to only library charges that were requested
            requested_charges = set(charge_states)
            available_charges = requested_charges & library_charge_states
            if not available_charges:
                print(f"  WARNING: Requested charges {charge_states} not in library {sorted(library_charge_states)}")
                print(f"  Using library charge states instead: {sorted(library_charge_states)}")
                charge_states = sorted(library_charge_states)
            else:
                charge_states = sorted(available_charges)
                print(f"  Restricting to requested charge states: {charge_states}")
    
    # For library mode, skip FASTA reading and protein mapping
    # We'll map high-scoring peptides to proteins after scoring
    if library_mode:
        print("\n*** Library-based search: Skipping FASTA digestion ***")
        print("  Proteins will be mapped after scoring for high-confidence peptides")
        
        # Create PeptideCandidate objects for library peptides without protein mapping
        print("  Creating peptide candidates from library...")
        all_target_peptides = []
        for sequence in library_sequences:
            # Skip length filtering - library defines the peptides
            # Use generic protein ID - will be mapped later for significant hits
            protein_id = f"LIBRARY_{sequence}"
            mass = xcorr_engine.calculate_peptide_mass(sequence)
            peptide = PeptideCandidate(sequence, protein_id, mass)
            all_target_peptides.append(peptide)
        
        print(f"  Created {len(all_target_peptides)} library peptide candidates")
        
        # Store proteins for later mapping, but don't process now
        proteins = None
    else:
        # Standard FASTA-based search workflow
        print("Reading FASTA file...")
        proteins = xcorr_engine.read_fasta(args.fasta_file)
        print(f"Loaded {len(proteins)} proteins")
        
        # Digest proteins to generate peptide candidates
        print("Digesting proteins...")
        all_target_peptides = []
        for protein_id, sequence in proteins.items():
            peptides = xcorr_engine.digest_protein(sequence, protein_id, 
                                                   enzyme=args.enzyme, 
                                                   missed_cleavages=args.missed_cleavages,
                                                   min_length=args.min_peptide_length,
                                                   max_length=args.max_peptide_length)
            all_target_peptides.extend(peptides)
        print(f"Generated {len(all_target_peptides)} target peptide candidates")
    
    # Make peptide list non-redundant (only needed for FASTA-based search)
    if library_mode:
        # Library peptides are already unique by design
        non_redundant_targets = all_target_peptides
    else:
        print("Making peptide list non-redundant...")
        non_redundant_targets = xcorr_engine.make_peptides_non_redundant(all_target_peptides)
        print(f"Non-redundant target peptides: {len(non_redundant_targets)} (removed {len(all_target_peptides) - len(non_redundant_targets)} duplicates)")
    
    print("Generating target-decoy pairs for competition...")
    target_decoy_pairs = xcorr_engine.generate_target_decoy_pairs(non_redundant_targets, args.decoy_cycle_length, args.enzyme)
    print(f"Target-decoy pairs: {len(target_decoy_pairs)} pairs ready for competition")
    
    # No need for separate peptide indexing - we'll search pairs directly
    mzml_start = time.time()
    print("Reading mzML file...")
    if args.max_spectra > 0:
        print(f"Limiting to first {args.max_spectra} MS2 spectra")
    
    # Check if DIA mode with library is enabled - use combined reader
    if args.dia_mode and args.speclib:
        print("Using combined single-pass mzML reader (MS1 + MS2)...")
        spectra, ms1_spectra = xcorr_engine.read_mzml_combined(args.mzml_file, args.max_spectra)
        mzml_elapsed = time.time() - mzml_start
        print(f"  Loaded {len(spectra)} MS2 spectra")
        print(f"  Loaded {len(ms1_spectra)} MS1 spectra for precursor isotope scoring")
        print(f"  mzML read completed in {mzml_elapsed:.1f}s (elapsed: {time.time()-workflow_start_time:.1f}s)")
    else:
        # Standard MS2-only read
        spectra = xcorr_engine.read_mzml(args.mzml_file, args.max_spectra)
        mzml_elapsed = time.time() - mzml_start
        print(f"  mzML read completed in {mzml_elapsed:.1f}s (elapsed: {time.time()-workflow_start_time:.1f}s)")
        ms1_spectra = None
    
    print(f"Processing {len(spectra)} MS2 spectra with Target-Decoy Competition")
    
    # Check if DIA mode is enabled
    if args.dia_mode:
        print("\n*** DIA PEPTIDE-CENTRIC SEARCH MODE ***")
        print("Strategy: Score ALL peptides in isolation window against ALL spectra in that window")
        print("Output: TSV file with paired target/decoy results (competition performed during analysis)")

        # Spectrum library already loaded above if provided
        # Workers will receive library object directly (no redundant loading)
        library_obj = library if library_mode else None

        # Determine DIA output filename
        if not args.dia_output:
            base_name = os.path.splitext(args.mzml_file)[0]
            args.dia_output = base_name + '.dia.tsv'

        print(f"- Summary results (TSV): {args.dia_output}")
        
        # Calibration workflow (if requested)
        calibration = None
        if library_mode and (args.auto_calibrate or args.use_calibration):
            if args.use_calibration:
                # Load existing calibration
                print(f"\nLoading calibration from: {args.use_calibration}")
                calibration = FastXCorr.load_calibration_json(args.use_calibration)
                print(f"  Loaded calibration from {calibration['calibration_metadata']['timestamp']}")
                print(f"  {calibration['calibration_metadata']['num_passing_fdr']} peptides used")
            elif args.auto_calibrate:
                # Run calibration workflow
                try:
                    calibration = run_calibration_workflow(
                        xcorr_engine=xcorr_engine,
                        library=library,
                        spectra=spectra,
                        ms1_spectra=ms1_spectra,
                        charge_states=charge_states,
                        cal_library_peptides=args.cal_library_peptides,
                        lib_fragment_tol=args.lib_fragment_tol,
                        lib_precursor_tol=args.lib_precursor_tol,
                        lib_fragment_tol_unit=args.lib_fragment_tol_unit,
                        lib_precursor_tol_unit=args.lib_precursor_tol_unit,
                        output_file=args.dia_output,
                        fasta_file=args.fasta_file,
                        enzyme=args.enzyme,
                        decoy_cycle_length=args.decoy_cycle_length,
                        verbose=args.verbose
                    )
                    
                    # If calibration-only mode, exit after calibration
                    if args.calibration_only:
                        print("\nCalibration complete. Exiting (--calibration_only mode).")
                        return
                        
                except RuntimeError as e:
                    print(f"\nERROR: {e}")
                    sys.exit(1)
        
        # Group spectra by isolation window
        print("\nGrouping spectra by isolation window...")
        window_groups = xcorr_engine.group_spectra_by_isolation_window(spectra)
        print(f"Found {len(window_groups)} unique isolation windows")
        
        # Determine number of worker processes
        if args.threads == 0:
            n_workers = max(1, cpu_count() - 1)  # Leave one core for system
        else:
            n_workers = max(1, args.threads)
        
        print(f"- Using {n_workers} parallel workers for processing isolation windows")
        
        # Serialize target_decoy_pairs for passing to workers
        # (PeptideCandidate objects need to be converted to dict form for pickling)
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
        
        # Prepare work items for parallel processing
        work_items = []
        for window_idx, (isolation_window, window_spectra) in enumerate(window_groups.items()):
            # No parquet files for full search (only TSV output)
            parquet_file = None
            
            work_items.append((
                window_idx,
                len(window_groups),
                isolation_window,
                window_spectra,
                args.fasta_file,
                target_decoy_pairs_data,
                charge_states,
                parquet_file,
                args.enzyme,
                args.decoy_cycle_length,
                library_obj,
                ms1_spectra,
                args.lib_fragment_tol if args.lib_fragment_tol_unit == 'ppm' else args.lib_fragment_tol,
                args.lib_precursor_tol if args.lib_precursor_tol_unit == 'ppm' else args.lib_precursor_tol,
                args.lib_fragment_tol_unit,
                args.lib_precursor_tol_unit,
                calibration,  # Pass calibration to workers
                args.verbose
            ))
        
        # Process isolation windows in parallel with incremental TSV writing
        print(f"\nProcessing {len(window_groups)} isolation windows in parallel...")
        
        # Initialize TSV file with header using DIAResultsWriter
        dia_writer = DIAResultsWriter(args.dia_output, args.mzml_file, library_mode=library_mode)
        with dia_writer:
            # Just write header, then close - actual results will be appended later
            pass
        
        all_dia_results = {}  # Keep for summary statistics
        completed_windows = 0
        
        # QC data aggregation from all workers
        all_qc_data = {
            'ms1_mass_errors': [],
            'ms2_mass_errors': [],
            'rt_pairs': [],
            'ms1_tol_unit': args.lib_precursor_tol_unit if library_mode else 'ppm',
            'ms2_tol_unit': args.lib_fragment_tol_unit if library_mode else 'ppm'
        }
        
        if n_workers == 1:
            # Sequential processing (for debugging or single-core machines)
            print("Running in single-threaded mode...")
            for item in work_items:
                search_result = process_isolation_window_worker(item)
                dia_results = search_result['results']
                all_dia_results.update(dia_results)
                
                # Aggregate QC data
                if 'qc_data' in search_result:
                    qc = search_result['qc_data']
                    all_qc_data['ms1_mass_errors'].extend(qc.get('ms1_mass_errors', []))
                    all_qc_data['ms2_mass_errors'].extend(qc.get('ms2_mass_errors', []))
                    all_qc_data['rt_pairs'].extend(qc.get('rt_pairs', []))
                    # Units are set once above, no need to update per worker
                
                # Write results immediately
                dia_writer.open_for_append()
                dia_writer.write_dia_results(dia_results)
                dia_writer.close()
                
                completed_windows += 1
                print(f"  Progress: {completed_windows}/{len(work_items)} windows completed")
        else:
            # Parallel processing with incremental writing
            from multiprocessing import Manager
            print(f"Running in parallel mode with {n_workers} workers...")
            
            # Create a lock for synchronized file writing
            manager = Manager()
            write_lock = manager.Lock()
            
            # Create writer with lock and library mode flag
            dia_writer_parallel = DIAResultsWriter(args.dia_output, args.mzml_file, write_lock, library_mode=library_mode)
            
            with Pool(n_workers) as pool:
                # Use imap_unordered to process results as they complete
                for search_result in pool.imap_unordered(process_isolation_window_worker, work_items):
                    dia_results = search_result['results']
                    all_dia_results.update(dia_results)
                    
                    # Aggregate QC data
                    if 'qc_data' in search_result:
                        qc = search_result['qc_data']
                        all_qc_data['ms1_mass_errors'].extend(qc.get('ms1_mass_errors', []))
                        all_qc_data['ms2_mass_errors'].extend(qc.get('ms2_mass_errors', []))
                        all_qc_data['rt_pairs'].extend(qc.get('rt_pairs', []))
                    
                    # Write results immediately with synchronized access
                    dia_writer_parallel.write_dia_results_synchronized(dia_results)
                    
                    completed_windows += 1
                    print(f"  Progress: {completed_windows}/{len(work_items)} windows completed")
        
        search_elapsed = time.time() - workflow_start_time
        print(f"\nAll {len(work_items)} isolation windows processed! (total elapsed: {search_elapsed:.1f}s)")
        
        # TSV results already written incrementally during processing
        
        # Perform target-decoy competition and calculate FDR
        print("\n=== TARGET-DECOY COMPETITION ANALYSIS ===")
        print("Reading TSV results to perform competition...")
        
        tsv_df = pd.read_csv(args.dia_output, sep='\t')
        
        # Determine if this is library mode or non-library mode
        is_library_mode = 'LibCosine' in tsv_df.columns
        
        if is_library_mode:
            # Library mode: each row has target and decoy scores
            # Perform competition: winner = peptide with higher LibCosine
            
            # Create separate rows for targets and decoys with competition labels
            competition_results = []
            for idx, row in tsv_df.iterrows():
                target_libcosine = row['LibCosine']
                decoy_libcosine = row['decoy_LibCosine']
                target_xcorr = row['XCorr']
                decoy_xcorr = row['decoy_XCorr']
                charge = row['Charge']
                
                # LibCosine competition
                if target_libcosine > decoy_libcosine:
                    competition_results.append({
                        'Peptide': row['Peptide'],
                        'Charge': charge,
                        'IsTarget': 'Target',
                        'LibCosine': target_libcosine,
                        'XCorr': target_xcorr
                    })
                elif decoy_libcosine > target_libcosine:
                    competition_results.append({
                        'Peptide': row['decoy_Peptide'],
                        'Charge': charge,
                        'IsTarget': 'Decoy',
                        'LibCosine': decoy_libcosine,
                        'XCorr': decoy_xcorr
                    })
                else:
                    # Tie - decoy wins (conservative)
                    competition_results.append({
                        'Peptide': row['decoy_Peptide'],
                        'Charge': charge,
                        'IsTarget': 'Decoy',
                        'LibCosine': decoy_libcosine,
                        'XCorr': decoy_xcorr
                    })
            
            winners_df = pd.DataFrame(competition_results)
            
            # Calculate FDR for LibCosine (primary score used in competition)
            def calculate_fdr_at_threshold(df, score_col, threshold=0.01):
                """Calculate number of precursors (peptide+charge) at a given FDR threshold."""
                df_sorted = df.sort_values(score_col, ascending=False).copy()
                df_sorted['cumulative_targets'] = (df_sorted['IsTarget'] == 'Target').cumsum()
                df_sorted['cumulative_decoys'] = (df_sorted['IsTarget'] == 'Decoy').cumsum()
                df_sorted['fdr'] = df_sorted['cumulative_decoys'] / df_sorted['cumulative_targets'].replace(0, 1)
                
                # Get max targets at or below FDR threshold
                valid = df_sorted[df_sorted['fdr'] <= threshold]
                if len(valid) > 0:
                    return int(valid['cumulative_targets'].max())
                return 0
            
            libcosine_1pct = calculate_fdr_at_threshold(winners_df, 'LibCosine', threshold=0.01)
            libcosine_5pct = calculate_fdr_at_threshold(winners_df, 'LibCosine', threshold=0.05)
            
            # Also calculate for XCorr (secondary score)
            xcorr_1pct = calculate_fdr_at_threshold(winners_df, 'XCorr', threshold=0.01)
            xcorr_5pct = calculate_fdr_at_threshold(winners_df, 'XCorr', threshold=0.05)
            
            print("\nLibrary Mode - Competition Results (LibCosine primary score):")
            print(f"  LibCosine at 1% FDR: {libcosine_1pct:,} precursors")
            print(f"  LibCosine at 5% FDR: {libcosine_5pct:,} precursors")
            print(f"  XCorr at 1% FDR:     {xcorr_1pct:,} precursors")
            print(f"  XCorr at 5% FDR:     {xcorr_5pct:,} precursors")
            
            # Print competition summary
            total_pairs = len(tsv_df)
            target_wins = len(winners_df[winners_df['IsTarget'] == 'Target'])
            decoy_wins = len(winners_df[winners_df['IsTarget'] == 'Decoy'])
            
            # Diagnostic: count unique peptide sequences vs precursors (peptide+charge)
            unique_peptides = tsv_df['Peptide'].nunique()
            unique_charges = tsv_df['Charge'].nunique() if 'Charge' in tsv_df.columns else 'N/A'
            
            print("\nCompetition Summary:")
            print(f"  Total precursors (rows in TSV): {total_pairs:,}")
            print(f"    - Unique peptides (sequences): {unique_peptides:,}")
            print(f"    - Unique charge states: {unique_charges}")
            print("    - Note: Precursor = peptide+charge (e.g., PEPTIDE+2, PEPTIDE+3 are 2 precursors)")
            print(f"  Target wins: {target_wins:,} ({target_wins/total_pairs*100:.1f}%)")
            print(f"  Decoy wins:  {decoy_wins:,} ({decoy_wins/total_pairs*100:.1f}%)")
        else:
            # Non-library mode: each row has target and decoy scores
            # Perform competition: winner = peptide with higher XCorr
            
            competition_results = []
            for idx, row in tsv_df.iterrows():
                target_xcorr = row['BestXCorr']
                decoy_xcorr = row['decoy_BestXCorr']
                
                # XCorr competition
                if target_xcorr > decoy_xcorr:
                    competition_results.append({
                        'Peptide': row['Peptide'],
                        'IsTarget': 'Target',
                        'BestXCorr': target_xcorr
                    })
                elif decoy_xcorr > target_xcorr:
                    competition_results.append({
                        'Peptide': row['decoy_Peptide'],
                        'IsTarget': 'Decoy',
                        'BestXCorr': decoy_xcorr
                    })
                else:
                    # Tie - decoy wins (conservative)
                    competition_results.append({
                        'Peptide': row['decoy_Peptide'],
                        'IsTarget': 'Decoy',
                        'BestXCorr': decoy_xcorr
                    })
            
            winners_df = pd.DataFrame(competition_results)
            
            def calculate_fdr_at_threshold(df, score_col, threshold=0.01):
                """Calculate number of precursors (peptide+charge) at a given FDR threshold."""
                df_sorted = df.sort_values(score_col, ascending=False).copy()
                df_sorted['cumulative_targets'] = (df_sorted['IsTarget'] == 'Target').cumsum()
                df_sorted['cumulative_decoys'] = (df_sorted['IsTarget'] == 'Decoy').cumsum()
                df_sorted['fdr'] = df_sorted['cumulative_decoys'] / df_sorted['cumulative_targets'].replace(0, 1)
                
                valid = df_sorted[df_sorted['fdr'] <= threshold]
                if len(valid) > 0:
                    return int(valid['cumulative_targets'].max())
                return 0
            
            xcorr_1pct = calculate_fdr_at_threshold(winners_df, 'BestXCorr', threshold=0.01)
            xcorr_5pct = calculate_fdr_at_threshold(winners_df, 'BestXCorr', threshold=0.05)
            
            print("\nNon-Library Mode - Competition Results (XCorr primary score):")
            print(f"  XCorr at 1% FDR: {xcorr_1pct:,} precursors")
            print(f"  XCorr at 5% FDR: {xcorr_5pct:,} precursors")
            
            # Print competition summary
            total_pairs = len(tsv_df)
            target_wins = len(winners_df[winners_df['IsTarget'] == 'Target'])
            decoy_wins = len(winners_df[winners_df['IsTarget'] == 'Decoy'])
            
            # Diagnostic: count unique peptide sequences vs precursors (peptide+charge)
            unique_peptides = tsv_df['Peptide'].nunique()
            unique_charges = tsv_df['Charge'].nunique() if 'Charge' in tsv_df.columns else 'N/A'
            
            print("\nCompetition Summary:")
            print(f"  Total precursors (rows in TSV): {total_pairs:,}")
            print(f"    - Unique peptides (sequences): {unique_peptides:,}")
            print(f"    - Unique charge states: {unique_charges}")
            print("    - Note: Precursor = peptide+charge (e.g., PEPTIDE+2, PEPTIDE+3 are 2 precursors)")
            print(f"  Target wins: {target_wins:,} ({target_wins/total_pairs*100:.1f}%)")
            print(f"  Decoy wins:  {decoy_wins:,} ({decoy_wins/total_pairs*100:.1f}%)")
        
        # Generate QC plots (after competition so we can filter by FDR)
        print("\n=== GENERATING QUALITY CONTROL PLOTS ===")
        qc_output_prefix = os.path.splitext(args.dia_output)[0]
        
        print("\nQC data collected (all target precursors scored):")
        print(f"  MS1 mass accuracy: {len(all_qc_data['ms1_mass_errors']):,} measurements")
        print(f"  MS2 mass accuracy: {len(all_qc_data['ms2_mass_errors']):,} measurements")
        print(f"  RT pairs (library vs measured): {len(all_qc_data['rt_pairs']):,}")
        
        # Filter QC data to only include peptides at <1% FDR
        if is_library_mode:
            filtered_qc_data = filter_qc_data_by_fdr(all_qc_data, winners_df, fdr_threshold=0.01)
        else:
            # For non-library mode, we don't have LibCosine so use all data
            filtered_qc_data = all_qc_data
        
        # Generate mass accuracy histograms
        num_precursors = filtered_qc_data.get('num_precursors')
        plot_mass_accuracy_histograms(
            filtered_qc_data['ms1_mass_errors'],
            filtered_qc_data['ms2_mass_errors'],
            qc_output_prefix,
            ms1_unit=filtered_qc_data['ms1_tol_unit'],
            ms2_unit=filtered_qc_data['ms2_tol_unit'],
            num_precursors=num_precursors
        )
        
        # Generate RT correlation plot (library mode only)
        if library_mode and len(filtered_qc_data['rt_pairs']) > 0:
            plot_rt_correlation(filtered_qc_data['rt_pairs'], qc_output_prefix)
        
        # Mokapot integration (library mode only, by default)
        if library_mode and not args.skip_mokapot:
            print("\n=== MOKAPOT RESCORING ===")
            
            # Use in-memory DataFrame from competition analysis
            # (tsv_df already loaded above)
            
            # Optional: Write PIN file
            if args.output_pin:
                write_pin_file(tsv_df, args.output_pin, library_mode=True)
            
            # Run Mokapot on in-memory DataFrame
            tsv_df = run_mokapot(tsv_df, library_mode=True, n_workers=n_workers)
            
            # Save updated results with Mokapot columns
            print(f"\nSaving results with Mokapot scores: {args.dia_output}")
            tsv_df.to_csv(args.dia_output, sep='\t', index=False)
            print("  Updated TSV file with mokapot_precursor_qvalue and mokapot_peptide_qvalue columns")
        
        # Print final summary
        print("\n=== DIA PEPTIDE-CENTRIC SEARCH COMPLETED ===")
        print(f"Summary (TSV): {args.dia_output}")
        
        # Print total analysis time
        total_elapsed = time.time() - total_start_time
        hours = int(total_elapsed // 3600)
        minutes = int((total_elapsed % 3600) // 60)
        seconds = total_elapsed % 60
        if hours > 0:
            print(f"\nTotal analysis time: {hours}h {minutes}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"\nTotal analysis time: {minutes}m {seconds:.1f}s")
        else:
            print(f"\nTotal analysis time: {seconds:.1f}s")
        
        
    else:
        # Standard spectrum-centric search mode
        print("Performing target-decoy competition search...")
        print(f"Writing results to {args.output}")
        print(f"Writing PIN results to {args.pin_output}")
        
        # Initialize pepXML and PIN writers and process spectra
        total_identifications = 0
        target_hits = 0
        decoy_hits = 0
    
        # Initialize pepXML and PIN writers and process spectra
        total_identifications = 0
        target_hits = 0
        decoy_hits = 0
        
        with PepXMLWriter(args.output, args.mzml_file, args.fasta_file) as pepxml_writer, \
             PINWriter(args.pin_output, args.mzml_file) as pin_writer:
            spectra_with_hits = 0
            for i, spectrum in enumerate(spectra):
                # Calculate isolation window info
                precursor_mz = spectrum.precursor_mz
                isolation_window_lower = spectrum.isolation_window_lower
                isolation_window_upper = spectrum.isolation_window_upper
                window_width = isolation_window_upper - isolation_window_lower
                
                # Count pairs in isolation window (for reporting)
                pairs_in_window = 0
                for target_peptide, decoy_peptide in target_decoy_pairs:
                    for charge in charge_states:
                        target_mz = (target_peptide.mass + charge * xcorr_engine.proton_mass) / charge
                        if isolation_window_lower <= target_mz <= isolation_window_upper:
                            pairs_in_window += 1
                            break  # Count each pair only once
                
                # Adaptive progress reporting: more frequent for smaller datasets
                if len(spectra) <= 100:
                    report_interval = 10
                elif len(spectra) <= 1000:
                    report_interval = 50
                else:
                    report_interval = 100
                
                if i % report_interval == 0 or i == 0:
                    print(f"Processing spectrum {i+1}/{len(spectra)} - Precursor: {precursor_mz:.4f} m/z, Window: [{isolation_window_lower:.5f}-{isolation_window_upper:.5f}] ({window_width:.5f} m/z), Pairs in window: {pairs_in_window} - {spectra_with_hits} spectra searched")
                
                # Search spectrum with target-decoy competition
                search_results = xcorr_engine.search_spectrum_target_decoy(spectrum, target_decoy_pairs, charge_states)
                
                # Count target vs decoy hits
                spectrum_target_hits = 0
                spectrum_decoy_hits = 0
                
                # Write results to both formats
                top_hits_per_charge = max(1, args.top_hits // len(charge_states))
                # Ensure we get exactly 3 hits per charge state when possible
                if args.top_hits >= 3 * len(charge_states):
                    top_hits_per_charge = 3
                
                # Write to pepXML (existing format)
                pepxml_writer.write_spectrum_query(spectrum, search_results, top_hits_per_charge)
                
                # Write to PIN (new format - best peptide per charge state only)
                pin_writer.write_spectrum_results(spectrum, search_results)
                
                if search_results:
                    spectra_with_hits += 1
                    # Count hits and track target vs decoy
                    hits_by_charge = {}
                    for peptide, score, e_value, charge in search_results:
                        if charge not in hits_by_charge:
                            hits_by_charge[charge] = 0
                        if hits_by_charge[charge] < top_hits_per_charge:
                            hits_by_charge[charge] += 1
                            total_identifications += 1
                            
                            # Count target vs decoy
                            if peptide.protein_id.startswith('decoy_'):
                                spectrum_decoy_hits += 1
                            else:
                                spectrum_target_hits += 1
                    
                    target_hits += spectrum_target_hits
                    decoy_hits += spectrum_decoy_hits
        
        print("Target-decoy competition search completed!")
        print(f"Total spectra processed: {len(spectra)}")
        print(f"Spectra with peptide matches: {spectra_with_hits}")
        print(f"Total identifications (competition winners): {total_identifications}")
        print(f"  Target winners: {target_hits}")
        print(f"  Decoy winners: {decoy_hits}")
        if total_identifications > 0:
            fdr_estimate = (decoy_hits / total_identifications) * 100
            print(f"  Estimated FDR: {fdr_estimate:.2f}%")
        print(f"pepXML results saved to: {args.output}")
        print(f"PIN results saved to: {args.pin_output}")


if __name__ == '__main__':
    import sys
    
    # Setup logging to file for DIA mode
    # Parse args to check if DIA mode and get mzML filename
    if '--dia_mode' in sys.argv:
        # Find mzML file argument (second positional arg)
        mzml_file = None
        for i, arg in enumerate(sys.argv):
            if i > 0 and not arg.startswith('-') and arg.endswith('.mzML'):
                mzml_file = arg
                break
        
        if mzml_file:
            import os
            log_file = os.path.splitext(mzml_file)[0] + '.dia.log'
            
            # Tee stdout/stderr to both console and log file
            class TeeLogger:
                def __init__(self, *files):
                    self.files = files
                def write(self, data):
                    for f in self.files:
                        f.write(data)
                        f.flush()
                def flush(self):
                    for f in self.files:
                        f.flush()
            
            log_handle = open(log_file, 'w')
            sys.stdout = TeeLogger(sys.stdout, log_handle)
            sys.stderr = TeeLogger(sys.stderr, log_handle)
            print(f"Logging output to: {log_file}")
    
    main()