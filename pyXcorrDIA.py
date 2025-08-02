#!/usr/bin/env python3
"""
Fast SEQUEST Cross-Correlation Implementation
Based on the paper by Eng et al. (2008): "A Fast SEQUEST Cross Correlation Algorithm"

This implementation calculates the cross-correlation score without FFTs,
enabling scoring of all candidate peptides and E-value calculation.

CORRECTED VERSION - Fixed XCorr calculation to match Comet exactly
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import math
from typing import List, Tuple, Dict, Optional
import argparse
import pymzml
from pyteomics import mgf
import xml.etree.ElementTree as ET
import os
import bisect
import sys
from datetime import datetime
from xml.dom import minidom
import os
from datetime import datetime


class MassSpectrum:
    """Represents a mass spectrum with m/z and intensity values."""
    
    def __init__(self, mz_array: np.ndarray, intensity_array: np.ndarray, 
                 scan_id: str = "", precursor_mz: float = 0.0, charge: int = 0,
                 isolation_window_lower: float = 0.0, isolation_window_upper: float = 0.0):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.scan_id = scan_id
        self.precursor_mz = precursor_mz
        self.charge = charge
        self.isolation_window_lower = isolation_window_lower
        self.isolation_window_upper = isolation_window_upper
        # Comet-style preprocessing results
        self.processed_spectrum: Optional[np.ndarray] = None  # After MakeCorrData windowing
        self.preprocessed_spectrum: Optional[np.ndarray] = None  # After fast XCorr preprocessing


class PeptideCandidate:
    """Represents a peptide candidate with its theoretical spectrum."""
    
    def __init__(self, sequence: str, protein_id: str, mass: float):
        self.sequence = sequence
        self.protein_id = protein_id
        self.mass = mass
        self.theoretical_spectrum = None


class FastXCorr:
    """
    Fast cross-correlation implementation based on Comet's approach.
    
    This class implements Comet's optimized cross-correlation calculation with:
    1. Spectrum binning (1.0005079 Da bins)
    2. MakeCorrData windowing normalization (10 windows, normalize to 50.0)
    3. Fast XCorr preprocessing with sliding window (offset=75)
    4. Simple dot product scoring (CORRECTED - no 0.005 scaling)
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
    
    def read_mzml(self, mzml_file: str, max_spectra: int = 0) -> List[MassSpectrum]:
        """Read mass spectra from mzML file using pymzml."""
        spectra = []
        
        # Open the mzML file with pymzml
        run = pymzml.run.Reader(mzml_file)
        
        for spectrum in run:
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
                    # Look for charge state in child elements
                    for child in element:
                        if 'charge' in child.attrib or 'selectedIonMZ' in child.tag:
                            for sub_child in child:
                                if 'charge' in sub_child.attrib:
                                    charge = int(sub_child.attrib['charge'])
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
            
            # Get scan ID
            scan_id = spectrum.ID
            if not scan_id:
                scan_id = f"scan_{spectrum.scan_time_in_minutes()}"
            
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
                spectra.append(mass_spectrum)
                
                # Stop reading if we've reached the maximum number of spectra
                if max_spectra > 0 and len(spectra) >= max_spectra:
                    break
        
        return spectra
    
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
    
    def generate_decoy_sequence(self, sequence: str, cycle_length: int = 1) -> str:
        """
        Generate decoy peptide sequence by cycling N amino acids (default 1).
        Keep the C-terminal K/R in place if present.
        
        Args:
            sequence: Original peptide sequence
            cycle_length: Number of positions to cycle (default 1)
            
        Returns:
            Decoy sequence with cycled amino acids
        """
        if len(sequence) <= 1:
            return sequence
            
        # Check if sequence ends with K or R
        if sequence[-1] in ['K', 'R']:
            # Keep C-terminal K/R in place, cycle the rest
            core_sequence = sequence[:-1]
            c_terminal = sequence[-1]
        else:
            # No tryptic C-terminus, cycle entire sequence
            core_sequence = sequence
            c_terminal = ''
            
        # If core sequence too short to cycle meaningfully
        if len(core_sequence) <= cycle_length:
            return sequence  # Return original if can't cycle
            
        # Cycle the sequence by moving first N amino acids to the end
        cycle_length = cycle_length % len(core_sequence)  # Handle cycle_length > sequence length
        if cycle_length == 0:
            cycled_core = core_sequence
        else:
            cycled_core = core_sequence[cycle_length:] + core_sequence[:cycle_length]
            
        return cycled_core + c_terminal
    
    def generate_reversed_decoy_sequence(self, sequence: str) -> str:
        """
        Generate decoy peptide sequence by reversing amino acids.
        Keep the C-terminal K/R in place if present.
        
        Args:
            sequence: Original peptide sequence
            
        Returns:
            Decoy sequence with reversed amino acids
        """
        if len(sequence) <= 1:
            return sequence
            
        # Check if sequence ends with K or R
        if sequence[-1] in ['K', 'R']:
            # Keep C-terminal K/R in place, reverse the rest
            core_sequence = sequence[:-1]
            c_terminal = sequence[-1]
        else:
            # No tryptic C-terminus, reverse entire sequence
            core_sequence = sequence
            c_terminal = ''
            
        # Reverse the core sequence
        reversed_core = core_sequence[::-1]
            
        return reversed_core + c_terminal
    
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
                                  cycle_length: int = 1) -> List[Tuple[PeptideCandidate, PeptideCandidate]]:
        """
        Generate target-decoy pairs for proper target-decoy competition.
        
        Uses reversal as the default decoy generation method (keeping C-terminal K/R fixed),
        with cycling as a fallback if reversal fails to generate a valid decoy.
        
        Args:
            target_peptides: List of target peptides (should be non-redundant)
            cycle_length: Number of positions to cycle for decoy generation (used in fallback)
            
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
            decoy_sequence = self.generate_reversed_decoy_sequence(target_peptide.sequence)
            
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
                    decoy_sequence = self.generate_decoy_sequence(target_peptide.sequence, retry_cycle)
                    
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
        print(f"Target-decoy pair generation summary:")
        print(f"  Target peptides: {len(target_peptides)}")
        print(f"  Target-decoy pairs created: {pairs_created}")
        print(f"  Collisions resolved: {collisions_resolved}")
        print(f"  Cycling fallback used: {cycling_fallback_used}")
        print(f"  Peptides without valid decoys: {max_retries_exceeded}")
        
        return target_decoy_pairs
    
    def digest_protein(self, sequence: str, protein_id: str, 
                      enzyme: str = 'trypsin', missed_cleavages: int = 2) -> List[PeptideCandidate]:
        """Digest protein sequence into peptides."""
        peptides = []
        
        if enzyme == 'trypsin':
            # Trypsin cleaves after K and R, but not before P
            cleavage_pattern = r'(?<=[KR])(?!P)'
        else:
            raise ValueError(f"Enzyme {enzyme} not supported")
        
        # Split sequence at cleavage sites
        fragments = re.split(cleavage_pattern, sequence)
        
        # Generate peptides with missed cleavages
        for i in range(len(fragments)):
            for j in range(i, min(i + missed_cleavages + 1, len(fragments))):
                peptide_seq = ''.join(fragments[i:j+1])
                
                # Filter by length (typically 6-50 amino acids)
                if 6 <= len(peptide_seq) <= 50:
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
        import bisect
        
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
    
    def calculate_fast_xcorr(self, theoretical_spectrum: np.ndarray, 
                           preprocessed_experimental: np.ndarray) -> float:
        """
        Calculate fast cross-correlation score using Comet's approach.
        
        FINAL CORRECTION: Apply the 0.005 scaling factor correctly.
        From Comet source: k = (int)(dFastXcorr*10.0*0.005 + 0.5) and comment 0.005=50/10000
        This means the final XCorr score SHOULD be scaled by 0.005.
        """
        # Ensure both spectra have the same length
        min_len = min(len(theoretical_spectrum), len(preprocessed_experimental))
        
        # Calculate dot product (Comet's core XCorr calculation)
        raw_xcorr = np.dot(theoretical_spectrum[:min_len], 
                          preprocessed_experimental[:min_len])
        
        # FINAL CORRECTION: Apply the 0.005 scaling factor
        # This is the scaling that Comet uses: 0.005 = 50/10000
        # It accounts for the 50.0 normalization and additional scaling for score range
        final_xcorr = raw_xcorr * 0.005
        
        # Round to reasonable precision like Comet
        final_xcorr = round(final_xcorr, 4)
        
        return final_xcorr
    
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
            
            if expect_value > 999.0:
                expect_value = 999.0
            
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


def main():
    """Main function to run the Comet-style fast XCorr search."""
    parser = argparse.ArgumentParser(description='Comet-style Fast XCorr Database Search with Target-Decoy Competition')
    parser.add_argument('fasta_file', help='FASTA file containing protein sequences')
    parser.add_argument('mzml_file', help='mzML file containing mass spectra')
    parser.add_argument('--output', '-o', default='', help='Output file (pepXML format). If not specified, uses mzML filename with .pepXML extension')
    parser.add_argument('--pin_output', '-p', default='', help='Percolator Input (PIN) output file. If not specified, uses mzML filename with .pin extension')
    parser.add_argument('--top_hits', '-n', type=int, default=10, 
                       help='Number of top hits to report per spectrum (distributed across charge states)')
    parser.add_argument('--max_spectra', '-m', type=int, default=0, 
                       help='Maximum number of MS2 spectra to process (0 = process all)')
    parser.add_argument('--charge_states', '-c', type=str, default='2,3',
                       help='Comma-separated list of charge states to consider (default: 2,3)')
    parser.add_argument('--decoy_cycle_length', '-d', type=int, default=1,
                       help='Number of amino acids to cycle for decoy generation (default: 1)')
    parser.add_argument('--static_mods', '-s', type=str, default='C:57.021464',
                       help='Static modifications as AA:mass pairs separated by commas (default: C:57.021464 for carbamidomethylation). Use "none" for no modifications.')
    parser.add_argument('--bin_width', '-bw', type=float, default=1.0005079,
                       help='Mass bin width in Th for spectrum binning (default: 1.0005079, Comet default)')
    parser.add_argument('--bin_offset', '-bo', type=float, default=0.4,
                       help='Bin offset for mass binning calculation (default: 0.4, Comet default)')
    
    args = parser.parse_args()
    
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
    print(f"Using charge states: {charge_states}")
    print("pyXcorrDIA: Using Comet XCorr with target-decoy competition")
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
    
    # Determine output filename
    if not args.output:
        base_name = os.path.splitext(args.mzml_file)[0]
        args.output = base_name + '.pepXML'
    
    # Determine PIN output filename
    if not args.pin_output:
        base_name = os.path.splitext(args.mzml_file)[0]
        args.pin_output = base_name + '.pin'
    
    # Initialize Comet-style XCorr engine with static modifications and bin parameters
    xcorr_engine = FastXCorr(bin_width=args.bin_width, bin_offset=args.bin_offset, static_modifications=static_modifications)
    
    print("Reading FASTA file...")
    proteins = xcorr_engine.read_fasta(args.fasta_file)
    print(f"Loaded {len(proteins)} proteins")
    
    print("Digesting proteins...")
    all_target_peptides = []
    for protein_id, sequence in proteins.items():
        peptides = xcorr_engine.digest_protein(sequence, protein_id)
        all_target_peptides.extend(peptides)
    print(f"Generated {len(all_target_peptides)} target peptide candidates")
    
    print("Making peptide list non-redundant...")
    non_redundant_targets = xcorr_engine.make_peptides_non_redundant(all_target_peptides)
    print(f"Non-redundant target peptides: {len(non_redundant_targets)} (removed {len(all_target_peptides) - len(non_redundant_targets)} duplicates)")
    
    print("Generating target-decoy pairs for competition...")
    target_decoy_pairs = xcorr_engine.generate_target_decoy_pairs(non_redundant_targets, args.decoy_cycle_length)
    print(f"Target-decoy pairs: {len(target_decoy_pairs)} pairs ready for competition")
    
    # No need for separate peptide indexing - we'll search pairs directly
    print("Reading mzML file...")
    if args.max_spectra > 0:
        print(f"Limiting to first {args.max_spectra} MS2 spectra")
    spectra = xcorr_engine.read_mzml(args.mzml_file, args.max_spectra)
    
    print(f"Processing {len(spectra)} MS2 spectra with Target-Decoy Competition")
    
    print("Performing target-decoy competition search...")
    print(f"Writing results to {args.output}")
    print(f"Writing PIN results to {args.pin_output}")
    
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
    main()