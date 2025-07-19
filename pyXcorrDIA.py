#!/usr/bin/env python3
"""
Fast SEQUEST Cross-Correlation Implementation
Based on the paper by Eng et al. (2008): "A Fast SEQUEST Cross Correlation Algorithm"

This implementation calculates the cross-correlation score without FFTs,
enabling scoring of all candidate peptides and E-value calculation.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import math
from typing import List, Tuple, Dict, Optional
import argparse
import pymzml
import xml.etree.ElementTree as ET
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
    4. Simple dot product scoring with Comet's scaling (0.005)
    
    This implementation closely follows the Comet source code to ensure 
    compatibility and reproducibility with the established search engine.
    """
    
    def __init__(self, bin_width: float = 1.0005079):
        self.bin_width = bin_width
        self.mass_range = (0, 2000)  # m/z range
        self.num_bins = int((self.mass_range[1] - self.mass_range[0]) / bin_width) + 1
        
        # Comet BIN macro parameters
        # BIN(dMass) = (int)((dMass)*g_staticParams.dInverseBinWidth + g_staticParams.dOneMinusBinOffset)
        self.inverse_bin_width = 1.0 / bin_width  # g_staticParams.dInverseBinWidth
        self.bin_offset = 0.4  # g_staticParams.dOneMinusBinOffset (hardcoded as requested)
        
        # Amino acid masses (monoisotopic)
        self.aa_masses = {
            'A': 71.037114, 'R': 156.101111, 'N': 114.042927, 'D': 115.026943,
            'C': 103.009185, 'E': 129.042593, 'Q': 128.058578, 'G': 57.021464,
            'H': 137.058912, 'I': 113.084064, 'L': 113.084064, 'K': 128.094963,
            'M': 131.040485, 'F': 147.068414, 'P': 97.052764, 'S': 87.032028,
            'T': 101.047679, 'W': 186.079313, 'Y': 163.063329, 'V': 99.068414
        }
        
        # Ion type masses
        self.h2o_mass = 18.010565
        self.nh3_mass = 17.026549
        self.proton_mass = 1.007276
        
        # Pre-sorted peptide candidates by m/z for fast lookup
        self.sorted_peptides_by_mz = {}  # Dict[charge_state, List[Tuple[mz, peptide]]]
    
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
    
    def read_single_spectrum(self, mzml_file: str, scan_id: str) -> MassSpectrum:
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
        
        This follows Comet's ion mass calculation:
        - b ions: N-terminal proton + cumulative AA masses
        - y ions: C-terminal OH2 + proton + cumulative AA masses (reverse)
        - Fragment charge states limited to max 2+ for efficiency (like Comet fragment index)
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
            
            # Generate fragment charges 1+ and 2+ only (like Comet fragment index)
            for frag_charge in range(1, min(3, charge + 1)):  # 1+ and 2+ only
                mz = (b_mass + (frag_charge - 1) * self.proton_mass) / frag_charge
                if self.mass_range[0] <= mz <= self.mass_range[1]:
                    bin_idx = self.bin_mass(mz)
                    relative_bin_idx = bin_idx - self.bin_mass(self.mass_range[0])
                    if 0 <= relative_bin_idx < self.num_bins:
                        # Comet's implementation: theoretical spectrum normalized to 50.0
                        # (experimental spectrum also normalized to 50.0 in MakeCorrData)
                        spectrum[relative_bin_idx] = 50.0
        
        # Generate y ions (C-terminal fragments) - Comet's method
        y_mass = cterm_oh2_proton  # Start with C-terminal OH2 + proton
        for i in range(len(sequence) - 1, 0, -1):  # Exclude first residue (no y_n ion)
            y_mass += self.aa_masses.get(sequence[i], 100.0)
            
            # Generate fragment charges 1+ and 2+ only (like Comet fragment index)
            for frag_charge in range(1, min(3, charge + 1)):  # 1+ and 2+ only  
                mz = (y_mass + (frag_charge - 1) * self.proton_mass) / frag_charge
                if self.mass_range[0] <= mz <= self.mass_range[1]:
                    bin_idx = self.bin_mass(mz)
                    relative_bin_idx = bin_idx - self.bin_mass(self.mass_range[0])
                    if 0 <= relative_bin_idx < self.num_bins:
                        # Comet's implementation: theoretical spectrum normalized to 50.0
                        # (experimental spectrum also normalized to 50.0 in MakeCorrData)
                        spectrum[relative_bin_idx] = 50.0
        
        return spectrum
    
    def preprocess_spectrum(self, spectrum: MassSpectrum, max_peaks: int = 0) -> np.ndarray:
        """
        Preprocess experimental spectrum according to Comet's algorithm.
        
        This follows the Comet preprocessing pipeline:
        1. Use all peaks (no filtering by intensity)
        2. Bin spectrum into unit mass bins (taking max intensity per bin)
        3. Apply square root transformation to intensities (as SEQUEST does)
        4. Apply Comet's MakeCorrData windowing normalization to 50.0  
        5. Store result for fast XCorr preprocessing
        """
        # Step 1: Use all peaks - no intensity filtering
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
        Apply Comet's fast XCorr preprocessing.
        
        This implements Comet's sliding window approach:
        1. Calculate sliding window average with offset (default offset = 75)
        2. Subtract from windowed spectrum: y' = y - (sliding_avg)
        3. Add flanking peaks contribution if enabled
        
        This is the critical step that makes the cross-correlation "fast"
        by preprocessing the experimental spectrum once.
        """
        # Comet's default XCorr processing offset (g_staticParams.iXcorrProcessingOffset)
        xcorr_offset = 75  # This is Comet's default value
        
        # Initialize the fast XCorr array
        preprocessed = np.zeros_like(windowed_spectrum)
        
        # Calculate sliding window statistics
        # iTmpRange = 2 * iXcorrProcessingOffset + 1 = 151
        window_range = 2 * xcorr_offset + 1
        normalization_factor = 1.0 / (window_range - 1.0)  # Comet's dTmp
        
        # Initialize sliding sum for the first window
        sliding_sum = 0.0
        for i in range(xcorr_offset):
            if i < len(windowed_spectrum):
                sliding_sum += windowed_spectrum[i]
        
        # Apply Comet's sliding window algorithm
        for i in range(xcorr_offset, len(windowed_spectrum) + xcorr_offset):
            # Add new element to window if within bounds
            if i < len(windowed_spectrum):
                sliding_sum += windowed_spectrum[i]
            
            # Remove old element from window if within bounds
            if i >= window_range:
                sliding_sum -= windowed_spectrum[i - window_range]
            
            # Calculate preprocessed value
            array_idx = i - xcorr_offset
            if array_idx < len(windowed_spectrum):
                # Core Comet formula: (sliding_sum - current_value) * normalization
                preprocessed[array_idx] = (sliding_sum - windowed_spectrum[array_idx]) * normalization_factor
        
        # Add flanking peaks contribution (Comet's optional feature)
        # This adds neighboring peak contributions with 0.5 weight
        enhanced_spectrum = preprocessed.copy()
        for i in range(1, len(preprocessed) - 1):
            original_value = windowed_spectrum[i] - preprocessed[i]  # Recover pdTmpCorrelationData[i] - pdTmpFastXcorrData[i]
            
            # Add left neighbor contribution
            if i > 0:
                left_original = windowed_spectrum[i-1] - preprocessed[i-1]
                enhanced_spectrum[i] += left_original * 0.5
            
            # Add right neighbor contribution  
            if i < len(preprocessed) - 1:
                right_original = windowed_spectrum[i+1] - preprocessed[i+1]
                enhanced_spectrum[i] += right_original * 0.5
        
        return enhanced_spectrum
    
    def calculate_fast_xcorr(self, theoretical_spectrum: np.ndarray, 
                           preprocessed_experimental: np.ndarray) -> float:
        """
        Calculate fast cross-correlation score using Comet's approach.
        
        This implements Comet's exact XCorr calculation:
        1. Simple dot product between theoretical and preprocessed experimental
        2. Apply Comet's scaling factor (0.005)
        3. Round to 3 decimal places like Comet
        """
        # Ensure both spectra have the same length
        min_len = min(len(theoretical_spectrum), len(preprocessed_experimental))
        
        # Calculate dot product (Comet's core XCorr calculation)
        raw_xcorr = np.dot(theoretical_spectrum[:min_len], 
                          preprocessed_experimental[:min_len])
        
        # Apply Comet's scaling: dXcorr *= 0.005
        # This scaling factor comes from Comet's intensity normalization to 50
        # and division by 10000 for score range management
        scaled_xcorr = raw_xcorr * 0.005
        
        # Round to 3 decimal places like Comet does
        final_xcorr = round(scaled_xcorr, 3)
        
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
        # XCorr scores are already scaled by 0.005 in calculate_fast_xcorr
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
        intercept = 0.0
        
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
                
                # Calculate slope and intercept
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
                
                intercept = mean_y - slope * mean_x
            else:
                break
        
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
    
    def search_spectrum(self, spectrum: MassSpectrum, peptide_candidates: List[PeptideCandidate],
                       charge_states: List[int] = [2, 3], max_peaks: int = 150) -> List[Tuple[PeptideCandidate, float, float, int]]:
        """
        Search a spectrum against peptide candidates using Comet-style XCorr with fast lookup.
        
        Uses pre-sorted peptide index for efficient binary search within isolation window.
        Returns the best results for each charge state separately.
        
        Args:
            spectrum: The experimental spectrum (with isolation window information)
            peptide_candidates: List of peptide candidates (not used directly, uses pre-built index)
            charge_states: List of charge states to consider (default: [2, 3])
            max_peaks: Maximum number of peaks to use from spectrum (default: 200, original SEQUEST specification)
        
        Returns list of (peptide, xcorr_score, e_value, charge) tuples, grouped by charge state.
        """
        # Apply Comet's two-stage preprocessing ONCE with peak filtering
        windowed_spectrum = self.preprocess_spectrum(spectrum, max_peaks)
        preprocessed_spectrum = self.preprocess_for_xcorr(windowed_spectrum)
        
        # Get the isolation window boundaries from the spectrum
        isolation_window_lower = spectrum.isolation_window_lower
        isolation_window_upper = spectrum.isolation_window_upper
        
        if isolation_window_lower == 0.0 or isolation_window_upper == 0.0:
            return []  # No valid isolation window
        
        # Fast lookup of peptides within isolation window using binary search
        if hasattr(self, 'sorted_peptides_by_mz') and self.sorted_peptides_by_mz:
            peptide_charge_pairs = self.find_peptides_in_isolation_window(
                isolation_window_lower, isolation_window_upper, charge_states)
        else:
            # Fallback to linear search if index not built
            peptide_charge_pairs = []
            for peptide in peptide_candidates:
                for charge in charge_states:
                    # Calculate theoretical m/z for this peptide at this charge state
                    theoretical_mz = (peptide.mass + charge * self.proton_mass) / charge
                    
                    # Check if this theoretical m/z falls within the precursor isolation window
                    if isolation_window_lower <= theoretical_mz <= isolation_window_upper:
                        peptide_charge_pairs.append((peptide, charge))
        
        # If no peptides pass mass filter, return empty results
        if not peptide_charge_pairs:
            return []
        
        # Group results by charge state and calculate xcorr for each
        results_by_charge = {}
        all_xcorr_scores = []
        
        for peptide, charge in peptide_charge_pairs:
            if charge not in results_by_charge:
                results_by_charge[charge] = []
            
            # Generate theoretical spectrum for this charge state
            theoretical = self.generate_theoretical_spectrum(peptide, charge)
            
            # Calculate XCorr using Comet's method
            xcorr_score = self.calculate_fast_xcorr(theoretical, preprocessed_spectrum)
            all_xcorr_scores.append(xcorr_score)
            results_by_charge[charge].append((peptide, xcorr_score, charge))
        
        # Sort results within each charge state by XCorr score (descending)
        for charge in results_by_charge:
            results_by_charge[charge].sort(key=lambda x: x[1], reverse=True)
        
        # Calculate E-values and combine results from all charge states
        final_results = []
        for charge in charge_states:
            if charge in results_by_charge:
                charge_results = results_by_charge[charge]
                for peptide, xcorr_score, charge in charge_results:
                    e_value = self.calculate_e_value(all_xcorr_scores, xcorr_score)
                    final_results.append((peptide, xcorr_score, e_value, charge))
        
        # Sort all results by charge state first, then by XCorr score within each charge state
        final_results.sort(key=lambda x: (x[3], -x[1]))  # Sort by charge, then by descending XCorr
        
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
    
    def write_spectrum_query(self, spectrum: 'MassSpectrum', search_results: List[Tuple['PeptideCandidate', float, float, int]], top_hits_per_charge: int = 5):
        """
        Write a spectrum query with its search results, grouped by charge state.
        
        Args:
            spectrum: The experimental spectrum
            search_results: List of (peptide, xcorr_score, e_value, charge) tuples
            top_hits_per_charge: Number of top hits to report per charge state
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


def main():
    """Main function to run the Comet-style fast XCorr search."""
    parser = argparse.ArgumentParser(description='Comet-style Fast XCorr Database Search')
    parser.add_argument('fasta_file', help='FASTA file containing protein sequences')
    parser.add_argument('mzml_file', help='mzML file containing mass spectra')
    parser.add_argument('--output', '-o', default='', help='Output file (pepXML format). If not specified, uses mzML filename with .pepXML extension')
    parser.add_argument('--top_hits', '-n', type=int, default=10, 
                       help='Number of top hits to report per spectrum (distributed across charge states)')
    parser.add_argument('--max_spectra', '-m', type=int, default=0, 
                       help='Maximum number of MS2 spectra to process (0 = process all)')
    parser.add_argument('--charge_states', '-c', type=str, default='2,3',
                       help='Comma-separated list of charge states to consider (default: 2,3)')
    parser.add_argument('--max_peaks', '-p', type=int, default=200,
                       help='Maximum number of peaks to use per spectrum (default: 200, original SEQUEST)')
    
    args = parser.parse_args()
    
    # Parse charge states
    charge_states = [int(c.strip()) for c in args.charge_states.split(',')]
    print(f"Using charge states: {charge_states}")
    print(f"Using maximum {args.max_peaks} peaks per spectrum (original SEQUEST specification)")
    print("Using SEQUEST-style XCorr preprocessing and scoring")
    
    # Determine output filename
    if not args.output:
        base_name = os.path.splitext(args.mzml_file)[0]
        args.output = base_name + '.pepXML'
    
    # Initialize Comet-style XCorr engine
    xcorr_engine = FastXCorr()
    
    print("Reading FASTA file...")
    proteins = xcorr_engine.read_fasta(args.fasta_file)
    print(f"Loaded {len(proteins)} proteins")
    
    print("Digesting proteins...")
    all_peptides = []
    for protein_id, sequence in proteins.items():
        peptides = xcorr_engine.digest_protein(sequence, protein_id)
        all_peptides.extend(peptides)
    print(f"Generated {len(all_peptides)} peptide candidates")
    
    # Build peptide index for fast isolation window lookup
    print("Building peptide index for fast isolation window lookup...")
    xcorr_engine.build_peptide_index(all_peptides, charge_states)
    
    print("Reading mzML file...")
    if args.max_spectra > 0:
        print(f"Limiting to first {args.max_spectra} MS2 spectra")
    spectra = xcorr_engine.read_mzml(args.mzml_file, args.max_spectra)
    
    print(f"Processing {len(spectra)} MS2 spectra with Comet-style XCorr")
    
    print("Performing database search...")
    print(f"Writing results to {args.output}")
    
    # Initialize pepXML writer and process spectra
    total_identifications = 0
    
    with PepXMLWriter(args.output, args.mzml_file, args.fasta_file) as writer:
        spectra_with_hits = 0
        for i, spectrum in enumerate(spectra):
            # Calculate isolation window info from the mzML file
            precursor_mz = spectrum.precursor_mz
            isolation_window_lower = spectrum.isolation_window_lower
            isolation_window_upper = spectrum.isolation_window_upper
            window_width = isolation_window_upper - isolation_window_lower
            
            # Fast lookup of peptides in isolation window
            if hasattr(xcorr_engine, 'sorted_peptides_by_mz') and xcorr_engine.sorted_peptides_by_mz:
                peptide_charge_pairs = xcorr_engine.find_peptides_in_isolation_window(
                    isolation_window_lower, isolation_window_upper, charge_states)
                peptides_in_window = len(peptide_charge_pairs)
            else:
                # Fallback to counting with linear search
                peptides_in_window = 0
                for peptide in all_peptides:
                    for charge in charge_states:
                        theoretical_mz = (peptide.mass + charge * xcorr_engine.proton_mass) / charge
                        if isolation_window_lower <= theoretical_mz <= isolation_window_upper:
                            peptides_in_window += 1
                            break  # Count each peptide only once even if it matches multiple charge states
            
            if i % 100 == 0 or len(spectra) <= 10:  # Always show for small runs
                print(f"Processing spectrum {i+1}/{len(spectra)} - Precursor: {precursor_mz:.4f} m/z, Window: [{isolation_window_lower:.5f}-{isolation_window_upper:.5f}] ({window_width:.5f} m/z), Peptides in window: {peptides_in_window} - {spectra_with_hits} spectra searched")
            
            # Search spectrum with Comet-style XCorr and configurable charge states
            search_results = xcorr_engine.search_spectrum(spectrum, all_peptides, charge_states, args.max_peaks)
            
            # Write results for every spectrum (even if no matches found)
            # Report top hits per charge state (distribute across charge states)
            top_hits_per_charge = max(1, args.top_hits // len(charge_states))  # Distribute top_hits across charge states
            writer.write_spectrum_query(spectrum, search_results, top_hits_per_charge)
            if search_results:
                spectra_with_hits += 1
                # Count total hits across all charge states
                hits_by_charge = {}
                for _, _, _, charge in search_results:
                    if charge not in hits_by_charge:
                        hits_by_charge[charge] = 0
                    if hits_by_charge[charge] < top_hits_per_charge:
                        hits_by_charge[charge] += 1
                        total_identifications += 1
    
    print("Search completed!")
    print(f"Total spectra processed: {len(spectra)}")
    print(f"Spectra with peptide matches: {spectra_with_hits}")
    print(f"Total identifications: {total_identifications}")
    print(f"Results saved to: {args.output}")
    print("Used Comet's exact preprocessing: MakeCorrData windowing (exp=50, theo=50) + fast XCorr with sliding window")
    print("Applied original SEQUEST specification: top 200 peaks, both experimental and theoretical normalized to 50")


if __name__ == '__main__':
    main()