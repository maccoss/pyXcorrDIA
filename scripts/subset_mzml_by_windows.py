#!/usr/bin/env python3
"""
Subset an mzML file by isolation windows for testing purposes.

This script reads an mzML file and extracts spectra from a specified number
of isolation windows, creating a smaller test file suitable for git commits.

Usage:
    python subset_mzml_by_windows.py input.mzML output.mzML --num_windows 10
    python subset_mzml_by_windows.py input.mzML output.mzML --window_range 500.0-600.0
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to import pyXcorrDIA
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pyteomics import mzml
    import pymzml
    import shutil
except ImportError as e:
    print(f"ERROR: Required library not installed: {e}")
    print("Install with: pip install pyteomics pymzml")
    sys.exit(1)


def get_isolation_window(spectrum):
    """Extract isolation window bounds from spectrum."""
    precursor_list = spectrum.get('precursorList', {})
    if not precursor_list:
        return None
    
    precursors = precursor_list.get('precursor', [])
    if not precursors:
        return None
    
    precursor = precursors[0]
    isolation_window = precursor.get('isolationWindow', {})
    
    if not isolation_window:
        return None
    
    # Get target m/z and offset
    target_mz = isolation_window.get('isolation window target m/z')
    lower_offset = isolation_window.get('isolation window lower offset')
    upper_offset = isolation_window.get('isolation window upper offset')
    
    if target_mz is None or lower_offset is None or upper_offset is None:
        return None
    
    lower = target_mz - lower_offset
    upper = target_mz + upper_offset
    
    return (round(lower, 4), round(upper, 4))


def subset_mzml_by_windows(input_file, output_file, num_windows=None, window_range=None, 
                           include_ms1=True, verbose=True):
    """
    Subset mzML file by selecting spectra from specified isolation windows.
    
    Args:
        input_file: Path to input mzML file
        output_file: Path to output mzML file
        num_windows: Number of isolation windows to keep (from start)
        window_range: Tuple of (min_mz, max_mz) to filter windows
        include_ms1: Whether to include MS1 spectra
        verbose: Print progress information
    """
    if verbose:
        print(f"Reading {input_file}...")
    
    # First pass: collect all isolation windows and their spectra
    window_spectra = defaultdict(list)
    ms1_spectra = []
    total_ms2 = 0
    
    with mzml.read(str(input_file)) as reader:
        for spectrum in reader:
            ms_level = spectrum.get('ms level')
            
            if ms_level == 1:
                if include_ms1:
                    ms1_spectra.append(spectrum)
            elif ms_level == 2:
                total_ms2 += 1
                window = get_isolation_window(spectrum)
                if window:
                    window_spectra[window].append(spectrum)
    
    if verbose:
        print(f"  Found {len(window_spectra)} unique isolation windows")
        print(f"  Total MS2 spectra: {total_ms2}")
        if include_ms1:
            print(f"  Total MS1 spectra: {len(ms1_spectra)}")
    
    # Determine which windows to keep
    sorted_windows = sorted(window_spectra.keys())
    
    if window_range:
        min_mz, max_mz = window_range
        selected_windows = [w for w in sorted_windows 
                          if w[0] >= min_mz and w[1] <= max_mz]
        if verbose:
            print(f"  Filtering windows in range {min_mz}-{max_mz} m/z")
    elif num_windows:
        selected_windows = sorted_windows[:num_windows]
        if verbose:
            print(f"  Selecting first {num_windows} windows")
    else:
        selected_windows = sorted_windows
    
    # Collect selected spectra
    selected_ms2 = []
    for window in selected_windows:
        selected_ms2.extend(window_spectra[window])
    
    if verbose:
        print(f"\nOutput summary:")
        print(f"  Selected windows: {len(selected_windows)}")
        print(f"  Window range: [{selected_windows[0][0]:.2f}-{selected_windows[0][1]:.2f}] to "
              f"[{selected_windows[-1][0]:.2f}-{selected_windows[-1][1]:.2f}] m/z")
        print(f"  MS2 spectra: {len(selected_ms2)}")
        if include_ms1:
            print(f"  MS1 spectra: {len(ms1_spectra)}")
        
        # Calculate file size estimate
        spectra_per_window = [len(window_spectra[w]) for w in selected_windows]
        avg_spectra = sum(spectra_per_window) / len(spectra_per_window) if spectra_per_window else 0
        print(f"  Average spectra per window: {avg_spectra:.1f}")
    
    # Write output file
    if verbose:
        print(f"Writing {output_file}...")
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the entire file first
    shutil.copy2(str(input_file), str(output_file))
    
    # Get the isolation windows we want to keep
    selected_window_set = set(selected_windows)
    
    # Now read the copied file and remove unwanted MS2 spectra
    from lxml import etree
    
    # Parse the mzML file
    tree = etree.parse(str(output_file))
    root = tree.getroot()
    
    # Define namespace
    ns = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # Find all spectrum elements
    spectrum_list = root.find('.//mzml:spectrumList', ns)
    if spectrum_list is None:
        # Try without namespace
        spectrum_list = root.find('.//spectrumList')
    
    if spectrum_list is not None:
        spectra_to_remove = []
        kept_count = 0
        removed_count = 0
        
        for spectrum in list(spectrum_list):
            # Get MS level
            ms_level = None
            for cv_param in spectrum.findall('.//{*}cvParam'):
                if cv_param.get('name') == 'ms level':
                    ms_level = int(cv_param.get('value', 0))
                    break
            
            # For MS2 spectra, check if they're in selected windows
            if ms_level == 2:
                # Get isolation window from spectrum
                window = None
                precursor_list = spectrum.find('.//{*}precursorList')
                if precursor_list is not None:
                    precursor = precursor_list.find('.//{*}precursor')
                    if precursor is not None:
                        isolation_window = precursor.find('.//{*}isolationWindow')
                        if isolation_window is not None:
                            target_mz = None
                            lower_offset = None
                            upper_offset = None
                            
                            for cv_param in isolation_window.findall('.//{*}cvParam'):
                                name = cv_param.get('name')
                                if name == 'isolation window target m/z':
                                    target_mz = float(cv_param.get('value'))
                                elif name == 'isolation window lower offset':
                                    lower_offset = float(cv_param.get('value'))
                                elif name == 'isolation window upper offset':
                                    upper_offset = float(cv_param.get('value'))
                            
                            if target_mz is not None and lower_offset is not None and upper_offset is not None:
                                lower = target_mz - lower_offset
                                upper = target_mz + upper_offset
                                window = (round(lower, 4), round(upper, 4))
                
                # Check if this window should be kept
                if window not in selected_window_set:
                    spectra_to_remove.append(spectrum)
                    removed_count += 1
                else:
                    kept_count += 1
            elif ms_level == 1 and not include_ms1:
                # Remove MS1 if requested
                spectra_to_remove.append(spectrum)
                removed_count += 1
            else:
                kept_count += 1
        
        # Remove unwanted spectra
        for spectrum in spectra_to_remove:
            spectrum_list.remove(spectrum)
        
        # Update spectrum count
        spectrum_list.set('count', str(kept_count))
        
        if verbose:
            print(f"  Removed {removed_count} spectra, kept {kept_count} spectra")
    
    # Write the modified file
    tree.write(str(output_file), encoding='utf-8', xml_declaration=True, pretty_print=True)
    
    if verbose:
        output_path = Path(output_file)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Output file size: {size_mb:.2f} MB")
        print("✓ Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Subset mzML file by isolation windows for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep first 10 isolation windows
  python subset_mzml_by_windows.py input.mzML output.mzML --num_windows 10
  
  # Keep windows in specific m/z range
  python subset_mzml_by_windows.py input.mzML output.mzML --window_range 500.0-600.0
  
  # Keep 5 windows, exclude MS1
  python subset_mzml_by_windows.py input.mzML output.mzML --num_windows 5 --no_ms1
  
  # Create small test file (3 windows)
  python subset_mzml_by_windows.py full_data.mzML test_data_3windows.mzML --num_windows 3
        """
    )
    
    parser.add_argument('input', help='Input mzML file')
    parser.add_argument('output', help='Output mzML file')
    parser.add_argument('--num_windows', type=int, 
                       help='Number of isolation windows to keep (from start)')
    parser.add_argument('--window_range', type=str,
                       help='m/z range for windows (e.g., 500.0-600.0)')
    parser.add_argument('--no_ms1', action='store_true',
                       help='Exclude MS1 spectra from output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Parse window range if provided
    window_range = None
    if args.window_range:
        try:
            min_mz, max_mz = map(float, args.window_range.split('-'))
            window_range = (min_mz, max_mz)
        except ValueError:
            print(f"ERROR: Invalid window range '{args.window_range}'. Use format: 500.0-600.0")
            sys.exit(1)
    
    # Validate arguments
    if not args.num_windows and not window_range:
        print("ERROR: Must specify either --num_windows or --window_range")
        sys.exit(1)
    
    if args.num_windows and window_range:
        print("ERROR: Cannot specify both --num_windows and --window_range")
        sys.exit(1)
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Subset the file
    try:
        subset_mzml_by_windows(
            args.input,
            args.output,
            num_windows=args.num_windows,
            window_range=window_range,
            include_ms1=not args.no_ms1,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
