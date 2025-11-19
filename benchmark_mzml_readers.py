#!/usr/bin/env python3
"""
Benchmark pyteomics mzML reader performance.
Tests both MS2-only and combined MS1+MS2 reading performance.
"""

import time
from pyXcorrDIA import FastXCorr


def benchmark_combined(mzml_file):
    """Benchmark pyteomics combined MS1+MS2 reader."""
    print("\n=== Testing pyteomics (combined MS1+MS2) ===")
    
    xcorr = FastXCorr()
    start = time.time()
    
    ms2_spectra, ms1_spectra = xcorr.read_mzml_combined(mzml_file, max_spectra=0)
    
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"  MS2 spectra: {len(ms2_spectra)}")
    print(f"  MS1 spectra: {len(ms1_spectra)}")
    
    # Calculate total data size
    total_peaks_ms2 = sum(len(s.mz_array) for s in ms2_spectra)
    total_peaks_ms1 = sum(len(s.mz_array) for s in ms1_spectra)
    print(f"  Total MS2 peaks: {total_peaks_ms2:,}")
    print(f"  Total MS1 peaks: {total_peaks_ms1:,}")
    print(f"  Throughput: {(len(ms2_spectra) + len(ms1_spectra)) / elapsed:.1f} spectra/sec")
    
    return elapsed, len(ms2_spectra), len(ms1_spectra)


def benchmark_ms2only(mzml_file):
    """Benchmark pyteomics MS2-only reader."""
    print("\n=== Testing pyteomics (MS2 only) ===")
    
    xcorr = FastXCorr()
    start = time.time()
    
    ms2_spectra = xcorr.read_mzml(mzml_file, max_spectra=0)
    
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"  MS2 spectra: {len(ms2_spectra)}")
    
    # Calculate total data size
    total_peaks_ms2 = sum(len(s.mz_array) for s in ms2_spectra)
    print(f"  Total MS2 peaks: {total_peaks_ms2:,}")
    print(f"  Throughput: {len(ms2_spectra) / elapsed:.1f} spectra/sec")
    
    return elapsed, len(ms2_spectra)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark_mzml_readers.py <mzml_file>")
        print("Example: python benchmark_mzml_readers.py data/Ast-Neo-15min-2mz-4ms-200agc-10.mzML")
        sys.exit(1)
    
    mzml_file = sys.argv[1]
    print(f"Benchmarking pyteomics mzML reader on: {mzml_file}")
    print("=" * 60)
    
    # Test combined reader
    print("\n" + "=" * 60)
    print("COMBINED MS1+MS2 READER")
    print("=" * 60)
    
    combined_time, ms2_count, ms1_count = benchmark_combined(mzml_file)
    
    # Test MS2-only reader
    print("\n" + "=" * 60)
    print("MS2-ONLY READER")
    print("=" * 60)
    
    ms2_time, ms2_only_count = benchmark_ms2only(mzml_file)
    
    # Verify consistency
    print("\n--- Verification ---")
    if ms2_count == ms2_only_count:
        print(f"✓ MS2 count matches: {ms2_count}")
    else:
        print(f"✗ MS2 count mismatch: combined={ms2_count}, ms2only={ms2_only_count}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nCombined MS1+MS2: {combined_time:.2f}s ({combined_time/60:.2f} min)")
    print(f"MS2-only:         {ms2_time:.2f}s ({ms2_time/60:.2f} min)")
    print(f"Overhead for MS1: {combined_time - ms2_time:.2f}s")
    
    print("\n" + "=" * 60)
