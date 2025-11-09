#!/usr/bin/env python3
"""
Quick test runner script to verify pytest infrastructure is working.
Run this script to quickly check if the test suite is properly set up.
"""
import sys
import subprocess
from pathlib import Path


def main():
    """Run basic tests to verify setup."""
    print("=" * 70)
    print("pyXcorrDIA Test Infrastructure Quick Check")
    print("=" * 70)
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent
    if not (project_root / "pyXcorrDIA.py").exists():
        print("❌ Error: pyXcorrDIA.py not found. Run this script from project root.")
        return 1
    
    # Check if tests directory exists
    if not (project_root / "tests").exists():
        print("❌ Error: tests/ directory not found.")
        return 1
    
    print("✓ Project structure looks good\n")
    
    # Try to import pytest
    print("Checking pytest installation...")
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__} is installed\n")
    except ImportError:
        print("❌ pytest is not installed. Install it with:")
        print("   pip install pytest pytest-cov")
        return 1
    
    # Try to import pyXcorrDIA
    print("Checking pyXcorrDIA module...")
    try:
        sys.path.insert(0, str(project_root))
        from pyXcorrDIA import FastXCorr, MassSpectrum, PeptideCandidate
        print("✓ pyXcorrDIA module can be imported\n")
    except ImportError as e:
        print(f"❌ Cannot import pyXcorrDIA: {e}")
        return 1
    
    # Check for test data files
    print("Checking test data files...")
    test_data_dir = project_root / "tests" / "data"
    test_files = {
        "YQSHTK.fasta": "Small FASTA test file",
        "YQSHTK.mzML": "mzML spectra file",
        "ot_centroid_8340.mgf": "MGF spectra file",
    }
    
    missing_files = []
    for filename, description in test_files.items():
        if (test_data_dir / filename).exists():
            print(f"  ✓ tests/data/{filename} - {description}")
        else:
            print(f"  ⚠ tests/data/{filename} - {description} (missing - some tests will skip)")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠ {len(missing_files)} test data files are missing.")
        print("  Tests requiring these files will be skipped automatically.")
    print()
    
    # Run a subset of quick tests
    print("=" * 70)
    print("Running Quick Test Suite (basic functionality only)")
    print("=" * 70)
    print()
    
    # Run only the basic functionality tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_basic_functionality.py", 
         "-v", 
         "--tb=short"],
        cwd=project_root
    )
    
    print()
    print("=" * 70)
    if result.returncode == 0:
        print("✓ Quick tests PASSED!")
        print()
        print("Next steps:")
        print("  - Run all tests: pytest")
        print("  - Run with coverage: pytest --cov=pyXcorrDIA --cov-report=html")
        print("  - Run specific test file: pytest tests/test_preprocessing.py")
        print("  - See tests/README.md for more options")
    else:
        print("❌ Some tests FAILED. Check the output above for details.")
    print("=" * 70)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
