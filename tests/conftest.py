"""
Pytest configuration and fixtures for pyXcorrDIA tests.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for importing pyXcorrDIA
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyXcorrDIA import FastXCorr, MassSpectrum, PeptideCandidate


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def yqshtk_fasta(test_data_dir):
    """Path to YQSHTK FASTA test file."""
    fasta_path = test_data_dir / "YQSHTK.fasta"
    if not fasta_path.exists():
        pytest.skip(f"Test data file not found: {fasta_path}")
    return str(fasta_path)


@pytest.fixture(scope="session")
def yqshtk_mzml(test_data_dir):
    """Path to YQSHTK mzML test file."""
    mzml_path = test_data_dir / "YQSHTK.mzML"
    if not mzml_path.exists():
        pytest.skip(f"Test data file not found: {mzml_path}")
    return str(mzml_path)


@pytest.fixture(scope="session")
def ot_centroid_mgf(test_data_dir):
    """Path to OT centroid MGF test file."""
    mgf_path = test_data_dir / "ot_centroid_8340.mgf"
    if not mgf_path.exists():
        pytest.skip(f"Test data file not found: {mgf_path}")
    return str(mgf_path)


@pytest.fixture(scope="session")
def dia_library_1000(test_data_dir):
    """Path to DIA test library (1000 random precursors from report-lib.parquet)."""
    lib_path = test_data_dir / "test_library_1000.parquet"
    if not lib_path.exists():
        pytest.skip(f"DIA test library not found: {lib_path}")
    return str(lib_path)


@pytest.fixture(scope="session")
def dia_fasta_1000(test_data_dir):
    """Path to DIA test FASTA (proteins for 1000 library precursors)."""
    fasta_path = test_data_dir / "test_proteins_1000.fasta"
    if not fasta_path.exists():
        pytest.skip(f"DIA test FASTA not found: {fasta_path}")
    return str(fasta_path)


@pytest.fixture(scope="session")
def dia_mzml_small(test_data_dir):
    """Path to small DIA mzML file (60-70 kDa m/z window)."""
    mzml_path = test_data_dir / "test_dia_60000-70000.mzML"
    if not mzml_path.exists():
        pytest.skip(f"DIA test mzML not found: {mzml_path}")
    return str(mzml_path)


@pytest.fixture(scope="session")
def dia_mzml_full(test_data_dir):
    """Path to full DIA mzML file (15-minute gradient, complete RT coverage)."""
    mzml_path = test_data_dir / "test_dia_full.mzML"
    if not mzml_path.exists():
        pytest.skip(f"DIA test mzML not found: {mzml_path}")
    return str(mzml_path)


@pytest.fixture(scope="session")
def large_fasta(test_data_dir):
    """Path to human proteome FASTA database file."""
    fasta_path = test_data_dir / "uniprot_human_jan2025_yeastENO1_contam_ADpeps.fasta"
    if not fasta_path.exists():
        pytest.skip(f"Test data file not found: {fasta_path}")
    return str(fasta_path)


@pytest.fixture
def xcorr_engine():
    """Create a FastXCorr instance with default settings."""
    return FastXCorr()


@pytest.fixture
def xcorr_engine_with_mods():
    """Create a FastXCorr instance with carbamidomethyl-C modification."""
    return FastXCorr(static_modifications={'C': 57.021464})


@pytest.fixture
def xcorr_engine_no_mods():
    """Create a FastXCorr instance with no modifications."""
    return FastXCorr(static_modifications={})


@pytest.fixture
def simple_spectrum():
    """Create a simple test spectrum."""
    mz_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    intensity_array = np.array([1000.0, 2000.0, 3000.0, 2000.0, 1000.0])
    return MassSpectrum(
        mz_array=mz_array,
        intensity_array=intensity_array,
        scan_id="test_001",
        precursor_mz=600.0,
        charge=2,
        isolation_window_lower=599.0,
        isolation_window_upper=601.0
    )


@pytest.fixture
def sample_peptide(xcorr_engine):
    """Create a simple test peptide with correctly calculated mass."""
    sequence = "YQSHTK"
    mass = xcorr_engine.calculate_peptide_mass(sequence)
    return PeptideCandidate(
        sequence=sequence,
        protein_id="test_protein",
        mass=mass
    )
