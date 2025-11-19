# Spectrum-Centric Preprocessing Update Summary

## Changes Made

### Code Changes (✅ COMPLETED)

1. **Removed `--dia_spectrum_centric_preprocessing` flag** - Now the default and only mode
2. **Removed `preprocess_theoretical_spectrum()` method** - No longer needed
3. **Simplified `calculate_peptide_centric_xcorr()`** - Always uses 0.005 scaling
4. **Removed `spectrum_centric_mode` parameter** - From all function signatures
5. **Simplified preprocessing logic** - Always uses spectrum-centric approach:
   - Theoretical spectra: Windowed only (no Fast XCorr preprocessing)
   - Experimental spectra: Fully preprocessed (windowed + Fast XCorr)
   - Scaling: 0.005 (standard spectrum-centric)

### Documentation Updates (✅ PARTIALLY COMPLETED)

#### Completed:
- ✅ `CLAUDE.md` - Updated XCorr scaling section
- ✅ `README.md` - Updated DIA vs Standard Mode table
- ✅ `TEST_INFRASTRUCTURE_SUMMARY.md` - Updated test descriptions (line 49, 208)
- ✅ `TESTING_QUICKSTART.md` - Updated scaling references (line 251, 266)

#### Remaining Updates Needed:

##### `DIA_MODE_IMPROVEMENTS.md`
- Search for references to "peptide-centric" preprocessing
- Update any mentions of "0.0001 scaling"
- Update preprocessing flow descriptions

##### `OPTIMIZATION_IMPROVEMENTS.md`
- Update preprocessing approach descriptions
- Verify no references to old peptide-centric mode

##### Test Files That Need Updates:
1. **`tests/test_peptide_centric_real_data.py`** (NEEDS UPDATES)
   - Line 39: Remove `EXPECTED_BEST_XCORR_0001` variable
   - Lines 135-180: Update `test_scaling_effects()` to only test 0.005 scaling
   - Lines 151-176: Update `test_peptide_centric_function_uses_correct_scaling()` to test 0.005
   - Line 341: Update `test_validate_scaling_factor_choice()` description and logic
   - Line 352: Remove references to "0.0001 scaling"

2. **`tests/test_unified_xcorr.py`** (NEEDS REVIEW)
   - Line 8: Update test description
   - Line 115: Update `test_peptide_centric_scaling()` to test 0.005
   - Line 162: Update `test_calculate_peptide_centric_xcorr_wrapper()` 

3. **`tests/test_peptide_centric.py`** (NEEDS REVIEW)
   - Update any references to "preprocessing theoretical" or "0.0001"
   - Verify asymmetry tests are still relevant

## Key Technical Changes

### Old Approach (Removed):
```python
# Peptide-centric: Preprocess theoretical, window experimental
theoretical_preprocessed = engine.preprocess_theoretical_spectrum(theoretical)
experimental_windowed = engine.preprocess_spectrum(spectrum)
xcorr = engine.calculate_xcorr(experimental_windowed, theoretical_preprocessed, 0.0001)
```

### New Approach (Current):
```python
# Spectrum-centric: Preprocess experimental, window theoretical
experimental_windowed = engine.preprocess_spectrum(spectrum)
experimental_preprocessed = engine.preprocess_for_xcorr(experimental_windowed)

# Window theoretical (no Fast XCorr)
theoretical = engine.generate_theoretical_spectrum(peptide, charge)
highest_ion_bin = 0
for i in range(len(theoretical) - 1, -1, -1):
    if theoretical[i] > 0:
        highest_ion_bin = i
        break
theoretical_windowed = engine._make_corr_data(theoretical, highest_ion_bin, 1.0)

# Score
xcorr = engine.calculate_xcorr(experimental_preprocessed, theoretical_windowed, 0.005)
```

## Benefits

1. **Performance**: More efficient when library size >> spectra per window
2. **Simplicity**: Single preprocessing approach, no mode switching
3. **Consistency**: Same 0.005 scaling factor throughout
4. **Code clarity**: Removed conditional logic and duplicate code paths

## Testing Status

- ✅ Code compiles without errors
- ✅ Command-line help shows correct arguments
- ⚠️  Test files need updates to reflect new scaling
- ⚠️  Integration tests should be run after test updates

## Next Steps

1. Update remaining test files (especially test_peptide_centric_real_data.py)
2. Run full test suite: `pytest tests/`
3. Update any remaining documentation references
4. Verify integration with real DIA data
