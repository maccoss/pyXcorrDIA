## This is a plan to perform a multi-step search using pyXcorrDIA.
This entire plan is for the --dia_mode option using a --speclib library file.

### Step 1: Import mzML Data and preprocess for DIA analysis
- This is part of our current workflow and doesn't need to change.
- It is important that this is only down once as it is expensive.  So the data should be stored and preprocessed to be used in multiple steps.

### Step 2: m/z and RT calibration
This will be performed exactly how we do the analysis no for the QC plots but with some modifications:
- Select a subset of peptide precursors from the library to search through the entire mzML file. 
- Use a random selection of N precursors from the library for this step. Generate decoy library spectra for these N precursors Lets call this setting --cal_library_peptides N. Lets start with a default of 2000 target peptide precursors (no decoys from the library) and with a q-value in the library <= 0.01.
- Using the LibCosine scores for these peptides and their corresponding decoys we will filter them for just the ones that pass a 1% FDR threshold.
- For the MS1 and MS2 data we will use the QC functions that calculate the   delta m/z error (in ppm or Da).
   - adjusted_mz = precursor_mz + (precursor_mz * mean_ppm / 1e6)
   - window_halfwidth = 3 * (precursor_mz * sd_ppm / 1e6)
   - If the user specified ppm tolerance we will use the ppm values otherwise we will convert to Da using the average precursor m/z.
   - The MS1 and MS2 need separate tolerances as some instruments have different accuracies for MS1 and MS2 because of different analyzers.
- For RT calibration we will use the same set of peptides and perform the loess regression against the library RTs to determine the RT shift and window to use for the full search. So for any peptide in the library we can take the library RT and use the loess regression to determine the expected RT in the data. We can then use the standard deviation of the RT differences to set a window. For example, if the standard deviation is 1.5 minutes we can use a window of +/- 3 * 1.5 = +/- 4.5 minutes around the expected RT for each peptide.
  - If loess fit isn't ideal we can consider other regression methods but loess should work for most cases.
  - I like the idea of starting with a kernel density estimate, then trying loess, and if that fails falling back to linear regression. 
- The output of this step will be the m/z error statistics (mean and SD for MS1 and MS2) and the RT calibration model and RT window to use for the full search. I also want to output QC plots for the m/z error distributions and the RT correlation plot.
- We do not need to output any search results from this step as it is only for calibration. We also don't need to store the XCorr scores or any other intermediate data. Everything will be performed using the LibCosine scores only.

### Step 3: Full DIA Search
- Using the m/z error statistics and RT calibration from step 2 we will perform a full DIA search.
- The m/z windows for MS1 and MS2 will be set based on the mean+delta and 3*SD from step 2 as described above.
- The RT window will be set based on the loess regression and 3*SD from step 2 as described above. This means that for each peptide in the library we will calculate its expected RT in the data using the loess regression and then use that RT +/- the RT window for filtering. This will require not only grouping peptides into the isolation windows but also sorting by RT range.  This will significantly reduce the number of peptides to score for each spectrum. It is not possible to bin peptides into fixed RT bins as they are a continuous variable. So we will need to sort them so that when we go to score spectrum at RT X we can quickly find the subset of peptides that fall within the RT window around X.
- The output of this step will be the full DIA search results with XCorr scores, etc... As with the current implementation the additional scores on the MS2 spectra will only be calculated at the scanID where the peptide (target or decoy respectively) scored best by LibCosine. The MS1 PrecursorCosine will be calculated from the MS1 spectrum nearest in RT. This should be the current behavior.
- Additionally I want to output some additional scores as well.
  - delta_mz_ppm_precursor: The delta m/z in ppm for the precursor
  - delta_mz_ppm_fragments: The delta m/z in ppm for the fragments (average over all matched fragments).
  - delta_rt: The delta RT in minutes between the library RT and the calibrated RT. 
  - decoy_delta_mz_ppm_precursor: The delta m/z in ppm for the precursor M+0 isotope for the decoy peptide.
  - decoy_delta_mz_ppm_fragments: The delta m/z in ppm for the fragments (average over all matched fragments) for the decoy peptide.
  - decoy_delta_rt: The delta RT in minutes between the library RT and the calibrated RT for the decoy peptide. The decoy peptide RT is the same as the target peptide RT since we are using a target-decoy competition approach.

### Step 4: Mokapot for scoring and final FDR calculation.
- I want to use mokapot (https://github.com/wfondrie/mokapot) to perform the final scoring and FDR calculation. Documantation can be found here:  https://mokapot.readthedocs.io/en/latest/.
- Because we are using a peptide-centric search we won't have PSMs the PSMid column in the mokapot input file can just be the peptide sequence + charge state + target/decoy. 

### Further Considerations
- Calibration quality checks.  We should only calibrate the m/z and RT if we have at least 100 peptide precursors passing the 1% FDR threshold in step 2. If we have less than that we might want to increase the random sampled subset to a larger size to get more peptides. If we have less than 100 peptides even after increasing the subset size to say 5000 peptides then we should skip calibration and use the user provided MS1 and MS2 tolerances. For the RT we will need to fall back to searching the entire RT range for each peptide.  We should also output a warning message to the user indicating that calibration was skipped due to insufficient peptides.
- We should output a JSON file with the calibration parameters used for the full search. This will include the mean and SD for MS1 and MS2 m/z errors, the RT calibration model parameters.
