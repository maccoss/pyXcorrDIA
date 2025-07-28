# pyXcorrDIA
Simple Xcorr search tool written in python. The intent was to test some new scoring algorithms for DIA data analysis but I haven't gotten there yet.  Currently all precursors within the precursor isolation window are scored against each spectrum.  I have implemented the Comet Xcorr and E-value scoring. I have also implemented target-decoy competition. I use a reversed decoy unless the reversed sequence is present in list of target peptide sequences and then I cycle the amino acids. Any peptides that don't have a suitable decoy won't be searched as they are unlikely to yield a good search regardless.

**Things to implement.**
- A RT scoring algorithm based on Carafe libraries. 
- A peptide-centric version of the Xcorr scoring.
- A smoothing algorithm for the Xcorr similar to reported in Venable and Yates Anal Chem 2004.
- Some XIC based scoring based on the Carafe predicted most intense fragments.
- XIC interference detection
