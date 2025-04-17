# Groupwise Multiresolution Deformable Medical Image Registration
This repository contains the implementation of a groupwise multiresolution network for deformable medical image registration.
The framework was implemented using Keras with Tensorflow backend.

If you use our code in your work please cite the following paper:
Strittmatter, A., Weis, M. & Zöllner, F.G. A groupwise multiresolution network for DCE-MRI image registration. Sci Rep 15, 9891 (2025). https://doi.org/10.1038/s41598-025-94275-9

# Manual
Usage:
1. In config.py:
    
   a) Provide the file paths to your data: images and segmentations (if available) (lines 10 and 11).
   
   b) Specify the path where the results should be saved (line 12) and the name of the output folder (line 14).
   
   c) Provide the paths to the CSV files containing the filenames of the images and segmentations (line 19 and 24).
   
   d) If segmentations are available, set seg_available = True (line 50).
   
4. Run the main program (main.py) with the following settings:
   
   a) To train a network from scratch: set is_training = True (line 362). After training, the weights for each fold of the five-fold cross-validation will be saved in subfolders within the output folder (numbered from 0 to 4).

   b) Specify the group size in line 375.
   
   c) For interference and evaluation, set is_apply = True and is_evaluate = True (line 363 and 364). The output folder will contain two subfolders: "predict" and "eval". The "predict" folder will contain the registered moved images and segmentations (if seg_available = True in config.py line 50). The "eval" folder will contain the MI, NMI and SSIM values and number of Jacobian determinants ≤ 0. Additionally, boxplots for the metrics and individual folds of the five-fold cross-validation will be stored in "eval/plots".
   
