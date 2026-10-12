# v1 post-mortem

v1 (github.com/lavanyaagarwal2006/Project_Arthritis_v1) reported 93.75% accuracy. That
number is not valid, for the reasons below. The list is kept here on purpose: reporting the
failure is part of the work.

1. **The data was about 4 radiographs, not 185.** Roboflow export names keep the original
   upload name (`image2_png_jpg.rf.<hash>.jpg`). Of the 182 filenames visible in the v1
   notebooks, all come from 4 source names (image1 ×61, image2 ×68, image3 ×50, image4 ×3).
   The test set contained augmented copies of the same images used in training.
2. **The labels were mostly anatomy, not disease.** The labels were ulna, radius, phalanges
   and similar structures, which are present in nearly every image.
3. **The metric was exact-match multi-label accuracy.** A constant prediction matches the
   majority baseline (15/16).
4. **The only varying label was dropped.** Soft tissue calcification had recall 0.0, and the
   "balanced" model reached 93.75% by deleting that column.
5. **The threshold was set from training predictions**, not validation data.
6. **Validation accuracy logged 0.0 every epoch**, which nobody caught.
7. **Horizontal flips were used as augmentation**, which turns a left hand into a right hand.

What v2 changes: a real labeled dataset with patient IDs, a pre-registered plan,
patient-grouped splits, AUROC with patient-clustered CIs, baselines in every table, and
thresholds chosen on validation folds only.
