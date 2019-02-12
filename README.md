# SEATLE
The code is for "Self-Attentive Few-Shot Learning for New Customer Recommendation in Location-based Social Networks".

## Environments 
  Python: 2.7
  
  Tensor-Flow: 1.11.0
  
  Numpy: 1.14.3
## Dataset
A sample dataset is provided in the data folder.
For training, validation, and test set, we split each of them into two sets: one positive set and one negative set.
The positive set contains obeserved user check-ins, while the negative set is composed of fake check-ins by negative sampling.
For each training, validation, and test file, the first column gives the business ID and the second column gives the user ID. Column three till column six give the geographical convinience features. The last column gives the label indicating whether it is oberved check-in or fake check-in. city_businessGeoDistance.txt gives the pairwise distances between businesses.
## Example to run this code
Train and evaluate the model:

`python seatle_learning.py`
