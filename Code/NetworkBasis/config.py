"""!
@file config.py
Sets the parameters for configuration
"""
import SimpleITK as sitk
import os
from enum import Enum

#Insert the paths to your folders here
path = ""
path_seg = ""
path_result =""

experiment_name = "Groupwise_5/"

logs_path = os.path.join(path_result, experiment_name)

#Insert the paths to your csv files here
csv=path+"../csv/Filenames.csv"
train_csv = logs_path + 'train_img.csv'
vald_csv = logs_path + 'vald_img.csv'
test_csv = logs_path + 'test_img.csv'

seg_csv=path+"../csv/seg.csv"
train_seg_csv = logs_path + 'train_seg.csv'
vald_seg_csv = logs_path + 'vald_seg.csv'
test_seg_csv = logs_path + 'test_seg.csv'


class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1
    PERCENTMEAN = 2
    PERCENTWINDOW = 3

normalizing_method = NORMALIZING.PERCENTWINDOW

number_of_vald = 10
num_train_files =-1

kfold=5

width=256
height=256
numb_slices=64

#bins for mutual information
nb_bins_glob=50

seg_available = True
training=True

orig_filepath=""

# Resampling
adapt_resolution = True
target_spacing = [2.0, 2.0, 2.0]
target_direction = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt8
data_background_value = -1000
label_background_value = 0
max_rotation = 0

# Preprocessing
norm_min_v = 0
norm_max_v = 500
norm_eps = 1e-5

norm_min_v_img = 0
norm_max_v_img = 500
norm_min_v_label = 0
norm_max_v_label = 1

intervall_max=1

print_details=False
