import json
from pathlib import Path

from keras import optimizers, layers, regularizers

import metrics
from custom_encoding import CustomEncoder

reu2018 = dict()

# Misc
reu2018['seed'] = 42

# Directory locations and names
reu2018['model_file_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/3DUnetCNN_master/brats')
# reu2018['sliced_mri_data_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/middleslices_label41/images')
# reu2018['sliced_mri_mask_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/middleslices_label41/masks')
# reu2018['sliced_mri_data_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU UNet 2D/original preprocessed slices')
# reu2018['sliced_mri_mask_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU UNet 2D/original preprocessed masks')
# reu2018['sliced_mri_data_dir'] = Path(r'testing_data_dir/Users/xiaoxiaozhou/Desktop/Research/REU UNet 2D/patients_images_recallhist')
# reu2018['sliced_mri_mask_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU UNet 2D/patients_masks_recallhist')
# reu2018['predicted_mri_data_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/predictionresults/images')
# reu2018['predicted_mri_mask_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/predictionresults/masks')
# reu2018['quick_prediction_preview_dir'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/datasetfortraining/savedir')
# /home/xjin/xiaoxiao/test/Desktop/Research/unet2dforgithub/test2/previews
# reu2018['testing_data_dir'] = Path('/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/quickprediction')
# /home/xjin/xiaoxiao/test/Desktop/Research/unet2dforgithub/test2/datasetfortraining/quickprediction/Brats18_2013_0_1
# reu2018['predictions_dir'] = Path('/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/quickprediction')
# reu2018['predictedresult_dir'] = /Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/predictionresults')
# reu2018['output_data_dir'] = reu2018['base_dir'] / 'out_data'
# reu2018['output_mask_dir'] = reu2018['base_dir'] / 'out_mask'

# Output file names
reu2018['epoch_metrics_csv'] = Path(r'/Users/xiaoxiaozhou/Desktop/Research/3DUnetCNN_master/brats/reu2018_training.csv')
reu2018['model_file_name'] = 'reu2018_model.h5'

# Input file naming scheme
reu2018['data_filename_contains'] = 'tice_hist'
reu2018['mask_filename_contains'] = 'truth'
# reu2018['all_modalities'] = ["t1", "t1ce", "flair", "t2"]
reu2018['mri_file_extensions'] = ('*.nii.gz')  # Tuple of file formats to look for when getting MRI data.
# reu2018['zca_matrix'] = reu2018['base_dir'] / 'zca_matrix.npy'      #???

# Input size and shape
reu2018['input_size'] = (64, 64, 64)
reu2018['input_channels'] = 1

reu2018['slice_orientation'] = 'axial'  # Orientation that the training will occur on.  [d: axial]
reu2018['slice_mode'] = 'tumor_only'  # 'tumor_only' or a tuple containing min and max slice numbers.  [d: tumor_only]

# Augmentations
reu2018['equalize_histogram'] = True
reu2018['contrast_enhance'] = True
reu2018['save_augments'] = False  # Set to True to save previews of the augmented images.  [d: False]
reu2018['augment_mode'] = 'feature'  # 'feature' or 'sample' to determine how norm/std are applied.  [d: feature]
reu2018['sample_size'] = 75  # The size of each sample used for featurewise calculations.  [d: 16]
reu2018['standardize_inputs'] = True  # Set to True to have voxel values be standard deviations.  [d: True]
reu2018['normalize_inputs'] = True  # Set to True to transform MRI slices to have a mean of 0.  [d: True]
reu2018['brightness_range'] = None # Random brightness adjustment on images.  [d: None]
reu2018['whitening'] = False  # Set to True to enable ZCA Whitening.  [d: False]
reu2018['flip'] = True  # Set to True to enable horizontal and vertical flipping of images.  [d: True]
reu2018['rotate'] = 120  # Rotation amount in degrees.  Set to None or 0 to disable random rotations.  [d: 90]
reu2018['shear'] = 0.4  # Shear amount.  Set to None or 0 to disable random shears.  [d: 0.2]
reu2018['shift'] = 0.1  # Vert. and horiz. shift from center.  Set to None or 0 to disable random shifts.  [d: 0.1]
reu2018['zoom'] = 0.3  # Zoom amount.  Set to None or 0 to disable random zooms.  [d: 0.1]

# Training and validation settings
reu2018['validation_split'] = 0.2  # Fraction of MRI slices that are allocated for validation only.  [d: 0.2]
reu2018['epochs'] = 1  # The number of epochs to perform training.  [d: 150]
reu2018['batch_size'] = 75  # Size of the batch of MRI slices the model trains on for each step.  [d: 100]
reu2018['training_steps'] = None  # Set to None to automatically calculate training steps.  [d: None]
reu2018['validation_steps'] = None  # Set to None to automatically calculate validation steps.  [d: None]

# Prediction settings
reu2018['clip_predictions'] = True
reu2018['prediction_alpha'] = 1.

# General model settings
reu2018['depth'] = 5
reu2018['segmentation_levels'] = 2
reu2018['labels'] = 1
reu2018['initial_filters'] = 32
reu2018['dropout_rate'] = 0.5
reu2018['pooling_size'] = (2, 2, 2)
reu2018['kernel_size'] = (3, 3, 3)
reu2018['padding_mode'] = 'same'

# Encoder block settings
reu2018['encoder_activation'] = layers.LeakyReLU
reu2018['encoder_kernel_initializer'] = 'he_uniform'
reu2018['encoder_kernel_size'] = reu2018['kernel_size']
reu2018['encoder_kernel_regularizer'] = None

# Bottom block settings
reu2018['bottom_activation'] = layers.ReLU
reu2018['bottom_kernel_initializer'] = 'he_normal'
reu2018['bottom_kernel_size'] = reu2018['kernel_size']
reu2018['bottom_kernel_regularizer'] = regularizers.l1_l2()

# Decoder block settings
reu2018['decoder_activation'] = layers.LeakyReLU
reu2018['decoder_kernel_initializer'] = 'he_uniform'
reu2018['decoder_kernel_size'] = reu2018['kernel_size']
reu2018['decoder_strides'] = (2, 2, 2)

# Final block settings
reu2018['final_activation'] = 'sigmoid'
reu2018['final_kernel_initializer'] = 'he_normal'
reu2018['final_kernel_size'] = (1, 1, 1)
reu2018['theta_cutoff'] = 0.9

# Model compilation options
reu2018['optimizer'] = optimizers.Adadelta()
reu2018['loss_function'] = metrics.dice_loss
reu2018['metrics'] = ['acc', metrics.dice_coef]

# clinical_slices = dict()
#
# clinical_slices['Brats18_CBICA_AQV_1_t1'] = 64 / 128
# clinical_slices['Brats18_CBICA_AQR_1_t1'] = 64 / 128
# clinical_slices['Brats18_CBICA_ARW_1_t1'] = 68 / 136
# clinical_slices['patient_04'] = 130 / 208
# clinical_slices['patient_06'] = 92 / 208
# clinical_slices['patient_07'] = 98 / 138
# clinical_slices['patient_08'] = 122 / 176
# clinical_slices['patient_09'] = 142 / 192
# clinical_slices['patient_10'] = 78 / 140
# clinical_slices['patient_11'] = 104 / 208
# clinical_slices['patient_12'] = 100 / 140
# clinical_slices['patient_13'] = 176 / 348
# clinical_slices['patient_14'] = 124 / 176
# clinical_slices['patient_15'] = 113 / 192
# clinical_slices['patient_16'] = 103 / 192
#


# TODO
if __name__ == "__main__":
    json_data = json.dumps([reu2018, clinical_slices],
                           cls=CustomEncoder)
    with open(str(reu2018['base_dir'] / 'config.json'), 'w') as config_file:
        config_file.write(json_data)