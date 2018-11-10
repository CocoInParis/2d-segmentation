import warnings
from math import sqrt
from random import randint

import keras
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import nibabel as nib
import os
from pathlib import Path
from typing import List

import nrrd
import nibabel as nib
import numpy as np
from scipy.misc import toimage
from scipy.ndimage import zoom
from skimage import exposure

from config2d import reu2018
# from patient import Patient

from keras.models import load_model
from skimage.io import imread, imsave
from metrics import dice_loss, dice_coef


# class QuickPrediction():
#     def __init__(self, patient, slice_id, save_dir, mode):
#         super().__init__()
#         self.patient = patient
#         self.slice = slice_id
#         self.save_dir = save_dir
#         self.batch_counter = -1
#         self.mode = mode


def predict(patient, slice_id, save_dir, mode):
    data, _ = patient.unpack()

    data = np.moveaxis(data, 2, 0)
    if mode == 'slice':
        data = data[slice_id, ...]
        data = (data - np.mean(data))/np.std(data)
        prediction = model.predict(data[None, ...], batch_size=64)[0, :, :, 0]
    else:
        prediction = model.predict(data, batch_size=64)[..., 0]

    data = data[..., 0]
    return prediction, data

def on_batch_end(self, batch, logs=None):
    prediction, data = self.predict()
    self.batch_counter += 1
    if self.mode == 'slice':
        self.plot_contour("", data, prediction, 'preview')
    else:
        self.plot_random_slices("", data, prediction, 'preview', 4)

def on_epoch_end(self, epoch, logs=None):
    prediction, data = self.predict()
    self.plot_colormap("Quick Preview {}".format(epoch), data,  prediction, 'save')

def plot_contour(save_dir, title, data, pred):
    pred[pred < 0.9] = 0

    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.contour(pred, colors='red', linewidths=1)

    plt.savefig(str(save_dir / "{}.png".format(title)), dpi='figure')
    plt.clf()
    plt.close('all')

    # if mode == 'preview':
    #     with open(str(save_dir / 'flag.txt')) as f:
    #         if f.read() == 'True':
    #             plt.axis('off')
    #             plt.show(block=True)
#
# def plot_random_slices(self, title, data, pred, mode, count):
#     side = int(sqrt(count))
#
#     fig, axes = plt.subplots(side, side)
#     fig.dpi = 150
#     fig.tight_layout()
#
#     pred[pred < 0.9] = 0
#
#     for i in range(side):
#         for j in range(side):
#             idx = randint(0, 127)
#             axes[i, j].imshow(data[idx, ...], cmap='gray')
#             if np.sum(pred[idx, ...]) > 0:
#                 axes[i, j].contour(pred[idx, ...], colors='red', linewidths=1)
#             axes[i, j].axis('off')
#
#     if mode == 'preview':
#         with open(str(self.save_dir / 'flag.txt')) as f:
#             if f.read() == 'True':
#                 plt.show(block=True)

def plot_colormap(title, data, pred, mode):
    fig, (da, pr) = plt.subplots(ncols=2, squeeze=True)
    fig.dpi = 150

    print("\nMin: {:.2f}\tMax: {:.2f}".format(np.min(pred), np.max(pred)))
    savedirmap = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/quickprediction/labelmap')
    da.imshow(data, cmap='gray')
    pr.imshow(pred, cmap='jet')

    fig.tight_layout()
    plt.axis('off')

    if mode == 'preview':
        with open(str(savedirmap / 'flag.txt')) as f:
            if f.read() == 'True':
                plt.show(block=True)
    elif mode == 'save':
        plt.savefig(str(savedirmap / "{}.png".format(title)), dpi='figure')
    else:
        raise ValueError("Quick Prediction mode must be either preview or save")
    plt.clf()
    plt.close('all')


class CheckpointOverhaul(keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath + '_{}.h5'.format(epoch), overwrite=True)
                else:
                    self.model.save(filepath + '_{}.h5'.format(epoch), overwrite=True)

def resample(in_array, target_size=None):
    in_shape = in_array.shape

    if target_size is None:
        target_size = (reu2018['input_size'])

    factor = [target_size[i] / in_shape[i] for i in range(len(in_shape))]
    out_array = zoom(in_array, factor, mode='constant', order=0)
    out_array = out_array[..., np.newaxis]

    return out_array

def get_test_patient(directory, name):
    path = os.path.join(directory, name)
    data, _ = read_file(path)

    data = np.moveaxis(data, 2, 0)
    new_data = np.zeros(data.shape)

    for i in range(data.shape[0]):
        if np.sum(data[i]) == 0:
            new_data[i] = data[i]
        else:
            newd = data[i]/np.max(data[i])
            new_data[i] = exposure.equalize_adapthist(newd)

    new_data = np.moveaxis(new_data, 0, 2)
    new_data = resample(new_data).astype(np.float32)

    mean = np.mean(new_data[..., 0])
    std = np.std(new_data[..., 0])

    new_data -= mean
    new_data /= std

    return Patient(name.split('.')[0], new_data)

def read_file(in_file):
    if '.nrrd' in str(in_file):
        return nrrd.read(in_file)[0], np.eye(4)
    elif '.nii.gz' in str(in_file):
        mri = nib.load(str(in_file))
        return mri.get_fdata(), mri.affine
    else:
        return None


# model = load_model('reu2018_model_149.h5',
#                    custom_objects={'dice_coefficient_loss': dice_loss, 'dice_coefficient': dice_coef})
# patientpath = Path(r'/Users/xiaoxiaozhou/Desktop/Research/datasetfortraining/quickprediction')
# patients = glob.glob(os.path.join(patientpath, "*"))
# savedir = Path(r'/Users/xiaoxiaozhou/Desktop/Research/REU_UNet_2D/datasetfortraining/quickprediction')
#
# batch_counter = -1
#
# for pat in patients:
#     base = os.path.basename(pat)
#     if base == 'Brats18_2013_0_1':
#         sliceid = int(64* 128 / 155)
#     elif base == 'Brats18_2013_1_1':
#         sliceid = int(61* 128 / 155)
#     elif base == 'Brats18_2013_2_1':
#         sliceid = int(98* 128 / 155)
#     elif base == 'Brats18_2013_3_1':
#         sliceid = int(65* 128 / 155)
#     elif base == 'Brats18_2013_4_1':
#         sliceid = int(66* 128 / 155)
#     elif base == 'Brats18_2013_5_1':
#         sliceid = int(87* 128 / 155)
#     elif base == 'Brats18_2013_6_1':
#         sliceid = int(77* 128 / 155)
#     elif base == 'Brats18_2013_7_1':
#         sliceid = int(108 * 128 / 155)
#     elif base == 'Brats18_2013_8_1':
#         sliceid = int(78* 128 / 155)
#     elif base == 'Brats18_2013_9_1':
#         sliceid = int(75* 128 / 155)
#
#
#     patient = get_test_patient(pat, "tice_hist.nii.gz")
#     print(sliceid)
#     pred, data = predict(patient, sliceid, savedir, mode='slice')
#     # plot_contour(savedir, base, data, pred)
#     plot_colormap(base, data, pred, mode='save')
