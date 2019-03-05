import numpy as np
import skimage.measure
import scipy.ndimage.morphology
from general_utils import misc_utils


def unet_weight_map_border(mask, sigma=5):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    labels = skimage.measure.label(mask, connectivity=2)
    # misc_utils.display_numpy_image(labels)

    no_labels = np.equal(labels, 0)
    label_ids = sorted(np.unique(labels))[1: ]
    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = scipy.ndimage.morphology.distance_transform_edt(labels != label_id)
        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        exp_denom = -2.0 * sigma * sigma
        exp_nom = d1 + d2
        exp_nom *= exp_nom
        w = np.exp(exp_nom / exp_denom) * no_labels
        w = w.astype(dtype=np.float32)
    else:
        w = np.zeros_like(mask, dtype=np.float32)

    return w