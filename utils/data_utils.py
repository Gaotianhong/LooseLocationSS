import numpy as np
import nibabel as nib
from scipy import ndimage


def read_nifti_file(filepath):
    """Load data"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def window_transform(volume, windowWidth, windowCenter):
    """
    Trucated image according to window center and window width and normalized to [0, 1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    volume = (volume - minWindow) / float(windowWidth)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    volume = volume.astype("float32")

    return volume


def normalize(volume, windowWidth, windowCenter):
    """Data normalization"""
    min = float(windowCenter) - 0.5 * float(windowWidth)
    max = float(windowCenter) + 0.5 * float(windowWidth)  # window_transform
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")

    return volume


def resize_image(img, width, height):
    """Resize img across the z axis"""
    desired_width = width
    desired_height = height
    current_width = img.shape[0]
    current_height = img.shape[1]
    # adjust
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1.0
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    img = np.flip(img, axis=2)

    return img


def process_img(path, width, height):
    """Process image"""
    volume = read_nifti_file(path)
    WW, WL = 400, 60
    volume = normalize(volume, WW, WL)
    volume = resize_image(volume, width, height)
    return volume
