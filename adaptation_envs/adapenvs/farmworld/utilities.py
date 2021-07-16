from typing import Dict, List, Tuple, Union

import numpy as np
import os
from pathlib import Path

RESOURCES_RELATIVE_PATH = "../resources/"

def get_resource_path(path: str):
    return Path(__file__) / RESOURCES_RELATIVE_PATH / path

def get_resource_path_string(path: str):
    return "/".join(str(Path(__file__)).split("/")[:-1]) + "/resources/" + path


def get_image_and_mask(rgba : np.ndarray):
    '''
    Converts an rgba of shape (width, height, 4) into a rgb (width, heigh, 3) and alpha mask
    '''
    return rgba[...,0:3], rgba[...,3]

def layer_images(images : List[np.ndarray]):
    '''
    Compose into one rgba image. Follows layering order provided by the ordering of the lists.

    images: a list of rgba images of the same size
    returns: a single rgba image
    '''
    base = np.zeros(images[0].shape, dtype=np.uint8)
    for img in images:
        # before adding any image, remove whatever the previous layer had where the current alpha layers are not 0
        alpha_layer_ones = img[...,3]
        base[...,0:3] = base[...,0:3] * ((255 - alpha_layer_ones) / 255).reshape(9,9,1)
        base[...,0:3] += img[...,0:3]
        base[...,3] += img[...,3]
    return base

def layer_two_images_fast(img1, img2):
    '''
    Significantly faster than layer_images. Sets RGBs of img2 over img1, and **does not** set alpha values.

    Mutates rgb values of img1
    '''
    img1[...,0:3] = img1[...,0:3] * (img2[...,3].view().reshape(9, 9, 1) == 0) + img2[...,0:3]
