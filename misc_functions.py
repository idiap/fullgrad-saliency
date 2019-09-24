"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models


def save_class_activation_on_image(org_img, gradient, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    gradient = gradient - gradient.min()
    gradient = gradient / (gradient.max())
    gradient = gradient.clip(0,1)

    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    gradient = cv2.resize(gradient, (224,224))
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    # Heatmap on picture
    org_img = np.uint8(org_img * 255).transpose(1,2,0)
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('results', file_name+'.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))
