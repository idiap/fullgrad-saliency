#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement Smooth FullGrad saliency algorithm, which involves 
    a SmoothGrad-like noise averaging of the input-gradient and 
    bias-gradient maps before proceeding to aggregate spatially.

    Note: this algorithm is only provided for convenience and
    performance may not be match that of FullGrad. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
from .fullgrad import FullGrad

class SmoothFullGrad():
    """
    Compute smooth Fullgrad 
    """

    def __init__(self, model, num_samples=25, std_spread=0.15, im_size = (3,224,224)):
        self.model = model
        self.num_samples = num_samples
        self.std_spread = std_spread
        self.fg = FullGrad(model, im_size)

    def saliency(self, image, target_class=None):
        #SmoothFullGrad saliency
        
        self.model.eval()
        std_dev = self.std_spread * (image.max().item() - image.min().item())

        cam = None
        for i in range(self.num_samples):
            noise = torch.normal(mean = torch.zeros_like(image).to(image.device), std = std_dev)
            input_grad, bias_grad = self.fg.fullGradientDecompose(image + noise, target_class)

            if cam is None:
                cam = [0]*(len(bias_grad)+1)

            # Aggregate Input-gradient * image
            grd = input_grad * (image + noise)
            cam[0] += grd.sum(1, keepdim=True)

            # Aggregate Bias-gradients of conv layers
            for j in range(len(bias_grad)):
                cam[j+1] += bias_grad[j]

        im_size = image.size()

        final = torch.zeros_like(cam[0]).to(image.device) 

        # Aggregate averaged gradient maps
        for k in range(len(cam)):
            if len(cam[k].size()) == len(im_size): 
                temp = self.fg._postProcess(cam[k])
                final += F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True).sum(1, keepdim=True) 

        return final
        
