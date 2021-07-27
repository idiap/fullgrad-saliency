#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""  
    Implement a simpler FullGrad-like saliency algorithm.

    Instead of exactly computing bias-gradients, we only
    extract gradients w.r.t. biases, which are simply
    gradients of intermediate spatial features *before* ReLU.
    The rest of the algorithm including post-processing
    and the aggregation is the same.

    Note: this algorithm is only provided for convenience and
    performance may not be match that of FullGrad. 
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

from saliency.tensor_extractor import FullGradExtractor

class SimpleFullGrad():
    """
    Compute simple FullGrad saliency map 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.model_ext = FullGradExtractor(model, im_size)

    def _getGradients(self, image, target_class=None):
        """
        Compute intermediate gradients for an image
        """

        self.model.eval()
        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        return self.model_ext.getFeatureGrads(image, output_scalar)

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        input_grad, intermed_grad = self._getGradients(image, target_class=target_class)
        
        im_size = image.size()

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Aggregate Intermediate-gradients
        for i in range(len(intermed_grad)):

            # Select only Conv layers 
            if len(intermed_grad[i].size()) == len(im_size):
                temp = self._postProcess(intermed_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True) 
                cam += gradient.sum(1, keepdim=True)

        return cam

