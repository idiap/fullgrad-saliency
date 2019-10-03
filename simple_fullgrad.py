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
    performance may not be match that of FullGrad for different
    post-processing functions.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


def linearity_test(m):
    # Find out if a given layer is linear or not
    # by manually checking the module type

    # Linear modules to check against
    lin_modules = [nn.Conv2d, nn.BatchNorm2d, nn.Linear]
    # Nonlinear modules to check against
    nonlin_modules = [nn.ReLU, nn.MaxPool2d]

    lin_match = False
    for mod in lin_modules:
        lin_match = lin_match or isinstance(m, mod)

    nonlin_match = False
    for mod in nonlin_modules:
        nonlin_match = nonlin_match or isinstance(m, mod)

    if lin_match:
        return 'linear'
    elif nonlin_match:
        return 'nonlinear'
    else:
        # Any other modules are ignored (E.g.: Sequential, ModuleList)
        return None


class SimpleFullGrad():
    """
    Compute simple FullGrad saliency map 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.im_size = (1,) + im_size


    def _getFeatures(self, image):
        """
        Compute intermediate features at the end of the every linear
        block, for a given input image. Get feature before every 
        ReLU layer at the convolutional (feature extraction) layers
        """

        self.model.eval()
        lin_block = 0
        blockwise_features = [image]
        feature = image

        for m in self.model.modules():
            # Assume modules are arranged in "chronological" fashion

            if isinstance(m, nn.ReLU):
                # Get pre-ReLU activations for conv layers
                if len(feature.size()) == 4:
                    blockwise_features.append(feature)

            if linearity_test(m) is not None:
                if isinstance(m, nn.Linear):
                    feature = feature.view(feature.size(0),-1)
                feature = m(feature)

        return feature, blockwise_features


    def _getGradients(self, image, target_class=None):
        """
        Compute intermediate gradients for an image
        """

        image = image.requires_grad_()
        out, features = self._getFeatures(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        agg = 0
        for i in range(image.size(0)):
            agg += out[i,target_class[i]]

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = features, only_inputs=True)

        # First element in the feature list is the image
        input_gradient = gradients[0]

        # Loop through remaining gradients
        intermediate_gradient = []
        for i in range(1, len(gradients)):
            intermediate_gradient.append(gradients[i]) 
        
        return input_gradient, intermediate_gradient

    def _postProcess(self, input):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        input = input - input.min()
        input = input / (input.max())
        return input

    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        input_grad, intermed_grad = self._getGradients(image, target_class=target_class)
        
        # Input-gradient * image
        grd = input_grad[0] * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Intermediate-gradients
        for i in range(len(intermed_grad)):
            temp = self._postProcess(intermed_grad[i])
            gradient = F.interpolate(temp, size=(self.im_size[2], self.im_size[3]), mode = 'bilinear', align_corners=False) 
            cam += gradient.sum(1, keepdim=True)

        return cam
        
