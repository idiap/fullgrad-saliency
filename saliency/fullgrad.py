#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Implement FullGrad saliency algorithm """

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.im_size = (1,) + im_size
        self.model.eval()
        self.blockwise_biases = self.model.getBiases()
        self.checkCompleteness()

    def _getimplicitBiases(self, image, target_class):
        # TODO: Compute implicit biases that arise due to non-ReLU non-linearities
        # This appends to both the blockwise_biases and blockwise_features list
        pass

    def checkCompleteness(self):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases). Check
        vgg_imagenet.py for more information.

        """

        #Random input image
        input = torch.randn(self.im_size)

        # Get raw outputs
        self.model.eval()
        raw_output = self.model(input)

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(input, target_class=None)

        fullgradient_sum = (input_grad * input).sum(dim=(1,2,3))
        for i in range(len(bias_grad)):
            temp = bias_grad[i].view(1,-1)
            fullgradient_sum += temp.sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients. Please check vgg_imagenet.py for more information."
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())  
        assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=0.00001), err_string + err_message
        print('Completeness test passed for FullGrad.') 


    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """

        image = image.requires_grad_()
        out, features = self.model.getFeatures(image)

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
        bias_gradient = []
        for i in range(1, len(gradients)):
            bias_gradient.append(gradients[i] * self.blockwise_biases[i]) 
        
        return input_gradient, bias_gradient

    def _postProcess(self, input):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        input = input - input.min()
        input = input / (input.max())
        return input

    def saliency(self, image, target_class=None):
        #FullGrad saliency
        
        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=target_class)
        
        # Input-gradient * image
        grd = input_grad[0] * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Bias-gradients of conv layers
        for i in range(len(bias_grad)):
            # Checking if bias-gradients are 4d tensors
            if len(bias_grad[i].size()) == 4: 
                temp = self._postProcess(bias_grad[i])
                gradient = F.interpolate(temp, size=(self.im_size[2], self.im_size[3]), mode = 'bilinear', align_corners=False) 
                cam += gradient.sum(1, keepdim=True)

        return cam
        
