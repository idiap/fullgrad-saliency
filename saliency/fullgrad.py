#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Implement FullGrad saliency algorithm """

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

from saliency.tensor_extractor import FullGradExtractor

class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.im_size = (1,) + im_size
        self.model_ext = FullGradExtractor(model, im_size)
        self.biases = self.model_ext.getBiases()
        self.checkCompleteness()

    def checkCompleteness(self):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases of non-linearities).

        """

        cuda = next(self.model.parameters()).is_cuda
        device = torch.device("cuda" if cuda else "cpu")

        #Random input image
        input = torch.randn(self.im_size).to(device)

        # Get raw outputs
        self.model.eval()
        raw_output = self.model(input)

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(input, target_class=None)

        fullgradient_sum = (input_grad * input).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients."
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())
        assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=1e-4), err_string + err_message
        print('Completeness test passed for FullGrad.')


    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """

        self.model.eval()
        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        input_gradient, feature_gradients = self.model_ext.getFeatureGrads(image, output_scalar)

        # Compute feature-gradients \times bias 
        bias_times_gradients = []
        L = len(self.biases)

        for i in range(L):

            # feature gradients are indexed backwards 
            # because of backprop
            g = feature_gradients[L-1-i]

            # reshape bias dimensionality to match gradients
            bias_size = [1] * len(g.size())
            bias_size[1] = self.biases[i].size(0)
            b = self.biases[i].view(tuple(bias_size))
            
            bias_times_gradients.append(g * b.expand_as(g))

        return input_gradient, bias_times_gradients

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
        #FullGrad saliency

        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=target_class)

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        im_size = image.size()

        # Aggregate Bias-gradients
        for i in range(len(bias_grad)):

            # Select only Conv layers
            if len(bias_grad[i].size()) == len(im_size): 
                temp = self._postProcess(bias_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True)
                cam += gradient.sum(1, keepdim=True)

        return cam

