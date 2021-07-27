#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement input-gradient saliency algorithm

    Original Paper:
    Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional 
    networks: Visualising image classification models and saliency maps." ICLR 2014.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


class InputGradient():
    """
    Compute input-gradient saliency map 
    """

    def __init__(self, model, loss=False):
        # loss -> compute gradients w.r.t. loss instead of 
        # gradients w.r.t. logits (default: False)
        self.model = model
        self.loss = loss

    def _getGradients(self, image, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        #outputs = torch.log_softmax(self.model(image),1)
        outputs = self.model(image)

        if target_class is None:
            target_class = (outputs.data.max(1, keepdim=True)[1]).flatten()

        if self.loss:
            outputs = torch.log_softmax(outputs, 1)
            agg = F.nll_loss(outputs, target_class, reduction='sum')
        else:
            agg = -1. * F.nll_loss(outputs, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = image, only_inputs=True, retain_graph=False)[0]

        # First element in the feature list is the image
        return gradients


    def saliency(self, image, target_class=None):

        self.model.eval()
        input_grad = self._getGradients(image, target_class=target_class)
        return torch.abs(input_grad).sum(1, keepdim=True)
        
