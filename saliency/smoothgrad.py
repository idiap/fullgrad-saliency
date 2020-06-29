#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement SmoothGrad saliency algorithm

    Original paper:
    Smilkov, Daniel, et al. "Smoothgrad: removing noise by adding noise." 
    arXiv preprint arXiv:1706.03825 (2017).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

class SmoothGrad():
    """
    Compute smoothgrad 
    """

    def __init__(self, model, num_samples=100, std_spread=0.15):
        self.model = model
        self.num_samples = num_samples
        self.std_spread = std_spread

    def _getGradients(self, image, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        out = self.model(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]
            target_class = target_class.flatten()

        loss = -1. * F.nll_loss(out, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        input_gradient = torch.autograd.grad(outputs = loss, inputs = image, only_inputs=True)[0]

        return input_gradient

    def saliency(self, image, target_class=None):
        #SmoothGrad saliency
        
        self.model.eval()

        grad = self._getGradients(image, target_class=target_class)
        std_dev = self.std_spread * (image.max().item() - image.min().item())

        cam = torch.zeros_like(image).to(image.device)
        # add gaussian noise to image multiple times
        for i in range(self.num_samples):
            noise = torch.normal(mean = torch.zeros_like(image).to(image.device), std = std_dev)
            cam += (self._getGradients(image + noise, target_class=target_class)) / self.num_samples

        return cam.abs().sum(1, keepdim=True)
        
