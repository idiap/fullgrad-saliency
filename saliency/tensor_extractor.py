#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Extract feature-gradient and bias tensors from PyTorch models. """

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model, im_size = (3,224,224)):
        self.model = model
        self.im_size = im_size

        self.biases = []
        self.feature_grads = []
        self.grad_handles = []

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                
                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)


    def _extract_layer_bias(self, module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different 
        # from batchnorm bias parameter. 
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b) 
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm2d):
            b = - (module.running_mean * module.weight 
                    / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    def getBiases(self):
        # dummy function to get biases
        return self.biases

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalar):
        
        # Empty feature grads list 
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs = output_scalar, inputs = x)[0]

        return input_gradients, self.feature_grads


        




    