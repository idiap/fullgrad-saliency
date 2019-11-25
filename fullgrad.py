#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Implement FullGrad saliency algorithm """

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


<<<<<<< HEAD
=======
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


>>>>>>> a840bb6aa16cfb37db3f4fc9b058a7ccccc1fd78
class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition 
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.im_size = (1,) + im_size
<<<<<<< HEAD
        self.model.eval()
        self.blockwise_biases = self.model.getBiases()
        self.checkCompleteness()

    def _getimplicitBiases(self, image, target_class):
        # TODO: Compute implicit biases that arise due to non-ReLU non-linearities
        # This appends to both the blockwise_biases and blockwise_features list
        pass
=======
        self.blockwise_biases = self._getBiases()
        self.checkCompleteness()

    def _getBiases(self):
        """
        Compute model biases by combining convolution and batchnorm into a single
        linear layer and computing the effective bias of the overall linear layer.
        This is done by passing a Tensor of zeros at the input and looking at the 
        output tensor at the end of every 'linear' block. 
        """

        self.model.eval()
        input_bias = torch.zeros(self.im_size) 
        lin_block = 0
        blockwise_biases = [0]

        for m in self.model.modules(): 
            # Assume modules are arranged in "chronological" fashion

            if linearity_test(m) == 'linear':

                if isinstance(m, nn.Linear):
                    input_bias = input_bias.view(1,-1)

                lin_block = 1
                input_bias = m(input_bias)
            elif linearity_test(m) == 'nonlinear':
                # check if previous module was linear
                if lin_block:
                    blockwise_biases.append(input_bias.clone())
                    lin_block = 0

                input_bias = m(input_bias) * 0.

        if lin_block:
            blockwise_biases.append(input_bias.clone().detach())

        return blockwise_biases

    def _getimplicitBiases(self, image, target_class):
        # TODO: Compute implicit biases that arise due to non-ReLU non-linearities
        # This appends to both the blockwise_biases and blockwise_features list
        None
>>>>>>> a840bb6aa16cfb37db3f4fc9b058a7ccccc1fd78

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
<<<<<<< HEAD

        fullgradient_sum = (input_grad * input).sum(dim=(1,2,3))
=======
        
        fullgradient_sum = (input_grad[0] * input).sum(dim=(1,2,3))
>>>>>>> a840bb6aa16cfb37db3f4fc9b058a7ccccc1fd78
        for i in range(len(bias_grad)):
            temp = bias_grad[i].view(1,-1)
            fullgradient_sum += temp.sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients. Please check vgg_imagenet.py for more information."
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())  
        assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=0.01), err_string + err_message
        print('Completeness test passed for FullGrad.') 

<<<<<<< HEAD
=======
    def _getFeatures(self, image):
        """
        Compute intermediate features at the end of the every linear
        block, for a given input image.
        """

        self.model.eval()
        lin_block = 0
        blockwise_features = [image]
        feature = image

        for m in self.model.modules():
            # Assume modules are arranged in "chronological" fashion

            if linearity_test(m) == 'linear':
                lin_block = 1

                if isinstance(m, nn.Linear):
                    feature = feature.view(feature.size(0),-1)

                feature = m(feature)
            elif linearity_test(m) == 'nonlinear':
                # check previous module was linear 
                if lin_block:
                    blockwise_features.append(feature)
                lin_block = 0
                feature = m(feature)

        if lin_block:
            blockwise_features.append(feature)

        assert len(blockwise_features) == len(self.blockwise_biases), "Number of features must be equal to number of biases"

        return feature, blockwise_features

>>>>>>> a840bb6aa16cfb37db3f4fc9b058a7ccccc1fd78

    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """

        image = image.requires_grad_()
<<<<<<< HEAD
        out, features = self.model.getFeatures(image)
=======
        out, features = self._getFeatures(image)
>>>>>>> a840bb6aa16cfb37db3f4fc9b058a7ccccc1fd78

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
        
