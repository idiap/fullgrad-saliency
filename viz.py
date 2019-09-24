import torch
import torch.nn as nn
import torch.nn.functional as F


def linearity_test(m):
    # Find out if a given layer is linear or not
    # by manually checking the module type

    # Linear modules to check against
    modules = [nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm]

    match = False
    for mod in modules:
        match = match or isinstance(m, mod)

    return match


class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition for any model
    """

    def __init__(self, model, im_size = (3,224,224) ):
        self.model = model
        self.im_size = (1,) + im_size
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

            if linearity_test(m):
                input_bias = m(input_bias)
                lin_block = 1
            else:
                # check if previous module was linear
                if lin_block:
                    blockwise_biases.append(input_bias.clone().detach())
                    lin_block = 0

        return blockwise_biases

    def _getimplicitBiases(self, image, target_class):
        # TODO: Compute implicit biases that arise due to non-linearities
        # This appends to both the blockwise_biases and blockwise_features list
        None

    def checkCompleteness(self):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases), or there is
        a bug in the code. 
        """

        #Random input image
        input = torch.randn(self.im_size)

        # Get raw outputs
        self.model.eval()
        raw_output = self.model(input)

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(input, target_class=None)
        
        fullgradient_sum = (input_grad[0] * input).sum(dim=(1,2,3))
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum(dim=(1,2,3))

        # Compare raw output and full gradient sum
        assert (raw_output == fullgradient_sum).all(), "Completeness test failed!"  

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

            if linearity_test(m):
                lin_block = 1
            else:
                # check previous module was linear 
                if lin_block:
                    blockwise_features.append(feature.clone())
                lin_block = 0

            feature = m(feature)

        return feature, blockwise_features


    def fullGradientDecompose(self, image, target_class):
        """
        Compute full-gradient decomposition for an image
        """

        out, features = self._getFeatures(input)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        agg = 0
        for i in range(input.size(0)):
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

    def saliency(self, image, target_class):
        #FullGrad saliency
        
        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=target_class)
        
        # Gradient * image
        grd = input_grad[0] * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Bias-gradients
        for i in range(len(bias_grad)):
            temp = self._postProcess(bias_grad[i]).sum(1, keepdim=True)
            gradient = F.interpolate(temp, size=(self.im_size(2), self.im_size(3)), mode = 'bilinear', align_corners=False) 
            cam += gradient

        return cam
        
