import torch
import torch.nn as nn
import torch.nn.functional as F


def linearity_test(m):
    # Find out if a given layer is linear or not
    # Do this by manually checking or some sort of test

    # modules = [nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm]

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        return True
    else:
        return False


class FullGrad():

    def __init__(self, model):
        self.model = model
        self.blockwise_biases = self.__getBiases__()


    def __getBiases__(self):
        """
        Compute model biases by combining convolution and batchnorm into a single
        linear layer and computing the effective bias of the overall linear layer.
        This is done by passing a Tensor of zeros at the input and looking at the 
        output tensor at the end of every 'linear' block. 
        """

        self.model.eval()
        input_bias = torch.zeros(1,3,224,224) 
        blockwise_biases = [0]

        for m in self.model.modules(): 
            # Assume modules are arranged in "chronological" fashion

            if linearity_test(m):
                input_bias = m(input_bias)
            else:
                if (input_bias != 0).all():
                    blockwise_biases.append(input_bias.clone().detach())
                    input_bias = input_bias * 0.

        return blockwise_biases



    def __getimplicitBiases__(self, image, target_class):
        # TODO: Compute implicit biases that arise due to non-linearities
        None

    def __getFeatures__(self, image):
        #Protected function

        self.model.eval()
        lin_block = 0
        blockwise_features = [image]
        for m in model.modules(): 
            # Assume modules are arranged in "chronological" fashion

            if linearity_test(m):
                lin_block = 1
            else:
                if lin_block == 0:
                    blockwise_features.append(image.clone())
                lin_block = 0


            image = model(image)        

        None

    def decompose(self, image, target_class):
        None

    def checkCompleteness(self):
        #Random input images
        None

    def __postProcess__(self, image):
        #Protected function
        None

    def saliency(self, image, target_class):
        #FullGrad saliency
        None
    



# Full-gradient helper functions
def get_biases_and_features(input, model):
    # Run through the model and pull out biases and intermediate features

    # TODO: Put in device
    input_bias = torch.zeros(1,3,224,224) 

    blockwise_biases = [0]
    blockwise_features = [input]

    for m in model.modules(): 
        # Assume modules are arranged in "chronological" fashion

        if linearity_test(m):
            input_bias = m(input_bias)
        else:
            if (input_bias != 0).all():
                blockwise_biases.append(input_bias.clone().detach())
                blockwise_features.append(feature.clone())

            # TODO: Compute implicit bias of nonlinearity

            input_bias = input_bias * 0.


        feature = model(feature)

    return feature, blockwise_biases, blockwise_features


def full_gradients(model, input, target_class=None):
    out, biases, features = get_biases_and_features(input, model)

    if target_class is None:
        target_class = out.data.max(1, keepdim=True)[1]

    agg = 0
    for i in range(input.size(0)):
        agg += out[i,target_class[i]]

    model.zero_grad()
    gradients = torch.autograd.grad(outputs = agg, inputs = features, only_inputs=True)

    input_gradient = gradients[0]

    bias_gradient = []
    for i in range(1, len(biases)):
        bias_gradient.append(gradients[i] * biases[i]) 

    return input_gradient, bias_gradient


def completeness_test(model, input):
    # Check if full gradients satisfy completeness
    # if they don't, it means there are some unaccounted biases in the model

    # Compute raw outputs of model
    model.eval()
    raw_output = model(input)

    # Compute full-gradients and add them up
    input_grad, bias_grad = full_gradients(model, input, target_class=None)
    
    fullgradient_sum = (input_grad[0] * input).sum(dim=(1,2,3))
    for i in range(len(bias_grad)):
        fullgradient_sum += bias_grad[i].sum(dim=(1,2,3))

    assert (raw_output == fullgradient_sum).all(), "Completeness test failed!"

def post_process(gradient):
    # Absolute value
    gradient = abs(gradient)

    # Rescale operations to ensure gradients lie between 0 and 1
    gradient = gradient - gradient.min()
    gradient = gradient / (gradient.max())
    return gradient


def FullGrad(model, prep_img, target_class=None, post_processing = 'image'):
    # FullGradsaliency maps

    model.eval()
    img_width = prep_img.size(2)

    input_grad, bias_grad = full_gradients(model, prep_img, target_class=target_class)
    
    # Gradient * image
    grd = input_grad[0] * prep_img
    gradient = post_process(grd).sum(1, keepdim=True)
    cam = (gradient)

    # Bias-gradients
    for i in range(len(bias_grad)):

        temp = post_process(bias_grad[i]).sum(1, keepdim=True)
        gradient = F.interpolate(temp, size=(img_width, img_width), mode = 'bilinear', align_corners=False) 
        cam += (gradient)

    return cam


