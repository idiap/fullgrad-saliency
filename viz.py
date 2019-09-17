from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import sys
import os
import numpy as np
from vgg_imgnet import *
from misc_functions import *
import cv2

# Import Baselines
from src.gradcam import GradCam
from src.guided_backprop import GuidedBackprop
from src.integrated_grad import IntegratedGrad
from src.smooth_grad import generate_smooth_grad_sq
from src.vanilla_backprop import VanillaBackprop

# Full-gradient helper functions
def get_namedlayer_bias(model):
    # Assumed order of modules : conv -> BN
    # Template not applicable to Layer norm anyway
    # as biases are shaped differently there

    bias = {}

    for m in model.modules():
        if isinstance(m, NamedLayer):
            if isinstance(m.named_layer, nn.Conv2d):
                temp = m.named_layer.bias
                bias[m.number] = temp

            if isinstance(m.named_layer, nn.BatchNorm2d):
                temp = bias[m.number]
                temp = (temp - m.named_layer.running_mean) / torch.sqrt(m.named_layer.running_var + 1e-5)
                temp = temp * m.named_layer.weight
                temp += m.named_layer.bias
                bias[m.number] = temp

    return bias

def get_biasinp(grad, bias):
    # Works only when gradients are 4D tensors
    # Hence not applicable to Linear layers

    veclen = len(grad.keys())
    if veclen != len(bias.keys()): 
        print('Error: bias extracted for ', len(bias.keys()), ' layers, while grad extracted for ', veclen , ' layers')

    for i in range(veclen):
        grad[i] = grad[i] * bias[i].view(1,-1,1,1)
    
    return grad

def post_process(gradient):
    gradient = abs(gradient)
    #gradient = F.relu(gradient)
    gradient = gradient - gradient.min()
    gradient = gradient / (gradient.max())
    return gradient


def full_gradient(model, prep_img, fg_mode = "full", target_class=None, unsigned=True):
    # Full-gradient saliency maps

    model.eval()
    out, feat = model(prep_img)

    img_width = prep_img.size(2)

    veclen = len(feat.keys())
    x = []
    x.append(prep_img)

    for i in range(veclen):
        x.append(feat[i])

    s_bias = get_namedlayer_bias(model)

    if target_class is None:
        target_class = out.data.max(1, keepdim=True)[1]

    agg = 0
    for i in range(prep_img.size(0)):
        agg += out[i,target_class[i]]

    model.zero_grad()
    gradients = torch.autograd.grad(outputs = agg, inputs = x, only_inputs=True)
    s_dict = {}
    for i in range(veclen):
        s_dict[i] = gradients[i+1]

    bias_jac = get_biasinp(s_dict, s_bias)

    # Gradient * image
    grd = gradients[0] * prep_img

    if unsigned == True:
        gradient = post_process(grd).sum(1, keepdim=True)
    else:
        gradient = grd.sum(1, keepdim=True)

    if fg_mode != "second_half":
        cam = (gradient)
    else:
        cam = torch.zeros_like(gradient).to(gradient.device)

    # Bias-gradients
    for i in range(0, veclen):
        if unsigned == True:
            temp = post_process(bias_jac[i]).sum(1, keepdim=True)
        else:
            temp = bias_jac[i].sum(1, keepdim=True)
        gradient = F.interpolate(temp, size=(img_width, img_width), mode = 'bilinear', align_corners=False) 

        if (i <= int(veclen / 2) ) and (fg_mode != "second_half"):
            cam += (gradient)

        if (i > int(veclen / 2)) and (fg_mode != "first_half"):
            cam += (gradient)

    return cam.abs()


def grad_cam(model, prep_img, layer_to_viz):
    # Grad CAM saliency map

    model.eval()
    gc = GradCam(model, target_layer=layer_to_viz)
    cam = gc.generate_cam(prep_img)
    return cam

def smooth_grad_sq(model, prep_img):
    # Smooth grad on vanilla gradients

    model.eval()
    VBP = VanillaBackprop(model)
    param_n = 10
    param_sigma_multiplier = 4
    smooth_grad = generate_smooth_grad_sq(VBP, 
                                        prep_img,
                                        param_n,
                                        param_sigma_multiplier)
    return smooth_grad

def integrated_grad(model, prep_img):
    # Integrated gradients saliency map

    model.eval()
    int_grad = IntegratedGrad(model, prep_img)
    int_grad = abs(int_grad).sum(dim=1, keepdim=True)
    return int_grad

def integrated_grad_multiple_ref(model, prep_img):
    # Integrated gradients over multiple references

    model.eval()
    mean = torch.zeros(prep_img.size()).to(prep_img.device)

    acc = torch.zeros(prep_img.size()).to(prep_img.device)
    for i in range(10):
        #print('Reference number: ', i)
        reference = torch.normal(mean, std = 0.1)
        int_grad = IntegratedGrad(model, prep_img, reference=reference)
        acc += int_grad
    
    return acc.abs().sum(dim=1, keepdim=True)


def guided_grad(model, prep_img):
    # Guided backprop
    model.eval()
    VBP = GuidedBackprop(model)
    grad = VBP.generate_gradients(prep_img)
    cam = abs(grad).sum(dim=1, keepdim=True)
    return cam

def vanilla_grad(model, prep_img):
    # Simple input gradients
    model.eval()
    VBP = VanillaBackprop(model)
    grad = VBP.generate_gradients(prep_img)
    cam = abs(grad).sum(dim=1, keepdim=True)
    return cam

def random(model=None, prep_img = 0):
    mean = torch.zeros_like(prep_img)
    z = torch.normal(mean, std = 1.).to(prep_img.device)
    return z.sum(dim=1, keepdim=True)
