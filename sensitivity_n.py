from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim import lr_scheduler
import time
import sys
import subprocess
import numpy as np
from viz import *
from vgg_imgnet import *
import cPickle
from scipy import stats

#PATH = '/idiap/temp/ssrinivas/Invertible_nets/viz_cifar/'
#dataset = '/idiap/resource/database/cifar100/'
PATH = '/idiap/temp/ssrinivas/Invertible_nets/viz_imgnet/'
dataset = PATH + 'dataset/'

parser = argparse.ArgumentParser(description='Sanity check arguments')

parser.add_argument( '-s','--saliency_method', choices=["gradCAM", "full-gradient", "random", "smoothgradsq" , "grad", "guidedbackprop", "integratedgrad", "intgradmultiple", "full-gradient-signed"],
                    default="full-gradient",
                    help="Choose the saliency method")

parser.add_argument( '-gl', '--gradCAM_layer', type = float,
                    default = 0.99,
                    help = "which layer to visualize for gradCAM")

parser.add_argument( '-fg', '--full_gradient_type', choices = ["full", "first_half", "second_half"],
                    default = "full",
                    help = "how to visualize full-gradients")

args = parser.parse_args()

opt = ''
if args.saliency_method == 'gradCAM':
        opt = '_' + str(args.gradCAM_layer)
elif args.saliency_method == 'full-gradient':
        opt = '_' + args.full_gradient_type

# Training settings
batch_size = 5
test_batch_size = 100
no_cuda = False
seed = 10

cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

#validation_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR100(dataset, train=False, transform=transforms.Compose([
#                       transforms.ToTensor()
#                   ])),
#    batch_size= batch_size , shuffle=False, **kwargs)
#
#
#def pretrain(model):
#    pretrained = torch.load(PATH + 'vgg09.pt')
#    load_pretrained_model(model, pretrained)
#    return model
#
#model = pretrain(VGG('VGG11').to(device))


validation_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False, **kwargs)


def pretrain(model):
    pretrained = torch.load(PATH + 'vgg16_bn.pth')
    load_pretrained_model(model, pretrained)
    return model

model = pretrain(VGG('DAll').to(device))

def saliency(model, img, args=args):
    
    if args.saliency_method == 'full-gradient':
        cam = full_gradient(model, img, args.full_gradient_type, unsigned=True)
    elif args.saliency_method == 'gradCAM':
        cam = grad_cam(model, img, args.gradCAM_layer)
    elif args.saliency_method == 'smoothgradsq':
        cam = smooth_grad_sq(model, img)
    elif args.saliency_method == 'random':
        cam = random(model, img)
    elif args.saliency_method == 'grad':
        cam = vanilla_grad(model,img)
    elif args.saliency_method == 'guidedbackprop':
        cam = guided_grad(model, img)
    elif args.saliency_method == 'integratedgrad':
        cam = integrated_grad(model, img)
    elif args.saliency_method == 'intgradmultiple':
        cam = integrated_grad_multiple_ref(model, img)
    elif args.saliency_method == 'full-gradient-signed':
        cam = full_gradient(model, img, args.full_gradient_type, unsigned=False)
    
    return cam


def pearson(x, y, accumulate):
    x = np.array(x)
    y = np.array(y)

    batch_size = x.shape[1]

    for i in range(batch_size):
        accumulate.append(abs(np.corrcoef((x[:,i]), (y[:,i]))[0,1]))
        #corr, _ = stats.spearmanr(x[:,i], y[:,i], axis=None)
        #accumulate.append(abs(corr))

    return accumulate


def corruptify(model, data, removal_fraction = 0.1, largest=False):
    # Input batch tensor and output batch tensor with missing pixels
    # largest -> remove most salient pixels?

    batch_size = data.size(0)

    #mean = torch.ones_like(data)
    #std = data.std().item()
    #mean_im = data.mean().item()
    #noise_image = torch.normal(mean_im * mean, std = std).to(device)
    
    salmaps = saliency(model, data) 
    
    salmaps_flat = salmaps.view(batch_size,-1)
    num_pixels_to_remove = int(removal_fraction * salmaps_flat.size(1))
    topvalues, _ = torch.topk(salmaps_flat, k= num_pixels_to_remove , largest=largest)
    leastval = topvalues[:, num_pixels_to_remove-1:]
    for i in range(batch_size):
        if largest == False:
            salmaps_flat[i,:] = (salmaps_flat[i,:] > leastval[i]).float()
        else:
            salmaps_flat[i,:] = (salmaps_flat[i,:] < leastval[i]).float()

    return data * salmaps #+ noise_image * (1. - salmaps), salmaps


def validate(removal_fraction = 0.1):
    model.eval()
    result = []
    for batch_idx, (data, target) in enumerate(validation_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        batch_size = data.size(0)
        model.zero_grad()
        out, feat = model(data)

        maxval, target = out.data.max(1)

        corrupted_data = corruptify(model, data, removal_fraction = removal_fraction, largest=True)
        noisy_out, feat = model(corrupted_data)

        noisy_maxval = torch.zeros_like(maxval).to(device)

        for j in range(batch_size):
            noisy_maxval[j] = noisy_out[j,target[j]]

        temp = abs((maxval - noisy_maxval) / maxval ).data.cpu().numpy()
        #del corrupted_data, feat

        result.append(np.array(temp)) 
    print(removal_fraction, np.mean(result), np.std(result)) # Mean across images of dataset

for i in [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5]:
    validate(i)
        
        




