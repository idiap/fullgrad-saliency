import argparse
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
import numpy as np
from viz import *
from vgg_imgnet import *
from misc_functions import *

PATH = '/idiap/temp/ssrinivas/Interpretation/full-grad/'
dataset = PATH + 'dataset/'

parser = argparse.ArgumentParser(description='Command line arguments for FullGrad')
args = parser.parse_args()

batch_size = 5

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)


def pretrain(model):
    pretrained = torch.load(PATH + 'vgg16_bn.pth')
    load_pretrained_model(model, pretrained)
    return model

model = VGG('B').to(device)
fullgrad = FullGrad(model)

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

unnorm = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

save_path = PATH + 'results/'

def validate():
    model.eval()
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        batch_size = data.size(0)
        model.zero_grad()

        cam = fullgrad.saliency(data)

        for i in range(batch_size):
            filename = save_path + str( (batch_idx+1) * (i+1)) # + '_' + args.saliency_method 

            #utils.save_image(sal[i,:,:,:], filename, nrow=1, padding = 0, normalize = True)
            unnorm_data = unnorm(data[i,:,:,:].cpu())
            save_class_activation_on_image(unnorm_data.data.cpu().numpy(), cam[i,:,:,:].data.cpu().numpy(), filename, use_image=1.0)

        
create_folder(save_path)
validate()

        
        




