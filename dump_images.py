#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torchvision import datasets, transforms, utils, models
import os

# Import saliency methods
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.smooth_fullgrad import SmoothFullGrad

from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.smoothgrad import SmoothGrad

from misc_functions import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 5

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

# Use pretrained ResNet-18 provided by PyTorch
model = models.resnet18(pretrained=True)
model = model.to(device)

# Initialize saliency methods
saliency_methods = {
# FullGrad-based methods
'fullgrad': FullGrad(model),
'simple_fullgrad': SimpleFullGrad(model),
'smooth_fullgrad': SmoothFullGrad(model),

# Other saliency methods from literature
'gradcam': GradCAM(model),
'inputgrad': InputGradient(model),
'smoothgrad': SmoothGrad(model)
}

def compute_saliency_and_save():
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to(device).requires_grad_()

        # Compute saliency maps for the input data
        for s in saliency_methods:
            saliency_map = saliency_methods[s].saliency(data)

            # Save saliency maps
            for i in range(data.size(0)):
                filename = save_path + str( (batch_idx+1) * (i+1))
                image = unnormalize(data[i].cpu())
                save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg')


if __name__ == "__main__":
    # Create folder to saliency maps
    save_path = PATH + 'results/'
    create_folder(save_path)
    compute_saliency_and_save()
    print('Saliency maps saved.')







