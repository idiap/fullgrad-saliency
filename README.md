# Full-Gradient Saliency Maps 

This code is the reference implementation of the methods described 
in our NeurIPS 2019 publication ["Full-Gradient Representation for Neural Network Visualization"](https://arxiv.org/abs/1905.00780).

This repository implements two methods: the reference FullGrad algorithm, and a variant called "simple FullGrad", which omits computation of bias parameters for bias-gradients. 


## Usage
Simply run the following command

``` 
python dump_images.py
``` 

The saliency maps should be saved consequently in a results folder. 

## Dependencies
``` 
torch torchvision cv2 numpy 
```

