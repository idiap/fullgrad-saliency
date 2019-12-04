# Full-Gradient Saliency Maps 

This code is the reference implementation of the methods described 
in our NeurIPS 2019 publication ["Full-Gradient Representation for Neural Network Visualization"](https://arxiv.org/abs/1905.00780).

This repository implements two methods: the reference FullGrad algorithm, and a variant called "simple FullGrad", which omits computation of bias parameters for bias-gradients. 

The codebase currents supports VGG, ResNet and ResNeXt architectures. Extending support for any other architecture of choice should be straightforward, and contributions are welcome! Among non-linearities, only ReLU-like functions are supported. For more information, please read the description of 'implicit  biases' in the paper on how to include support for non-ReLU functions.

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

## Research
If you found our work helpful for your research, please do consider citing us.
```
@inproceedings{srinivas2019fullgrad,
    title={Full-Gradient Representation for Neural Network Visualization},
    author={Srinivas, Suraj and Fleuret, François},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2019}
}
```
