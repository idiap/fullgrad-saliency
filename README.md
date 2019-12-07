# Full-Gradient Saliency Maps 

This code is the reference implementation of the methods described 
in our NeurIPS 2019 publication ["Full-Gradient Representation for Neural Network Visualization"](https://arxiv.org/abs/1905.00780).

This repository implements two methods: the reference `FullGrad` algorithm, and a variant called `Simple FullGrad`, which omits computation of bias parameters for bias-gradients. The related `full-gradient decomposition` is implemented within `FullGrad`. Note that while `full-gradient decomposition` applies to any ReLU neural network, `FullGrad` saliency is <b>specific to CNNs</b>.

The codebase currently supports VGG, ResNet and ResNeXt architectures. Extending support for any other architecture of choice should be straightforward, and contributions are welcome! Among non-linearities, only ReLU-like functions are supported. For more information, please read the description of "implicit  biases" in the paper on how to include support for non-ReLU functions.

## Usage
Simply run  `python dump_images.py`, the saliency maps should be saved consequently in a results folder.

## Interfaces

The FullGrad class has the following methods implemented.

```python
from saliency.fullGrad import FullGrad

# Initialize FullGrad object
# see below for model specs
fullgrad = FullGrad(model)

# Check completeness property
# done automatically while initializing object
fullgrad.checkCompleteness()

# Obtain fullgradient decomposition
input_gradient_term, bias_gradient_term = 
fullgrad.fullGradientDecompose(input_image, target_class)

# Obtain saliency maps
saliency_map = fullgrad.saliency(input_image, target_class)
```

Here `model` is an object instance with the following interface. A correctly implemented model interface results in passing the `fullgrad.checkCompleteness()` test.

```python
class CustomModel(nn.Module):
    def forward(self, x):
        # implement forward pass
        ...

    def getBiases(self):
        # obtain all bias parameters
        # and batch norm running means
        # in some order
        ...

    def getFeatures(self, x):
        # obtain intermediate features
        # in the same order as the biases
        # above
        ...

model = CustomModel()
```

We also introduce a simpler variant called Simple FullGrad which skips bias parameter computations which results in a simpler interface, but no related completeness property or decomposition.

```python
from saliency.simple_fullgrad import SimpleFullGrad

# Initialize Simple FullGrad object
simple_fullgrad = SimpleFullGrad(model)

# Obtain saliency maps
saliency_map = simple_fullgrad.saliency(input_image, target_class)
```

Here `model` is an object instance with the following simplified interface.

```python
class CustomModel(nn.Module):
    def forward(self, x):
        # implement forward pass
        ...

    def getFeatures(self, x):
        # obtain intermediate features
        ...

model = CustomModel()
```


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
