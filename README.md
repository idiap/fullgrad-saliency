# Full-Gradient Saliency Maps 

This code is the reference implementation of the methods described in our NeurIPS 2019
publication ["Full-Gradient Representation for Neural Network
Visualization"](https://arxiv.org/abs/1905.00780).

This repository implements two methods: the reference `FullGrad` algorithm, and a variant called
`Simple FullGrad`, which omits computation of bias parameters for bias-gradients. The related
`full-gradient decomposition` is implemented within `FullGrad`. Note that while `full-gradient
decomposition` applies to any ReLU neural network, `FullGrad` saliency is <b>specific to
CNNs</b>.


## Usage
Simply run  `python dump_images.py`, the saliency maps should be saved consequently in a results folder.

## Interfaces

The FullGrad class has the following methods implemented.

```python
from saliency.fullGrad import FullGrad

# Initialize FullGrad object
fullgrad = FullGrad(model)

# Check completeness property
# done automatically while initializing object
fullgrad.checkCompleteness()

# Obtain fullgradient decomposition
input_gradient, bias_gradients = 
fullgrad.fullGradientDecompose(input_image, target_class)

# Obtain saliency maps
saliency_map = fullgrad.saliency(input_image, target_class)
```

We also introduce a simpler variant called `SimpleFullGrad` which skips bias parameter computations which results in a simpler interface, but no related completeness property or decomposition. 

```python
from saliency.simple_fullgrad import SimpleFullGrad

# Initialize Simple FullGrad object
simple_fullgrad = SimpleFullGrad(model)

# Obtain saliency maps
saliency_map = simple_fullgrad.saliency(input_image, target_class)
```


The relevant bias parameters and bias-gradients of `model` are computed using a `FullGradExtractor` class implemented in `saliency/tensor_extractor.py`.

A correctly implemented `FullGradExtractor` for a given model results in passing the 
`fullgrad.checkCompleteness()` test. The current implementation of `FullGradExtractor` works
for ReLU/BatchNorm based models, and not for models employing LayerNorm, SelfAttention, etc. In most such cases, `SimpleFullGrad` can be used instead.




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
