## Continual Object Detection

This repo has code for running continual object detection experiments on top of the 
[Detectron2](https://github.com/facebookresearch/detectron2) library. It is fully written outside Detectron, 
which allows easy modifiation and upgrades to newer, future versions of Detectron. This repo serves three main 
purposes: continual object dataloaders, an implemenation and improvement of FasterILOD, and extensive logging 
capabilities for analysis of object detectors.

### Dataloaders

The [data-sets](./datasets) module contains dataloaders and integration with Detectron's dataloaders. See 
[here](./datasets/utils.py) for a list of all registered datasets. For VOC, it's easy to define new splits on the
VOC2007 benchmark. COCO is currently limited to the 40+40 setting, but this will be upgraded soon. The SODA10M are those
defined for the ICCV '21 challenge, see [TODO]. 

### Faster ILOD+
The `ilod.py` file is both an example of how to implement COD methods without changing the Detectron2 library, 
a reimplemenation of [FasterILOD](https://doi.org/10.1016/j.patrec.2020.09.030), and an improvement thereof. See the
documentation of the file for more details. `train.py` contains logic to train and test such a model. The 
[config](./configs) folder contains example configs for VOC10+10. The [scripts](./scripts) folder contains an example
script for the training thereof.

### Logging and Analysis
`log_utils.py` implements a decorator for the inference call on a GeneralizedRCNN model of detectron. It has two main 
capabilities: (1) it creates a hook of the RPN, which logs all proposals to a file on inference. This allows for a
separate analysis of the RPN within a two-stage detector, since its performance is hard to judge given only the
predictions of the ROI-heads. (2) Seperate evaluation of the ROI-heads, by evaluating the ground truth boxes of each
image and log those to a file. Both files can then be analyzed with the [analysis](./analysis) module. _(**Note:** 
the code of the analysis module is not well documented yet, but that will come soon)_. These logging functionalities
are separate from continual detection, and can thus be used with any GeneralizedRCNN model.