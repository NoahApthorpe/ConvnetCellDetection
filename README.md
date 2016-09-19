# Convnet Cell Detection!
Automatic cell detection in microscopy data using convolutional networks

## Contributors
- Noah Apthorpe (apthorpe@princeton.edu)
- Alex Riordan (ariordan@princeton.edu)

Please feel free to send us emails with questions, comments, or bug-reports.  Include "Convnet Cell Detection" in the subject.

## Citing
We encourage the use of this tool for research and are excited to see Convnet Cell Detection being applied to experimental workflows.  
If you publish the results of research using this tool or any of the code contained in this repository, we ask that you cite the following paper:
-  N. Apthorpe, et al. "Automatic cell detection in microscopy data using convolutional networks." *Advances in Neural Information Processing Systems.* 2016. 

## Documentation Contents
- [Overview](#overview)
- [Installation](#installation)
- [General Use](#general-use)
- [Detailed Parameter Configuration](#detailed-parameter-configuration) (Advanced Users)

## Overview
Convnet Cell Detection is a data processing pipeline for automatically detecting cells in microscope images using convolutional neural networks (convnets).  We developed and tested this pipeline to find neuron cell bodies in two-photon microscope images, but believe that the technique will be be effective for other cellular microscopy applications as well. 

Convolutional networks are the current state-of-the-art machine learning technique for image and video analysis. There are many excellent online resources available if you would like to learn more about convnets, but we have structured the Convnet Cell Detection pipeline such that only a cursory understanding is necessary.  Convolutional networks are a supervised learning technique, meaning that they need to be trained with images that already have cells-of-interest labeled. If you have existing images with labeled cells, you can train a convolutional network with no additional overhead. Otherwise, you will need to hand-label cells or use another automated cell-detection method on a representative sample of your images to generate training data. 

Once you have trained a convolutional network, you can use it to quickly detect cells in all new images from the same or similar experimental procedure. The Convnet Cell Detection tool has a straightforward command-line and configuration file interface to make this as easy as possible. 

## Installation

## General Use

### Setup a new experiment directory

### Prepare configuration file

### Provide training data

Training data for Convnet Cell Dectection should be a representative sample of your microscopy images large enough to showcase the varying appearance of cells you wish to find. The convolutional network will learn to detect cells that share characteristics with the labeled cells in the training data. If you are missing an entire class or orientation of cells in the training data, do not expect the convolutional network to detect them either.  

The amount of training data you need depends on your particular application and accuracy requirements.  In general, more training data results in more accurate cell detection, but increasing the amount of training data has diminishing returns. The best solution is to use existing hand-labeled images from previous analyses. Otherwise, decide on a tradeoff between hand-labeling time and convolutonal network accuracy, remembering that it's always possible to add additional training data if necessary. 

### Run Convnet Cell Detection Pipeline

### Label new data (forward pass)

## Detailed Parameter Configuration
