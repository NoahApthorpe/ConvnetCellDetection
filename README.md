# Convnet Cell Detection
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
  - [Set up new experiment](#set-up-new-experiment)
  - [Prepare configuration file](#prepare-configuration-file)
  - [Provide training data](#provide-training-data)
  - [Run Convnet Cell Detection pipeline](#run-convnet-cell-detection-pipeline)
  - [Label new data](#label-new-data)
- [Detailed Parameter Configuration](#detailed-parameter-configuration) (Advanced Users)

## Overview
Convnet Cell Detection is a data processing pipeline for automatically detecting cells in microscope images using convolutional neural networks (convnets).  We developed and tested this pipeline to find neuron cell bodies in two-photon microscope images, but believe that the technique will be be effective for other cellular microscopy applications as well. 

Convolutional networks are the current state-of-the-art machine learning technique for image and video analysis. There are many excellent online resources available if you would like to learn more about convnets, but we have structured the Convnet Cell Detection pipeline such that only a cursory understanding is necessary. 

Convolutional networks are a supervised learning technique, meaning that they need to be trained with images that already have cells-of-interest labeled. If you have existing labeled images, you can train a convolutional network with no additional overhead. Otherwise, you will need to hand-label cells or use another automated cell-detection method on a representative sample of your images to generate training data. 

Once you have trained a convolutional network, you can use it to quickly detect cells in all new images from the same or similar experimental procedure. The Convnet Cell Detection tool has a straightforward command-line and configuration file interface to make this as easy as possible. 

The Convnet Cell Detection tool uses the ZNN convolutional network implementation ([https://github.com/seung-lab/znn-release](https://github.com/seung-lab/znn-release)).  While you do not need to understand ZNN in order to use this tool, advanced users may wish to read the [ZNN documentation](http://znn-release.readthedocs.io/en/latest/index.html) to understand some intermediate files created by the Convnet Cell Detection pipeline. 

## Installation

## General Use

All interaction and customization of the Convnet Cell Detection tool occur through the `pipeline.py` script and the `main_config.cfg` configuration file. The following sections will walk through the process of setting up and using the tool with default settings. 

The `data/example` directory contains an correctly set up example experiment you can use for comparing file types and naming conventions. 

#### Set up new experiment

#### Prepare configuration file

#### Provide training data

Training data for Convnet Cell Dectection should be a representative sample of your microscopy images large enough to showcase the varying appearance of cells you wish to find. The convolutional network will learn to detect cells that share characteristics with the labeled cells in the training data. If you are missing an entire class or orientation of cells in the training data, do not expect the convolutional network to detect them either.  

The amount of training data you need depends on your particular application and accuracy requirements.  In general, more training data results in more accurate cell detection, but increasing the amount of training data has diminishing returns. The best solution is to use existing hand-labeled images from previous analyses. Otherwise, decide on a tradeoff between hand-labeling time and convolutonal network accuracy, remembering that it's always possible to add additional training data if necessary. 

Training data should have the following format:
- Image sequences should be multipage TIFF files 
- Labels should be in ImageJ/Fiji ROI format with one zipped folder containing individual `.roi` files per image sequence.

Training images and labels should be divided into 3 sets, 1) `training`, 2) `validation`, and 3) `test`, and placed in the corresponding folders in the `labeled/` directory. The `training` data is used to train the convolutional network.  The `validation` data is used to optimize various postprocessing parameters. The `test` data is used to evaluate the accuracy of the trained network.  We recommend an 60%/20%/20% training/validation/test split if you have lots of labeled data and a 80%/10%/10% split otherwise. 

#### Run Convnet Cell Detection pipeline

###### Preprocessing

###### Train convolutional network

###### Postprocessing

#### Label new data

## Detailed Parameter Configuration
