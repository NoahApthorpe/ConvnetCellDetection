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
The Convnet Cell Detection pipeline relies on a number of software packages, all of which (except for MATLAB) are free and open source. Please follow the instructions below to install each package. 

###MATLAB (ver R2015b)
  The initial image preprocessing steps of the pipeline rely on MATLAB. We have only tested the pipeline using MATLAB version R2015b, however more recent versions are likely to work as well.  
  
  To download MATLAB R2015b, follow the instructions here: https://www.mathworks.com/matlabcentral/answers/99265-how-do-i-download-an-older-version-of-matlab

###Python 2.7 and related modules
  The majority of the pipeline is based on code written in Python. The Anaconda platform is a convenient tool for installing and maintaining Python modules and environments.  

  Download the Anaconda platform appropriate for your operating system here: 
  https://www.continuum.io/downloads
	
  We’ll need to use Anaconda to install some additional python modules. To do so, navigate to the following links and run the commands therein in a terminal window: 
  https://anaconda.org/anaconda/pil
  https://anaconda.org/anaconda/scikit-image
  https://anaconda.org/conda-forge/tifffile

###Docker
  Our pipeline uses the Docker platform to run ZNN's suite of convolutional neural network tools. Docker is used to create software containers that can be run on any type of machine or operating system. 

  Install the Docker engine for your operating system by following the instructions here: https://docs.docker.com/engine/installation/
	
  To check that Docker is working, run the following commands in a terminal window:
        docker-machine create -d virtualbox --virtualbox-memory 4096 default
	docker-machine start default
	eval $(docker-machine env) #configure shell
	docker run hello-world #check that everything is running properly
	
Now install the ZNN Docker Image by running
	docker pull jpwu/znn:v0.1.4

###FIJI 
  FIJI (FIJI Is Just ImageJ) is an image processing package. Our pipeline uses it to preprocess data images and postprocess the output of the convnets. 

  Install Fiji using the links provided here: 
  http://imagej.net/Fiji/Downloads


#####You are now ready to use the convnet cell detection pipeline. 
## General Use

The following sections will walk through the process of setting up and using the Convnet Cell Detection tool with default settings.  

All `python` commands must either be run from the `src` directory or have `src/` appended to the name of the `.py` file. 

#### Set up new experiment

Run `python create_experiment_dir.py <experiment_name>` to create a new folder in the `data` directory for your experiment. This folder will be pre-initialized with a `main_config.cfg` configuration file and a `labeled` directory to hold pre-labeled training, valdiation, and testing data.

#### Prepare configuration file

You will next need to make one change to the `main_config.cfg` configuration file in your experiment directory.  This is the minimum necessary for the Convnet Cell Detection tool to work on your system. Details about further customization using other parameters in the `main_config.cfg` file are provided in the [Detailed Parameter Configuration](#detailed-parameter-configuration) section.

1. Change the `matlab_path` parameter in the `general` section to point to your system's matlab installation.  The default setting is for a Mac OSX system with Matlab R2015b.

#### Provide training data

Training data for Convnet Cell Dectection should be a representative sample of your microscopy images large enough to showcase the varying appearance of cells you wish to find. The convolutional network will learn to detect cells that share characteristics with the labeled cells in the training data. If you are missing an entire class or orientation of cells in the training data, do not expect the convolutional network to detect them either.  

The amount of training data you need depends on your particular application and accuracy requirements.  In general, more training data results in more accurate cell detection, but increasing the amount of training data has diminishing returns. The best solution is to use existing hand-labeled images from previous analyses. Otherwise, decide on a tradeoff between hand-labeling time and convolutonal network accuracy, remembering that it's always possible to add additional training data if necessary. 

Training data should have the following format:
- Image sequences should be multipage TIFF files with `.tiff` or `.tif` extensions
- Labels should be in ImageJ/Fiji ROI format with one `.zip` zipped folder containing individual `.roi` files per image sequence.
- Each image sequence  should have the same name as its corresponding zipped ROI folder, e.g. V1_09_2016.tiff and V1_09_2016.zip.  There are no restictions on the names themselves, and the names will be preserved throughout the Convnet Cell Detection Pipeline.

Training images and labels should be divided into 3 sets, 1) `training`, 2) `validation`, and 3) `test`, and placed in the corresponding folders in the `data/<experiment_name>/labeled/` directory. The `training` data is used to train the convolutional network.  The `validation` data is used to optimize various postprocessing parameters. The `test` data is used to evaluate the accuracy of the trained network.  We recommend an 60%/20%/20% training/validation/test split if you have lots of labeled data and a 80%/10%/10% split otherwise. 

The `data/example` directory contains an correctly set up example experiment you can use for comparing file types and naming conventions. 

#### Run Convnet Cell Detection pipeline

All components of the Convnet Cell Detection pipeline are acessed via the `pipeline.py` script.  You can choose to run the entire pipeline at once or execute each component individually. 

The command `python pipeline.py complete` will run the entire pipeline, training a new convolutional network from your training data and scoring the result. You can then move directly to [labeling new data](#label-new-data) to use the trained convnet to detect cells in unlabeled images. 

Each of the following sections describes a component of the pipeline in greater detail and provides instructions for executing it individually. 

###### Preprocessing

###### Train convolutional network

###### Postprocessing

#### Label new data

## Detailed Parameter Configuration

#### Custom Network Architecture
