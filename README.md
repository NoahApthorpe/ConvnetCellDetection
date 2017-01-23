# Convnet Cell Detection
Automatic cell detection in microscopy data using convolutional networks


## Documentation Contents
- [Contributors](#contributors)
- [Citing](#citing)
- [Overview](#overview)
- [Installation](#installation)
- [General Use](#general-use)
  - [Set up new experiment](#set-up-new-experiment)
  - [Prepare configuration file](#prepare-configuration-file)
  - [Provide training data](#provide-training-data)
  - [Run Convnet Cell Detection pipeline](#run-convnet-cell-detection-pipeline)
    - [Preprocessing](#preprocessing)
    - [Training](#train-convolutional-network)
    - [Forward Pass](#forward-pass)
    - [Postprocessing](#postprocessing)
  - [Label new data](#label-new-data)
  - [Output Format](#output-format)
  - [Scoring](#scoring)
  - [Visualization & Manual Thresholding](#visualization-and-manual-thresholding) (Optional)
- [Changing Network Architectures](#changing-network-architectures) (Advanced Users)
- [Parameter Descriptions](#parameter-descriptions) (Advanced Users)

## Contributors
- Noah Apthorpe (apthorpe@princeton.edu)
- Alex Riordan (ariordan@princeton.edu)

Please feel free to send us emails with questions, comments, or bug-reports.  Include "Convnet Cell Detection" in the subject.

## Citing
We encourage the use of this tool for research and are excited to see Convnet Cell Detection being applied to experimental workflows. If you publish the results of research using this tool or any of the code contained in this repository, we ask that you cite the following paper:
-  N. Apthorpe, et al. "Automatic neuron detection in calcium imaging data using convolutional networks." *Advances in Neural Information Processing Systems.* 2016. 

## Overview
Convnet Cell Detection is a data processing pipeline for automatically detecting cells in microscope images using convolutional neural networks (convnets).  We developed and tested this pipeline to find neuron cell bodies in two-photon microscope images, but believe that the technique will be be effective for other cellular microscopy applications as well. 

Convolutional networks are the current state-of-the-art machine learning technique for image and video analysis. There are many excellent online resources available if you would like to learn more about convnets, but we have structured the Convnet Cell Detection pipeline such that only a cursory understanding is necessary. 

Convolutional networks are a supervised learning technique, meaning that they need to be trained with images that already have cells-of-interest labeled. If you have existing labeled images, you can train a convolutional network with no additional overhead. Otherwise, you will need to hand-label cells or use another automated cell-detection method on a representative sample of your images to generate training data. 

Once you have trained a convolutional network, you can use it to quickly detect cells in all new images from the same or similar experimental procedure. The Convnet Cell Detection tool has a straightforward command-line and configuration file interface to make this as easy as possible. 

The Convnet Cell Detection tool uses the ZNN convolutional network implementation ([https://github.com/seung-lab/znn-release](https://github.com/seung-lab/znn-release)).  While you do not need to understand ZNN in order to use this tool, advanced users may wish to read the [ZNN documentation](http://znn-release.readthedocs.io/en/latest/index.html) in order to create new network architectures and understand some intermediate files created by the Convnet Cell Detection pipeline.

## Installation

### Amazon AMI
The easiest way to use the Convnet Cell Detection pipeline is with the [Amazon EC2 AMI ami-a2f8d1b5](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-a2f8d1b5).  We recommend launching the AMI in an EC2 instance with at least 32GB of RAM (we use a c4.8xlarge instance).  The ConvnetCellDetection repository is in the home directory of the AMI and all dependencies are pre-installed. We suggest you use a session manager such as `tmux` to ensure that training is not interrupted if your connection to the EC2 instance is lost.

### Local Install
The Convnet Cell Detection pipeline can also be installed and run on your computer or server. The pipeline relies on a number of software packages, all of which are free and open source. Please follow the instructions below to install each package. 

####Python 2.7 and related modules
The majority of the pipeline is based on code written in Python. The `requirements.txt` file in the `src/` directory of the ConvnetCellDetection repository lists all required Python libraries.  The bash commands `pip install numpy; pip install -r requirements.txt` (preferably in a Python [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)) will install all requirements. 

Alternatively, you can use the Anaconda platform for installing and maintaining Python modules and environments.  Download the Anaconda platform appropriate for your operating system here:  
https://www.continuum.io/downloads
	
 You’ll need to use Anaconda to install some additional python modules. To do so, run these commands in a terminal window: 
  - `conda install -c anaconda pil=1.1.7`
  - `conda install -c anaconda scikit-image=0.12.3`
  - `conda install -c conda-forge tifffile=0.9.0`

####Docker
  Our pipeline uses the Docker platform to run ZNN's suite of convnet tools. Docker is used to create software containers that can be run on any machine or operating system. 

  Install the Docker engine for your operating system by following the instructions here: https://docs.docker.com/engine/installation/
	
  To check that Docker is working, run the following commands in a terminal window:
  ```
  docker-machine create -d virtualbox --virtualbox-memory 8192 convnet-cell-detection-8192
  docker-machine start convnet-cell-detection-8192
  eval $(docker-machine env convnet-cell-detection-8192) #configure shell
  docker run hello-world #check that everything is running properly
  ```
  
  Now install the ZNN Docker Image by running
  ```
  docker pull jpwu/znn:v0.1.4
  ```
  
####FIJI 
  FIJI (FIJI Is Just ImageJ) is an image processing package. FIJI is *not* explicitly required for our pipeline, but it is the best way to view multipage TIFF videos and hand-label cells-of-interest for training.  

Install Fiji using the links provided here:  
http://imagej.net/Fiji/Downloads


#####You are now ready to use the convnet cell detection pipeline. 

## General Use

The following sections will walk through the process of setting up and using the Convnet Cell Detection tool with default settings.  

The primary interface to the pipeline is via the `pipeline.py` script, which takes 2 command line arguments. The first argument is the step of the pipeline you wish to run. The options are `create`, `complete`, `preprocess`, `train`, `forward`, `postprocess`, and `score`. The second argument is either a new experiment name or a path to a configuration file. Each pipeline step will be explained below. 

All `python` commands must be run from the `src` directory of the `ConvnetCellDetection` repository to ensure corrent relative file paths. 

### Set up new experiment

Run `python pipeline.py create <experiment_name>` to create a new folder in the `data` directory for your experiment. This folder will be pre-initialized with a `main_config.cfg` configuration file and a `labeled` directory to hold pre-labeled training, valdiation, and testing data.

Each "experiment" directory in `data/` corresponds to a single convnet. Once the convnet is trained, you can use it to label as much new data as you wish (see [Label new data](#label-new-data))

The `labeled` directory and it's `training/validation/test` subdirectories are specifically for labeled data and are detected by name in the Python scripts.  Do not re-name these directories.

For ease of inspection, the output of each intermediate step in the pipeline is saved in a unique directory (e.g. `labeled_preprocessed` and `labeled_training_output`).  Do not re-name these directories as they are detected by name in the Python scripts. 

The `data/example` directory contains an correctly set up example experiment you can use for comparing file types and naming conventions. You will need to change the `net_arch_fpath` parameter in `data/example/main_config.cfg` to the absolute path in your filesystem in order to be able to run the pipeline on the example experiment. 

### Prepare configuration file

The `main_config.cfg` file in each experiment directory contains customizeable parameters.  This file defaults to the parameters used for training the (2+1)D network on the V1 dataset described in our [NIPS paper](#citing).  Note that all relative filepaths in the config file are relative to the `src/` directory. 

The first parameter (`data_dir`) points to the subdirectory of your experiment containing the data you wish to process on the next run of the pipeline. The `data_dir` parameter for training should be a `labeled` directory (e.g. `../data/example/labeled`). When you are ready to run a forward pass on new (non-labeled) data, you will need to change this to point to a new subdirectory of your experiment (see [Label new data](#label-new-data)).

Details about further customization using other parameters in the `main_config.cfg` file are provided in the [Detailed Parameter Configuration](#detailed-parameter-configuration) section.

### Provide training data

Training data for Convnet Cell Dectection should be a representative sample of your microscopy images large enough to showcase the varying appearance of cells you wish to find. The convolutional network will learn to detect cells that share characteristics with the labeled cells in the training data. If you are missing an entire class or orientation of cells in the training data, do not expect the convolutional network to detect them either.  

The amount of training data you need depends on your particular application and accuracy requirements.  In general, more training data results in more accurate cell detection, but increasing the amount of training data has diminishing returns. The best solution is to use existing hand-labeled images from previous analyses. Otherwise, decide on a tradeoff between hand-labeling time and convolutonal network accuracy, remembering that it's always possible to add additional training data if necessary. 

Training data should have the following format:
- Image sequences should be multipage TIFF files with `.tiff` or `.tif` extensions
- Labels should be in ImageJ/Fiji ROI format with one `.zip` zipped folder containing individual `.roi` files per image sequence.
- Each image sequence  should have the same name as its corresponding zipped ROI folder, e.g. V1_09_2016.tiff and V1_09_2016.zip.  There are no restictions on the names themselves, and the names will be preserved throughout the Convnet Cell Detection Pipeline.

Training images and labels should be divided into 3 sets, 1) `training`, 2) `validation`, and 3) `test`, and placed in the corresponding folders in the `data/<experiment_name>/labeled/` directory. The `training` data is used to train the convolutional network.  The `validation` data is used to optimize various postprocessing parameters. The `test` data is used to evaluate the accuracy of the trained network.  We recommend an 60%/20%/20% training/validation/test split if you have lots of labeled data and a 80%/10%/10% split otherwise.  

### Run Convnet Cell Detection pipeline

The command `python pipeline.py complete <config file path>` will run the entire pipeline. For the example experiment, this would be `python pipeline.py complete ../data/example/main_config.cfg`. 

If the `data_dir` parameter of the `main_config.cfg` file points to a `labeled` directory, this command will train the convolutional network from your training data and score the result. You can then move directly to [labeling new data](#label-new-data) to use the trained convnet to detect cells in unlabeled images. 

Each of the following sections describes a component of the pipeline in greater detail and provides instructions for executing it individually. This is useful if there is a problem and the script stops prematurely, if you wish to inspect the intermediate output of each pipeline step before moving on, or if you wish to re-run a particular step using different parameters.

##### Preprocessing

The preprocessing component of the Convnet Cell Detection pipeline prepares supplied images for convnet training, as follows:

1. Downsampling (*off by default*): The images stacks are average-pooled with default 167-frame strides and then max-pooled with default 6-frame strides. This downsampling reduces noise and makes the dataset into a more manageable size. *This step can be turned on by setting `do_downsample = 1` in the `[general]` section of the `main_config.cfg` file.*
2. Time equalize: All image stacks are equalized to the same number of frames (default 50) by averaging over sets of consecutive frames. This is necessary for 3-dimensional filters in the ZNN convolutional network implementation. 
3. Contrast improvement: Pixel values above the 99th percentile and below the 3rd percentile are clipped and the resulting values are normalized to [0,1].
4. Convert cell labels to centroids: Our research indicates that convolutional networks do a better job distinguishing adjacent cells if the cell labels provided in the training data are reduced to a small centroid.  

You can run just the preprocessing component of the pipeline with the command `python pipeline.py preprocess <config file path>`. For the example experiment, this would be `python pipeline.py preprocess ../data/example/main_config.cfg`. 

##### Train convolutional network

The training component of the pipeline trains a convnet using ZNN in a Docker virtual machine. You can run just the training component of the pipeline with the command `python pipeline.py train <config file path>`. For the example experiment, this would be `python pipeline.py train ../data/example/main_config.cfg`. 

This command will start a docker image and begin ZNN training. It will print the training iteration and the current pixel error periodically. The trained network is automatically saved every 1000 iterations.  Training will continue until you press `ctrl-c` or it reaches the value of the `max_iter` parameter of the configuration file. If you re-run the training command, it will resume training at the last saved iteration. If you wish to restart training, you will need to delete the saved `.h5` files in the `labeled_training_output` directory. If you are running the pipeline on a server, we suggest you use a session manager such as `tmux` to ensure that training is not interrupted if your connection to the server is lost.  

##### Forward Pass

The forward pass uses the trained network to label the input images with probabilities of being inside an ROI.  You can run just the forward pass component of the pipeline with the command `python pipeline.py forward <config file path>`. For the example experiment, this would be `python pipeline.py forward ../data/example/main_config.cfg`. The resulting files are saved in the `labeled_training_ouput/` subdirectory of your experiment directory. The files are sorted into `training|validation|test` directories in the first step of postprocessing. The files ending with `_output_0.tif` are images with lighter pixels having higher probabilities of being inside a cell.  These are the files used for the rest of the pipeline.

##### Postprocessing

The postprocessing component of the pipeline converts the output of the convnet into ROI binary masks, as follows:

1. Thresholding out pixels with low probability values
2. Removing small connected components
3. Weighting resulting pixels with a normalized distance transform ro favor pixels in the center of circular regions.
4. Performing marker-based watershed labeling with local max markers
5. Merging small watershed regions
6. Applying the Cell Magic Wand tool to the preprocessed images at the centroids of the watershed regions. 

You can run just the preprocessing component of the pipeline with the command `python pipeline.py postprocess <config file path>`. For the example experiment, this would be `python pipeline.py postprocess ../data/example/main_config.cfg`. 

To optimize thresholding and minimum size values using the validation set, run the same command, but set

1. The `do_gridsearch_postprocess_params` parameter in `main_config.cfg` to `1`. 
2. The parameters in the `postprocessing optimization` section of `main_config.cfg` to custom grid search ranges and number of steps if desired.   

This wil create a new file in your experiment directory with the best postprocessing parameters that will be used for forward passes as long as the `do_gridsearch_postprocess_params` parameter in `main_config.cfg` is set to `1`. 

### Label new data

Once a convnet is trained, labeling new data is simple:

1. Place the TIFF files you wish to label in a new directory in your experiment folder. 
2. Change the `data_dir` parameter in the `main_config.cfg` file of your experiment to point to the new directory
3. Run `python pipeline.py complete <path to config>` (or all of the individual pipeline steps except training)

### Output Format

The postprocessing step of the pipeline saves automatically detected ROIs in two formats:

1. A compressed Numpy .npz file containing two saved arrays:
   1. The ROIs as binary masks. The code `rois = numpy.load('<name>.npz')['rois']` (one line) will load the ROIs into a 3D Numpy array with dimensions (ROI #, binary mask width, binary mask height). The code `matplotlib.pyplot.imshow(rois[0], cmap='gray')` will plot a single ROI. The code `matplotlib.pyplot.imshow(rois.max(axis=0), cmap='gray')` will plot all ROIs on a single image.  
   2. The average probability assigned by the convolutional network to the pixels inside each ROI. The code `roi_probs = numpy.load('<name>.npz')['roi_probabilities']` will load these probabilties as a 1D Numpy array with indices corresponding to the 3D ROI array in the same file. These probabilities can be used as a heuristic to rank detected ROIs in order of confidence if desired. 
2. A 2D TIFF binary mask with white pixels inside detected ROIs

### Scoring

The scoring step of the pipeline assigns precision, recall, and F1 (harmonic mean of precision and recall) scores to the training, validation, and test sets. This step can only be run on labeled data (because a "ground-truth" is needed for scoring). The scores are output as `.txt` files in  `labeled_postprocessed/<training|validation|test>` directories of your experiment. The most important scores are `total_f1_score`, `total_precision`, and `total_recall`.  The scores are also broken down into arrays of per-image-series scores.

You can run just the scoring component of the pipeline with the command `python pipeline.py score <config file path>`. For the example experiment, this would be `python pipeline.py score ../data/example/main_config.cfg`. 

### Visualization and Manual Thresholding

We received several requests while beta-testing for an easy method of viewing and manually thresholding ROIs by convnet detection confidence. The command `python visualize.py <config file path>` will open a simple GUI for this purpose. 

ConvnetCellDetection is meant to be a completely automatic pipeline, and using the GUI to manually threshold ROIs is **entirely optional**. Most importantly, the GUI provides an easy way to view the results of ConvnetCellDetection without needing to import ROI binary masks into FIJI. 

The GUI overlays convnet-detected ROIs in blue over the original preprocessed image stack.  The slider to the right of the image allows you to add and remove ROIs in order of average probability assigned by the convolutional network to the pixels inside each ROI. The slider beneath the image allows you to move through frames of the image stack. You can move between image stacks with the "Next Image" and "Previous Image" buttons and enter a manual probability threshold value for all images in the "set threshold value:" box.  The "Save Displayed ROIs" button will create a new .npz file in the same directory as the original with a "\_MANUAL" label containing only the displayed ROIs. This file can be read with the code `rois = numpy.load('<name>_MANUAL.npz')['rois']`.  The "Show/Hide Ground Truth Labels" button overlays human labels in red if they exist for the current image. 

## Changing Network Architectures

You can use a different network architecture than the default (2+1)D network as follows:

1. Create (or use an existing) `.znn` file in the `ConvnetCellDetection/celldetection_znn/` directory.  We have provided `.znn` files for the (2+1)D network (`2plus1d.znn`) and the 2D network (`2d.znn`) described in the [NIPS paper](#citing) and for a small one-level network for testing and debugging (`N1.znn`). The [ZNN documentation](http://znn-release.readthedocs.io/en/latest/index.html) describes the `.znn` format for defining a network architecture in detail. 

2. Replace all 3 instances of "2plus1d" in the `main_config.cfg` file for your experiment with the name of the new `.znn` file.

3. Change the `filter_size` parameter in the `main_config.cfg` file for your experiment to the size of the convolutional filters for the new network

4. Set the `is_squashing` parameter in the `main_config.cfg` file for your experiment to 1 if the new network takes 3D input or to 0 if the new network takes 2D input

##  Parameter Descriptions

The `main_config.cfg` configuration file for each experiment contains many parameters that will not need to be changed for general use. However, advanced users may wish to adjust these parameters for particular use cases. The default values are stored in the template `main_config.cfg` in the `src/` directory.  Please [contact us](#contributors) if you have specific questions about these parameters or other aspects of the ConvnetCellDetection pipeline. 

general

- data_dir = location of video files (and ROIs for training) relative to `src/` directory 
- img_width = image width in pixels
- img_height = image height in pixels (note that img_width != img_height has not been well tested) 
- do_downsample = [0 or 1] whether to downsample using mean projection followed by max projection
- do_gridsearch_postprocess_params = [0 or 1] whether to optimize (if data dir is `../data/<experiment>/labeled`) or use optimized postprocessing parameters (if forward pass)

preprocessing

- time_equalize = number of frames to equalize each video to for 3D kernels/projection
- mean_proj_bin = number of consecutive frames to mean project during downsampling
- max_proj_bin = number of consecutive frames to max project during downsampling
- upper_contrast = upper percentile cutoff for preprocessing contrast improvement
- lower_contrast = lower percentile cutoff for preprocessing contrast improvement
- centroid_radius = radius of ROI centroids used for training

network

- net_arch_fpath = absolute path of network architecture `.znn` file
- filter_size = size of one side of square convnet filters (pixels)
- is_squashing = "yes" for (2+1)D or other network that takes 3D input, "no" otherwise.

training

- learning_rate = learning rate of convnet training
- momentum = momentum of convnet training
- max_iter = maximum iterations of convnet training
- num_iter_per_save = number of iterations between each automatic save of convnet during training
- patch_size = patch size for convnet training (pixels,pixels,pixels)
- training_input_dir = location of preprocessed labeled data for training (relative to `src/` directory)
- training_output_dir = location of training convnet output (relative to `src/` directory)
- training_net_prefix = location and prefix for saving trained networks (relative to `src/` directory)

forward

- forward_outsz = patch size for convnet forward pass (pixels,pixels,pixels)
- forward_net = location of saved `.h5` network file to use for forward pass (relative to the `src/` directory) 

docker

- use_docker_machine = [1 or 0]. 1 if the ZNN Docker container needs to run inside a VirtualBox virtual machine (e.g. you are running ConvnetCellDetection on your personal computer. 0 if the pipeline is being executed on a machine that can start Docker containers directly (e.g. an Amazon EC2 instance with the Docke daemon running).
- memory = memory (in MB) to allocate to the docker virtual machine. Changing this will create a new docker virtual machine and take longer on the first training or forward pass with the updated memory value
- machine_name = prefix of docker virtual machine -- the complete name is machine_name + '-' + memory
- container_name = name of docker container with ZNN

postprocessing

- probability_threshold = only pixels with probability values above this threshold are included
- min_size_watershed = connected units with fewer than this many nonzero pixels are removed
- merge_size_watershed = watersheds with fewer than this many pixels are merged
- max_footprint = footprint size of local max operation performed to find watershed centers
- min_size_wand = minimum radius of ROIs labeled by cell magic wand and output by pipeline
- max_size_wand = maximum radius of ROIs labeled by cell magic wand and output by pipeline

postprocessing optimization 

- ranges and step sizes for grid search optimization of postprocessing parameter values above
