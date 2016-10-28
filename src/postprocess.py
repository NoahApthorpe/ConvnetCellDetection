###########################################################
#
# ConvNet Output Postprocessing
#
# Author: Noah Apthorpe
#
# Description: Thresholding, watershedding, and Magic Wand tool
#     postprocessing to convert network output probability 
#     maps to detected ROIs
#
############################################################

import sys
from PIL import Image
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.measure import grid_points_in_poly
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib
import matplotlib.patches as patches
import subprocess
import itertools
import os
import os.path
import cPickle as pickle
import ConfigParser
import tifffile
from load import *
from score import Score
from preprocess import is_labeled, add_pathsep, get_labeled_split, split_labeled_directory
from cell_magic_wand import cell_magic_wand


def read_network_output(directory):
    '''reads tiff output of ZNN into numpy arrays and corresponding filenames'''
    images = []
    filenames = []
    for fname in os.listdir(directory):
        f = directory + fname
        if "output_0.tif" not in f : continue
        im = Image.open(f)
        im = np.array(im, dtype=np.float32)
        images.append(im)
        trunc_fname = fname.rpartition("_output")[0]
        filenames.append(trunc_fname)
    return images, filenames


def read_preprocessed_images(directory, filenames):
    '''reads and correlates preprocessed images
    input to ZNN with filenames read by read_network_output'''
    directory = add_pathsep(directory)
    images = []
    for fname in filenames:
        found = False
        for f in os.listdir(directory):
            if not os.path.isfile(directory + f): continue
            base = os.path.splitext(f)[0]
            if base in fname:
               im = Image.open(directory + f)
               im = np.array(im, dtype=np.float32)
               images.append(im)
               found = True
               break
        if not found:
            raise ValueError("Couldn't find " + fname + " in " + directory)
    return images


def watershed_centroids(labels):
    '''finds centroids of watershed regions'''
    new_markers = np.zeros(labels.shape)
    for label in set(labels.flatten()):
        cx,cy = np.where(labels == label)
        cx,cy = int(cx.mean()), int(cy.mean()) 
        new_markers[cx,cy] = label
    return new_markers


def find_neuron_centers(im, threshold, min_size, merge_size, max_footprint=(7,7)):
    '''finds putative centers of neurons by thresholding and 
    watershedding with a distance transform'''
    t = im > threshold
    t = remove_small_objects(t, min_size)
    c = im.copy()
    c[np.logical_not(t)] = 0
    c = (c-c.min())/(c.max()-c.min())
    d = ndi.distance_transform_edt(c)
    d = (d-d.min())/(d.max()-d.min())
    p = c+d
    local_max = peak_local_max(p, indices=False, footprint=np.ones(max_footprint), labels=t)
    markers = ndi.label(local_max)[0]
    labels = watershed(-p, markers, mask=t)
    markers = watershed_centroids(labels)
    for i in set(labels.flatten()):
        if i == 0: continue
        if np.sum(labels==i) < merge_size:
            markers[markers==i] = 0
    labels = watershed(-p, markers, mask=t)
    markers = watershed_centroids(labels)
    return markers, labels


def markers_to_seeds(markers, border):
    '''converts watershed markers to seed points for magic wand'''
    centers = []
    for m in set(markers.flatten()):
        if m == 0: continue
        x,y = np.where(markers == m)
        x = x[0] + border
        y = y[0] + border
        centers.append((y,x))
    return centers


def postprocessing(preprocess_dir, network_output_dir, postprocess_dir, 
                   border, threshold, min_size_watershed, merge_size_watershed, max_footprint, 
                   min_size_wand, max_size_wand):
    '''Performs postprocessing with argument parameters. Returns ROIs 
    and associated filenames'''
    # convert probability maps into neuron centers
    network_images, filenames = read_network_output(network_output_dir)
    preprocessed_images = read_preprocessed_images(preprocess_dir, filenames)
    markers, labels = zip(*[find_neuron_centers(im, threshold, min_size_watershed, merge_size_watershed, max_footprint=max_footprint) for im in network_images])

    # run magic wand cell edge detection
    all_rois = []
    all_roi_probs = []
    for i in range(len(network_images)):
        print "Running magic wand for " + filenames[i]
        size_diff_0th = preprocessed_images[i].shape[0] - network_images[i].shape[0]
        size_diff_1th = preprocessed_images[i].shape[0] - network_images[i].shape[0]
        padded_network_image = np.pad(network_images[i], ((int(np.floor(size_diff_0th/2.0)), int(np.ceil(size_diff_0th/2.0))),
                                                          (int(np.floor(size_diff_1th/2.0)), int(np.ceil(size_diff_1th/2.0)))), 'constant')
        seeds = markers_to_seeds(markers[i], border)
        rois = []
        roi_probs = []
        for j,c in enumerate(seeds):
            print str(j) + "/" + str(len(seeds)-1),
            roi = cell_magic_wand(preprocessed_images[i], c, min_size_wand, max_size_wand)
            roi_prob = np.sum(np.multiply(roi, padded_network_image))/np.sum(roi)
            rois.append(roi)
            roi_probs.append(roi_prob)
        rois = np.array(rois)
        all_rois.append(rois)
        all_roi_probs.append(roi_probs)
    return all_rois, all_roi_probs, filenames        


def parameter_optimization(data_dir, preprocess_dir, network_output_dir, postprocess_dir,
                           border, min_size_wand, max_size_wand, 
                           ground_truth_directory, params_cfg_fn, cfg_parser):
    '''Performs grid search optimization of postprocessing parameters and stores result in
    new configuration file'''

    # get ground truth ROIs
    ground_truth_rois, filenames = load_data(data_dir, img_width, img_height, rois_only=True)
    
    # get ranges for grid search
    min_threshold = cfg_parser.getfloat('postprocessing optimization', 'min_threshold')
    max_threshold = cfg_parser.getfloat('postprocessing optimization', 'max_threshold')
    steps_threshold = cfg_parser.getint('postprocessing optimization', 'steps_threshold')
    min_minsize = cfg_parser.getfloat('postprocessing optimization', 'min_minsize')
    max_minsize = cfg_parser.getfloat('postprocessing optimization', 'max_minsize')
    steps_minsize = cfg_parser.getint('postprocessing optimization', 'steps_minsize')
    min_footprint = cfg_parser.getfloat('postprocessing optimization', 'min_footprint')
    max_footprint = cfg_parser.getfloat('postprocessing optimization', 'max_footprint')
    steps_footprint = cfg_parser.getint('postprocessing optimization', 'steps_footprint')
    steps_wand = cfg_parser.getint('postprocessing optimization', 'steps_wand')

    # make ranges for grid search
    threshold_range = np.linspace(min_threshold, max_threshold, steps_threshold)
    min_size_watershed_range = np.linspace(min_minsize, max_minsize, steps_minsize)
    max_footprint_range = np.linspace(min_footprint, max_footprint, steps_footprint)
    if steps_wand < 2:
        max_size_wand_range = np.array([max_size_wand])
    else:
        max_size_wand_range = np.linspace(min_size_wand+1, max_size_wand, steps_wand)

    # run grid search and save scores
    scores_params = []
    for threshold, min_size_watershed, max_footprint, max_size_wand in itertools.product(threshold_range, min_size_watershed_range, max_footprint_range, max_size_wand):
        merge_size_watershed = min_size_watershed
        print "Testing threshold: " + str(threshold) + " min_size_watershed: " + str(min_size_watershed) + " max_footprint: " + str(max_footprint) + " max_size_wand: " + str(max_size_wand)
        rois, roi_probs, filenames = postprocessing(preprocess_directory, network_output_directory,
                                                    postprocess_directory, border, threshold, 
                                                    min_size_watershed, merge_size_watershed,
                                                    (max_footprint,max_footprint),
                                                    min_size_wand, max_size_wand)
        s = Score(ground_truth_rois, rois)
        scores_params.append((s.total_f1_score, {'probability_threshold':threshold,
                                                 'min_size_watershed':min_size_watershed, 
                                                 'merge_size_watershed':merge_size_watershed,
                                                 'max_footprint':(max_footprint, max_footprint),
                                                 'max_size_wand':max_size_wand}))

    # find best score and save corresponding parameters
    scores_params.sort()
    best_params = scores_params[-1][1]
    print "Best F1 score: " + str(scores_params[-1][0])
    params_cfg_parser = ConfigParser.SafeConfigParser()
    params_cfg_parser.add_section("postprocessing")
    for param in best_params:
        params_cfg_parser.set("postprocessing", param, str(best_params[param]))
    params_cfg_parser.write(open(params_cfg_fn,'w'));


def main(main_config_fpath='../data/example/main_config.cfg'):
    '''Get user-specified information from main_config.cfg'''
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath,'r'))
    
    # get directory paths
    data_dir = add_pathsep(cfg_parser.get('general', 'data_dir'))
    parent_dir = add_pathsep(os.path.dirname(data_dir[0:-1]))
    downsample_dir = data_dir[0:-1] + "_downsampled" + os.sep
    preprocess_dir = data_dir[0:-1] + "_preprocessed" + os.sep
    network_output_dir = data_dir[0:-1] + "_training_output" + os.sep
    postprocess_dir = data_dir[0:-1] + "_postprocessed" + os.sep
    ttv_list = ['training' + os.sep, 'validation' + os.sep, 'test' + os.sep]
    
    # ensure directories exist
    if not os.path.isdir(data_dir):
        sys.exit("Specified data directory " + data_dir + " does not exist.")
    for ttv in ttv_list if is_labeled(data_dir) else ['']:
        if not os.path.isdir(postprocess_dir + ttv):
            os.makedirs(postprocess_dir + ttv)

    # split training ouput directory if necessary
    if is_labeled(data_dir):
        split_dict = get_labeled_split(data_dir)
        split_labeled_directory(split_dict, network_output_dir, False, False)
 
    # get non-optimized postprocessing parameters
    img_width = cfg_parser.getint('general','img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    field_of_view = cfg_parser.getint('network', 'field_of_view')
    border = field_of_view/2
    do_gridsearch_postprocess_params = cfg_parser.getboolean('general', 'do_gridsearch_postprocess_params')
    min_size_wand = cfg_parser.getfloat('postprocessing', 'min_size_wand')
    max_size_wand = cfg_parser.getfloat('postprocessing', 'max_size_wand')
    
    # locate optimized postprocessing parameters or run grid search optimization
    params_cfg_parser = ConfigParser.SafeConfigParser()
    opt_params_cfg_fn = parent_dir + "optimized_postprocess_params.cfg"
    if do_gridsearch_postprocess_params and os.path.isfile(opt_params_cfg_fn):
        params_cfg_parser.readfp(open(opt_params_cfg_fn, 'r'))
    elif (do_gridsearch_postprocess_params and
          not os.path.isfile(opt_params_cfg_fn) and is_labeled(data_dir)):
        parameter_optimization(preprocess_dir + ttv_list[1], network_output_dir + ttv_list[1],
                               postprocess_dir + ttv_list[1], border, min_size_wand,
                               max_size_wand, opt_params_cfg_fn, cfg_parser)
    else:
        params_cfg_parser = cfg_parser
         
    # read postprocessing-specific parameters
    threshold = params_cfg_parser.getfloat('postprocessing', 'probability_threshold')
    min_size_watershed = params_cfg_parser.getfloat('postprocessing', 'min_size_watershed')
    merge_size_watershed = params_cfg_parser.getfloat('postprocessing', 'merge_size_watershed')
    max_footprint_str = params_cfg_parser.get('postprocessing', 'max_footprint')
    max_footprint = tuple([int(c) for c in max_footprint_str.strip().strip(')').strip('(').split(",")])
    max_size_wand = params_cfg_parser.getfloat('postprocessing', 'max_size_wand')
    assert(len(max_footprint) == 2)

    # run postprocessing
    for ttv in ttv_list if is_labeled(data_dir) else ['']:
        final_rois, final_roi_probs, filenames = postprocessing(preprocess_dir + ttv, network_output_dir + ttv, 
                                               postprocess_dir + ttv, border, threshold, 
                                               min_size_watershed, merge_size_watershed,
                                               max_footprint, min_size_wand, max_size_wand)
    
        # Save final ROIs
        for i,roi in enumerate(final_rois):
            r = roi.max(axis=0)
            roi_name = postprocess_dir + ttv + filenames[i] + '.tif'
            tifffile.imsave(roi_name, r.astype(np.float32))
            np.savez_compressed(postprocess_dir + ttv + filenames[i] + '.npz', rois=roi, roi_probabilities=final_roi_probs[i])

            
if __name__ == "__main__":
    main()
