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
    new_markers = np.zeros(labels.shape)
    for label in set(labels.flatten()):
        cx,cy = np.where(labels == label)
        cx,cy = int(cx.mean()), int(cy.mean()) 
        new_markers[cx,cy] = label
    return new_markers


def find_neuron_centers(im, threshold, min_size, merge_size, max_footprint=(7,7)):
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

'''
def markers_to_param_file(markers, min_size, max_size, fname, border, upscale_factor=1):
    text = "minDiameter {}\nmaxDiameter {}\nroughness 1.0\nimageType brightCells\n".format(min_size, max_size)
    for m in set(markers.flatten()):
        if m == 0: continue
        x,y = np.where(markers == m)
        x = (x[0]*upscale_factor)+border
        y = ((y[0]*upscale_factor)+border)
        text += "seed {} {}\n".format(y,x)
    f = open(fname,'w')
    f.write(text)
    f.close()
'''

def markers_to_seeds(markers, border):
    centers = []
    for m in set(markers.flatten()):
        if m == 0: continue
        x,y = np.where(markers == m)
        x = x[0] + border
        y = y[0] + border
        centers.append((y,x))
    return centers

'''        
def sort_clockwise(edges):
    x = np.array([e[0] for e in edges])
    y = np.array([e[1] for e in edges])
    cx = np.mean(x)
    cy = np.mean(y)
    a = np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = list(x[order])
    y = list(y[order])
    x.append(x[0])
    y.append(y[0])
    return zip(np.array(x),np.array(y))


def edge_file_to_rois(fname, size=512):
    f = open(fname, 'r')
    rois = []
    for ln,line in enumerate(f):
        nroi = np.zeros((size,size))
        points = [s.strip('()\n') for s in line.split('),(')]
        edges = np.zeros((len(points),2))
        for i,p in enumerate(points):
            x,y = p.split(',')
            edges[i,0] = int(x)
            edges[i,1] = int(y)
        edges = sort_clockwise(edges)
        mask = grid_points_in_poly((size,size), edges).astype(np.float32)
        nm = np.zeros(mask.shape)
        for x,y in itertools.product(range(size), range(size)):
            nm[x,y] = mask[y,x]
        rois.append(nm)
    return np.array(rois)
'''

def postprocessing(preprocess_dir, network_output_dir, postprocess_dir, 
                   border, threshold, min_size_watershed, merge_size_watershed, max_footprint, 
                   min_size_wand, max_size_wand):

    '''
    # create directory for magic wand parameters and edge output
    magicwand_directory = postprocess_directory + "magicwand" + os.path.sep
    if not os.path.isdir(magicwand_directory):
        os.mkdir(magicwand_directory)
    '''

    # convert probability maps into neuron centers
    network_images, filenames = read_network_output(network_output_dir)
    preprocessed_images = read_preprocessed_images(preprocess_dir, filenames)

    markers, labels = zip(*[find_neuron_centers(im, threshold, min_size_watershed, merge_size_watershed, max_footprint=max_footprint) for im in network_images])

    all_rois = []
    all_roi_probs = []
    for i in range(len(network_images)):
        print i
        print network_images[i].shape
        print preprocessed_images[i].shape
        size_diff_0th = preprocessed_images[i].shape[0] - network_images[i].shape[0]
        size_diff_1th = preprocessed_images[i].shape[0] - network_images[i].shape[0]
        padded_network_image = np.pad(network_images[i], ((int(np.floor(size_diff_0th/2.0)), int(np.ceil(size_diff_0th/2.0))),
                                                          (int(np.floor(size_diff_1th/2.0)), int(np.ceil(size_diff_1th/2.0)))), 'constant')
        seeds = markers_to_seeds(markers[i], border)
        rois = []
        roi_probs = []
        for j,c in enumerate(seeds):
            print "   " + str(c)
            if j > 15: break
            roi = cell_magic_wand(preprocessed_images[i], c, min_size_wand, max_size_wand)
            roi_prob = np.sum(np.multiply(roi, padded_network_image))
            rois.append(roi)
            roi_probs.append(roi_prob)
        rois = np.array(rois)
        all_rois.append(rois)
        all_roi_probs.append(roi_probs)
    print filenames
    return all_rois, all_roi_probs, filenames        
        
    
    '''
    # write parameter files for magic wand tool    
    for i in range(len(images)):
        magicwand_params_fn = magicwand_directory + filenames[i]+".txt"
        markers_to_param_file(markers[i], min_size_wand, max_size_wand, 
                              magicwand_params_fn, border)

    # run magic wand tool
    # magicwand_classpath = "./src/cellMagicWand"
    magicwand_classpath = os.path.dirname(os.path.abspath(__file__)) + '/cellMagicWand'
    magicwand_compile = " ".join(["javac -cp", magicwand_classpath, "MagicWand.java"])
    magicwand_command = " ".join(["java -cp", magicwand_classpath, "MagicWand", 
                                  preprocess_directory, magicwand_directory])
    process = subprocess.Popen(magicwand_command, shell=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print stderr
        
    # convert magic wand output into ROIs
    rois = []
    filenames = []    
    for fn in os.listdir(magicwand_directory):
        if os.path.basename(fn).find('edges') != -1:
            rois.append(edge_file_to_rois(magicwand_directory + fn))
            filenames.append(fn.partition('_edges')[0])
    
    return rois, filenames
    '''

def parameter_optimization(data_dir, preprocess_dir, network_output_dir, postprocess_dir,
                           border, min_size_wand, max_size_wand, 
                           ground_truth_directory, params_cfg_fn):
    data, filenames = load_data(data_dir, img_width, img_height)
    ground_truth_rois = [r for (s,r) in data]
    threshold_range = np.linspace(0.7,0.9,1)
    min_size_watershed_range = np.linspace(15,30,1)
    max_footprint_range = np.linspace(7,7,1)
    max_size_wand_range = np.linspace(min_size_wand+1, max_size_wand, 1)
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
    scores_params.sort()
    best_params = scores_params[-1][1]
    print "Best F1 score: " + str(scores_params[-1][0])
    params_cfg_parser = ConfigParser.SafeConfigParser()
    params_cfg_parser.add_section("postprocessing")
    for param in best_params:
        params_cfg_parser.set("postprocessing", param, str(best_params[param]))
    params_cfg_parser.write(open(params_cfg_fn,'w'));
    return


def main(main_config_fpath='../main_config_ar.cfg'):
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
                               max_size_wand, opt_params_cfg_fn)
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
            
    '''
    # Impose test/train/validation split on postprocess directory if applicable 
    if preprocess.is_labeled(data_directory) :
        split_dict = preprocess.get_labeled_split(data_directory)
        preprocess.split_labeled_directory(split_dict, postprocess_directory, False, True)
    '''

if __name__ == "__main__":
    main()
