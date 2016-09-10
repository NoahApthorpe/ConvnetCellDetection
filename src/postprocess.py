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
import preprocess

def add_path_sep(directory):
    if directory[-1] == os.path.sep:
        return directory 
    else:
        return directory + os.path.sep


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


def postprocessing(preprocess_directory, network_output_directory, postprocess_directory, 
                   border, threshold, min_size_watershed, merge_size_watershed, max_footprint, 
                   min_size_wand, max_size_wand):
    
    # create directory for magic wand parameters and edge output
    magicwand_directory = postprocess_directory + "magicwand" + os.path.sep
    if not os.path.isdir(magicwand_directory):
        os.mkdir(magicwand_directory)

    # convert probability maps into neuron centers
    images, filenames = read_network_output(network_output_directory)
    markers, labels = zip(*[find_neuron_centers(im, threshold, min_size_watershed, 
                                                merge_size_watershed, 
                                                max_footprint=max_footprint) for im in images])
    
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
    

def parameter_optimization(preprocess_directory, network_output_directory, postprocess_directory,
                           border, min_size_wand, max_size_wand, 
                           ground_truth_directory, params_cfg):
    data, filenames = load_data(ground_truth_directory, img_width, img_height)
    ground_truth_rois = [r for (s,r) in data]
    threshold_range = np.linspace(0.7,0.9,1)
    min_size_watershed_range = np.linspace(15,30,1)
    max_footprint_range = np.linspace (7,7,1)
    scores_params = []
    for threshold, min_size_watershed, max_footprint in itertools.product(threshold_range, min_size_watershed_range, max_footprint_range):
        merge_size_watershed = min_size_watershed
        print threshold, min_size_watershed, max_footprint
        rois, filenames = postprocessing(preprocess_directory, network_output_directory,
                                         postprocess_directory, border, threshold, 
                                         min_size_watershed, merge_size_watershed,
                                         (max_footprint,max_footprint),
                                         min_size_wand, max_size_wand)
        s = Score(ground_truth_rois, rois)
        scores_params.append((s.total_f1_score, {'probability_threshold':threshold,
                                                 'min_size_watershed':min_size_watershed, 
                                                 'merge_size_watershed':merge_size_watershed,
                                                 'max_footprint':(max_footprint, max_footprint)}))
    scores_params.sort()
    best_params = scores_params[-1][1]
    print scores_params[-1][0]
    params_cfg_parser = ConfigParser.SafeConfigParser()
    params_cfg_parser.add_section("postprocessing")
    for param in best_params:
        params_cfg_parser.set("postprocessing", param, str(best_params[param]))
    params_cfg_parser.write(open(params_cfg,'w'));
    return


def main(main_config_fpath='../main_config_ar.cfg'):
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath,'r'))

    # read parameters
    data_directory = cfg_parser.get('general', 'data_dir')
    downsample_directory = add_path_sep(cfg_parser.get('general', 'downsample_dir'))
    preprocess_directory = add_path_sep(cfg_parser.get('general', 'preprocess_dir'))
    network_output_directory = add_path_sep(cfg_parser.get('forward', 'forward_output_dir')) #I updated this -AR 9/5/16
    postprocess_directory = add_path_sep(cfg_parser.get('general', 'postprocess_dir'))
    img_width = cfg_parser.getint('general','img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    field_of_view = cfg_parser.getint('network', 'field_of_view')
    border = field_of_view/2
    do_gridsearch_postprocess_params = cfg_parser.getboolean('general', 'do_gridsearch_postprocess_params')
    min_size_wand = cfg_parser.getfloat('postprocessing', 'min_size_wand')
    max_size_wand = cfg_parser.getfloat('postprocessing', 'max_size_wand')
    if not os.path.isdir(postprocess_directory):
        os.mkdir(postprocess_directory)
    else:
        raise Exception("Please delete existing postprocessing directory ", postprocess_directory)
    
    # locate postprocessing parameters or run grid search optimization
    params_cfg_parser = ConfigParser.SafeConfigParser()
    if do_gridsearch_postprocess_params:
        params_cfg = postprocess_directory+"optimized_postprocess_params.cfg"
        if not os.path.isfile(params_cfg):
            parameter_optimization(preprocess_directory, network_output_directory, 
                                   postprocess_directory, border, min_size_wand, max_size_wand,
                                   downsample_directory, params_cfg)
    else:
        params_cfg = main_config_fpath
    params_cfg_parser.readfp(open(params_cfg,'r'))
    
    # read postprocessing-specific parameters
    threshold = params_cfg_parser.getfloat('postprocessing', 'probability_threshold')
    min_size_watershed = params_cfg_parser.getfloat('postprocessing', 'min_size_watershed')
    merge_size_watershed = params_cfg_parser.getfloat('postprocessing', 'merge_size_watershed')
    max_footprint_str = params_cfg_parser.get('postprocessing', 'max_footprint')
    max_footprint = tuple([float(c) for c in max_footprint_str.strip().strip(')').strip('(').split(",")])
    assert(len(max_footprint) == 2)
    
    # run postprocessing
    final_rois, filenames = postprocessing(preprocess_directory, network_output_directory, 
                                postprocess_directory, border, threshold, 
                                min_size_watershed, merge_size_watershed, max_footprint,
                                min_size_wand, max_size_wand)
    
    # Save final ROIs
    for i,roi in enumerate(final_rois):
        print "Saving postprocessed version of ", filenames[i]
        r = roi.max(axis=0)
        roi_name = postprocess_directory + filenames[i] + '.tif'
        tifffile.imsave(roi_name, r.astype(np.float32))
        np.savez_compressed(postprocess_directory + filenames[i] + '.npz', roi)
    
    # Impose test/train/validation split on postprocess directory if applicable 
    if preprocess.is_labeled(data_directory) :
        split_dict = preprocess.get_labeled_split(data_directory)
        preprocess.split_labeled_directory(split_dict, postprocess_directory, False, True)

if __name__ == "__main__":
    main()
