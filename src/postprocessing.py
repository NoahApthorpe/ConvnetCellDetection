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
from load import *
from test import Score


def read_network_output(d):
    images = []
    for fname in os.listdir(d):
        f = d + fname
        if not f.endswith(".tif"): continue
        if not int((f.partition("output_")[2]).partition(".tif")[0]) == 0: continue
        sample_num = int((f.partition("sample")[2]).partition("_")[0])
        im = Image.open(f)
        im = np.array(im, dtype=np.float32)
        images.append((sample_num, im))
    sample_nums, images = zip(*sorted(images))
    return images, sample_nums


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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xbound(lower=0,upper=512)
    ax.set_ybound(lower=0,upper=512)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    for ln,line in enumerate(f):
        #print ln
        nroi = np.zeros((size,size))
        points = [s.strip('()\n') for s in line.split('),(')]
        edges = np.zeros((len(points),2))
        for i,p in enumerate(points):
            x,y = p.split(',')
            edges[i,0] = int(x)
            edges[i,1] = int(y)
        edges = sort_clockwise(edges)
        path = Path(edges)  
        patch = patches.PathPatch(path, facecolor='none', lw=0.6)
        ax.add_patch(patch)
        mask = grid_points_in_poly((size,size), edges).astype(np.float32)
        nm = np.zeros(mask.shape)
        for x,y in itertools.product(range(size), range(size)):
            nm[x,y] = mask[y,x]
        rois.append(nm) #np.flipud(np.rot90(mask.astype(np.float32))))# np.flipud(np.rot90(nroi))) #mask.astype(np.float32))

    #plt.axis('off')    
    plt.savefig("../../ZZZ.png",bbox_inches='tight', dpi=200, pad_inches=0)
    plt.show()
    return np.array(rois)


def postprocessing(directory, threshold, min_size_watershed, merge_size_watershed, min_size_wand, max_size_wand, border):
    if directory[-1] != os.path.sep:
        directory += os.path.sep
    params_directory = directory + "params" + os.path.sep
    if not os.path.isdir(params_directory):
        os.mkdir(params_directory)

    full_sp, sn = read_network_output(directory)
    markers, labels = zip(*[find_neuron_centers(d, threshold, min_size_watershed, merge_size_watershed) for d in full_sp])
    for i in range(len(sn)):
        markers_to_param_file(markers[i], min_size_wand, max_size_wand, params_directory+"params"+str(sn[i])+".txt", border)

    process = subprocess.Popen("java -cp /Users/noahapthorpe/Documents/NeuronSegmentation/deepmau5/src/cellMagicWand MagicWand /Users/noahapthorpe/Documents/NeuronSegmentation/znn/all_jan_data/full_jan_sp/ 29 /Users/noahapthorpe/Documents/NeuronSegmentation/znn/full_jan_sp/fwd/params/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print stderr
    
    for fn in os.listdir(params_directory):
        rois = []
        if os.path.basename(fn).find('edges') != -1:
            rois.append(edge_file_to_rois(params_directory + fn))

    return rois


def parameter_optimization():
    pass


"""
def postprocessing(net_out_directory, data_directory):
    if net_out_directory[-1] != os.path.sep:
        net_out_directory += os.path.sep
    if data_directory[-1] != os.path.sep:
        data_directory += os.path.sep

    full_sp, sn = read_network_output(net_out_directory)
    data = load_data(data_directory)

    min_size_wand = 10
    max_size_wand = 22
    border = 18
    threshold_range = np.linspace(0.80, 0.92,5)#30
    min_size_watershed_range = np.linspace(20,60,5)#15

    for t,m in itertools.product(threshold_range, min_size_watershed_range):
        param_values.append((t,m))
        print t,m
        markers, labels = zip(*[find_neuron_centers(d, t, m, m) for d in full_sp])
        for i in range(len(sn)):
            markers_to_param_file(markers[i], min_size_wand, max_size_wand, directory+"params/params"+str(sn[i])+".txt", border)
        process = subprocess.Popen("java -cp /Users/noahapthorpe/Documents/NeuronSegmentation/deepmau5/src/cellMagicWand MagicWand /Users/noahapthorpe/Documents/NeuronSegmentation/znn/all_data/full_sp/ 29 /Users/noahapthorpe/Documents/NeuronSegmentation/Alex-nds/jan-temporal/params/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        test_rois = [edge_file_to_rois(directory + "params/edges" + str(i) + ".txt") for i in [2,7,16,25,28]]
        val_rois = [edge_file_to_rois(directory + "params/edges" + str(i) + ".txt") for i in [1,3,19,24,26]]
    



    test_data = [data[i] for i in [1,6,15,24,27]]
    val_data = [data[i] for i in [0,2,18,23,25]]


val_f_scores = []
test_f_scores = []
val_p_scores = []
test_p_scores = []
val_r_scores = []
test_r_scores = []
param_values = []

    test_s = Score(None, None, [make_it_fair(d[1],border) for d in test_data], test_rois)
    val_s = Score(None, None, [make_it_fair(d[1],border) for d in val_data], val_rois)
    val_f_scores.append(val_s.f1_scores)
    test_f_scores.append(test_s.f1_scores)
    val_p_scores.append(val_s.precisions)
    test_p_scores.append(test_s.precisions)
    val_r_scores.append(val_s.recalls)
    test_r_scores.append(test_s.recalls)
pickle.dump(param_values, open("jan_param_values.pickle", 'w'))
pickle.dump(val_f_scores, open("jan_val_f_scores.pickle", 'w'))
pickle.dump(test_f_scores, open("jan_test_f_scores.pickle", 'w'))
pickle.dump(val_p_scores, open("jan_val_p_scores.pickle", 'w'))
pickle.dump(test_p_scores, open("jan_test_p_scores.pickle", 'w'))
pickle.dump(val_r_scores, open("jan_val_r_scores.pickle", 'w'))
pickle.dump(test_r_scores, open("jan_test_r_scores.pickle", 'w'))
"""
