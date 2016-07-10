###################################################
#
# Video and ROI preprocessing functions
#
# Author: Noah Apthorpe
#
# Description: Video contrast improvement
#   and ROI centroid conversion
# 
# Usage: run file as executable to 
#   pass data and file_names from load_data 
#   in load.py  consecutively to preprocessing
#   functions to produce .tif files ready for ZNN 
#
###################################################

import numpy as np
import skimage.io
import os.path
from load import load_data
import tifffile
from PIL import Image
import ConfigParser

def improve_contrast(data, upper_contrast, lower_contrast):
    new_data = []
    for i, (stk,roi) in enumerate(data):
        low_p = np.percentile(stk.flatten(), lower_contrast)
        high_p = np.percentile(stk.flatten(), upper_contrast)
        new_stk = np.clip(stk, low_p, high_p)
        new_stk = new_stk - new_stk.mean()
        new_stk = np.divide(new_stk-np.min(new_stk), np.max(new_stk) - np.min(new_stk))
        new_data.append((new_stk, roi))
    return new_data

def get_centroids(data, radius, img_width, img_height): 
    new_data = []
    for j,(stk, rois) in enumerate(data):
        new_rois = np.zeros(rois.shape, dtype='float32') # MAYBE NOT NECESSARY TO CAST
        for i,r in enumerate(rois):  
            cx,cy = np.where(r!=0)                                                  
            cx,cy = int(cx.mean()), int(cy.mean())  
            x,y = np.ogrid[0:img_width, 0:img_height]
            index = (x-cx)**2 + (y-cy)**2 <= radius**2
            new_rois[i, index] = 1 
        new_data.append((stk,new_rois))
    return new_data

def save_tifs(data, file_names, directory):
    if directory[-1] != os.path.sep:
        directory += os.path.sep
    for i,(stk,roi) in enumerate(data):
        stk = np.mean(stk, 0)
        print stk.shape
        print roi.dtype
        skimage.io.imshow(stk)
        stk_name = directory + file_names[i] + ".tif"
        roi_name = directory + file_names[i] + "_ROI.tif"
        #tifffile.imsave(stk_name, stk.squeeze())
        #tifffile.imsave(roi_name, roi)
        skimage.io.imsave(stk_name, stk.squeeze())
        skimage.io.imsave(roi_name, roi)
        #im_stk = Image.fromarray(stk.squeeze())
        #im_roi = Image.fromarray(roi)
        #im_stk.save(stk_name)
        #im_roi.save(roi_name)


if __name__ == "__main__":
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open('./main_config.cfg', 'r'))
        
    preprocess_directory = cfg_parser.get('general', 'preprocess_dir')
    downsample_directory = cfg_parser.get('general', 'downsample_dir')
    img_width = cfg_parser.getint('general', 'img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    upper_contrast = cfg_parser.getfloat('preprocessing', 'upper_contrast')
    lower_contrast = cfg_parser.getfloat('preprocessing', 'lower_contrast')
    centroid_radius = cfg_parser.getint('preprocessing', 'centroid_radius')

    if not os.path.isdir(preprocess_directory):
        os.mkdir(preprocess_directory)

    data, file_names = load_data(downsample_directory, img_width, img_height)
    data = improve_contrast(data, upper_contrast, lower_contrast)
    data = get_centroids(data, centroid_radius, img_width, img_height)
    save_tifs(data, file_names, preprocess_directory)
