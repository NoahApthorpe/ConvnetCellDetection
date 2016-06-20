###################################################
#
# Video and ROI preprocessing functions
#
# Author: Noah Apthorpe
#
# Description: Video contrast improvement
#   and ROI centroid conversion
# 
# Usage: pass data from load_data in load.py
#   consecutively to the functions in this
#   file to produce .tif files ready for ZNN 
#
###################################################

import numpy as np
from PIL import Image
import os.path

def improve_contrast(data):
    new_data = []
    for i, (stk,roi) in enumerate(data):
        low_p = np.percentile(stk.flatten(), 3)
        high_p = np.percentile(stk.flatten(), 99)
        new_stk = np.clip(stk, low_p, high_p)
        new_stk = new_stk - new_stk.mean()
        new_stk = np.divide(new_stk-np.min(new_stk), np.max(new_stk) - np.min(new_stk))
        new_data.append((new_stk, roi))
    return new_data

def get_centroids(data, radius): 
    new_data = []
    for j,(stk, rois) in enumerate(data):
        new_rois = np.zeros(rois.shape)                                                             
        for i,r in enumerate(rois):  
            cx,cy = np.where(r!=0)                                                  
            cx,cy = int(cx.mean()), int(cy.mean())  
            x,y = np.ogrid[0:512, 0:512]
            index = (x-cx)**2 + (y-cy)**2 <= radius**2
            new_rois[i, index] = 1 
        new_data.append((stk,new_rois))
    return new_data

def save_tifs(data, directory):
    if directory[-1] != os.path.sep:
        directory += os.path.sep
    for i,(stk,roi) in enumerate(data):
        stk_name = directory + "stk{}".format(i+1) + ".tif"
        roi_name = directory + "roi{}".format(i+1) + ".tif"
        im_stk = Image.fromarray(stk.squeeze())
        im_roi = Image.fromarray(roi)
        im_stk.save(stk_name)
        im_roi.save(roi_name)

