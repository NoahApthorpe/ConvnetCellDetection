###################################################
#
# Video and ROI preprocessing functions
#
# Authors: Noah Apthorpe and Alex Riordan
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
        roi = roi.max(axis=0)
        stk_name = directory + file_names[i] + ".tif"
        roi_name = directory + file_names[i] + "_ROI.tif"
        tifffile.imsave(stk_name, stk.squeeze())
        tifffile.imsave(roi_name, roi)

'''
 is_labeled()
 Checks if directory is named "labeled"
'''
def is_labeled(dir_path):
    if dir_path.split('/')[-1] == 'labeled':
        return True
    return False 

'''
 get_labeled_split_paths()
 Builds and returns a dict of the train/test/val split given in split_dir
 Keys of dict are fnames, values are one of ['training', 'test', 'val']
'''
def get_labeled_split(already_split_dir):
    if already_split_dir[-1] != '/':
            already_split_dir += '/'
    split_dict = dict()
    subdir_list = ['training', 'test', 'validation']
    for subdir in subdir_list:
        files = os.listdir(already_split_dir + subdir) # get files in subdir
        remove_ds_store(files)
        split_dict.update(dict.fromkeys(files,subdir)) # add each file as key in split_dict, with value subdir name
    return split_dict 

'''
 impose_split_on_labeled_directory()
 Moves files from dir_to_split into newly created train, test, and validation subdirectories in dir_to_split
 Files go to a particular subdirectory based on split_dict dictionary
 is_ROI_tif should be true when ROI labels in directory are tif files, not zip files
'''
def split_labeled_directory(split_dict, dir_to_split,is_ROI_tif, is_post_process):
    if dir_to_split[-1] != '/':
        dir_to_split += '/'
    for subdir in split_dict.itervalues():
        subdir_path = dir_to_split + subdir 
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    for fname, subdir in split_dict.items():
        if is_post_process and ".zip" in fname:
            continue
        if is_ROI_tif:
            fname = fname.replace(".zip","_ROI.tif")
        try :
            os.rename(dir_to_split + fname, dir_to_split + subdir + '/' + fname) # move fname into new subdir
        except AssertionError:
            print fname, ' was not found in ', dir_to_split, ' while attempting to maintain training/test/validation split'
    
def put_labeled_at_end_of_path_if_not_there(fpath) : 
    if fpath[-1] == '/':
        fpath = fpath[:-1]
    if not is_labeled(fpath):
        return fpath + '/labeled'
    else:
        return fpath

def remove_ds_store(file_list):
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass



def main(main_config_fpath = '../main_config_ar.cfg'):
    '''Get user-specified information from main_config.cfg'''
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open('../main_config_ar.cfg', 'r'))
        
    preprocess_directory = cfg_parser.get('general', 'preprocess_dir')
    downsample_directory = cfg_parser.get('general', 'downsample_dir')
    data_directory = cfg_parser.get('general', 'data_dir')
    img_width = cfg_parser.getint('general', 'img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    do_downsample = cfg_parser.getboolean('general', 'do_downsample')
    upper_contrast = cfg_parser.getfloat('preprocessing', 'upper_contrast')
    lower_contrast = cfg_parser.getfloat('preprocessing', 'lower_contrast')
    centroid_radius = cfg_parser.getint('preprocessing', 'centroid_radius')
    
    '''add '/labeled' to preprocess_dir in main config file if data_dir ends with '/labeled' '''
    if is_labeled(data_directory) and not is_labeled(preprocess_directory):
        preprocess_directory = put_labeled_at_end_of_path_if_not_there(preprocess_directory)
        cfg_parser.set('general','preprocess_dir',preprocess_directory)
        with open('../main_config_ar.cfg', 'wb') as configfile:
            cfg_parser.write(configfile)
    
    if not os.path.isdir(preprocess_directory):
        os.makedirs(preprocess_directory)
    
    '''Run actual preprocessing'''
    data, file_names = load_data(downsample_directory, img_width, img_height)
    data = improve_contrast(data, upper_contrast, lower_contrast)
    data = get_centroids(data, centroid_radius, img_width, img_height)
    save_tifs(data, file_names, preprocess_directory)
    
    '''Impose training/test/validation split on preprocess_dir and downsample_dir'''
    if is_labeled(data_directory) :
        split_dict = get_labeled_split(data_directory)
        split_labeled_directory(split_dict, preprocess_directory, True, False)
        if do_downsample :
            split_labeled_directory(split_dict, downsample_directory, False, False)
            
if __name__ == "__main__":
    main()