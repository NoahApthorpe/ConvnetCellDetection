###################################################
#
# Video and ROI preprocessing functions
#
# Authors: Noah Apthorpe and Alex Riordan
#
# Description: Video contrast improvement
#   and ROI centroid conversion
#
# Usage: Call main() function with path to
#   configuration file to downsample,
#   time equalize, improve contrast,
#   and find centroids.
#
###################################################

import sys
import shutil
import numpy as np
import skimage.io
import os.path
from load import load_data, load_stack
import tifffile
from PIL import Image
import ConfigParser
from scipy.ndimage import zoom


def improve_contrast(stks, upper_contrast, lower_contrast):
    '''Increase contrast of images by trimming high and low intensity values and re-normalizing'''
    new_data = []
    for i, stk in enumerate(stks):
        low_p = np.percentile(stk.flatten(), lower_contrast)
        high_p = np.percentile(stk.flatten(), upper_contrast)
        new_stk = np.clip(stk, low_p, high_p)
        new_stk = new_stk - new_stk.mean()
        new_stk = np.divide(new_stk-np.min(new_stk), np.max(new_stk) - np.min(new_stk))
        new_data.append(new_stk)
    return new_data


def get_centroids(input_rois, radius, img_width, img_height):
    '''Convert ImageJ ROIs into centroids for improved convnet boundary detection'''
    new_data = []
    for j, rois in enumerate(input_rois):
        new_rois = np.zeros(rois.shape, dtype='float32')
        for i, r in enumerate(rois):
            cx, cy = np.where(r != 0)
            cx, cy = int(cx.mean()), int(cy.mean())
            x, y = np.ogrid[0:img_width, 0:img_height]
            index = (x-cx)**2 + (y-cy)**2 <= radius**2
            new_rois[i, index] = 1
        new_data.append(new_rois)
    return new_data


def save_image_tifs(stks, file_names, directory):
    '''Save image stacks as tif files'''
    directory = add_pathsep(directory)
    for i, stk in enumerate(stks):
        stk_name = directory + file_names[i] + ".tif"
        tifffile.imsave(stk_name, stk.squeeze())
        
        
def save_roi_tifs(rois, file_names, directory):
    '''Save ROIs as tif files'''
    directory = add_pathsep(directory)
    for i, roi in enumerate(rois):
        roi = roi.max(axis=0)
        roi_name = directory + file_names[i] + "_ROI.tif"
        tifffile.imsave(roi_name, roi)


def add_pathsep(directory_name):
    '''Add path seperater to end of directory name if not already there'''
    if directory_name[-1] != os.sep:
        return directory_name + os.sep
    return directory_name


def is_labeled(dir_path):
    '''Returns true if the argument path is to the special 'labeled' directory'''
    return 'labeled' in os.path.basename(os.path.dirname(dir_path))


def downsample_helper(files_list, img_width, img_height, mean_proj_bins, max_proj_bins):
    '''Mean and max project to covert image files in list to single downsampled numpy array'''
    mean_stack = np.zeros((0, img_width, img_height), dtype=np.float32)
    full_stack = np.zeros((0, img_width, img_height), dtype=np.float32)

    # mean projections
    for f in files_list:
        full_stack = np.vstack((full_stack, load_stack(f)))
        for i in range(0, full_stack.shape[0], mean_proj_bins):
            if full_stack.shape[0] - (i + mean_proj_bins) < mean_proj_bins and f == files_list[-1]:
                    m = np.mean(full_stack[i:, :, :], axis=0)
                    m = np.expand_dims(m, axis=0)
                    mean_stack = np.vstack((mean_stack, m))
                    break
            elif full_stack.shape[0] - i < mean_proj_bins:
                full_stack = full_stack[i:, :, :]
            else:
                m = np.mean(full_stack[i:i+mean_proj_bins, :, :], axis=0)
                m = np.expand_dims(m, axis=0)
                mean_stack = np.vstack((mean_stack, m))

    # max projections
    max_stack = np.zeros((0, img_width, img_height), dtype=np.float32)
    for i in range(0, mean_stack.shape[0], max_proj_bins):
        if mean_stack.shape[0] - (i + max_proj_bins) < max_proj_bins:
            m = np.max(mean_stack[i:, :, :], axis=0)
            m = np.expand_dims(m, axis=0)
            max_stack = np.vstack((max_stack, m))
            break
        else:
            m = np.max(mean_stack[i:i+max_proj_bins, :, :], axis=0)
            m = np.expand_dims(m, axis=0)
            max_stack = np.vstack((max_stack, m))
    return max_stack


def downsample(src_dir, dst_dir, img_width, img_height, mean_proj_bins, max_proj_bins):
    '''Downsample image stacks in src_dir and place results and roi .zip files in dst_dir'''
    do_copy = src_dir != dst_dir
    for f in os.listdir(src_dir):
        ext = os.path.splitext(f)[1].lower()

        # skip hidden files
        if ext.strip() == '':
            continue
        
        # copy .zip roi files without modification
        if ext == '.zip' and do_copy:
            shutil.copy(src_dir + f, dst_dir)

        # downsample individual videos
        elif ext == '.tif' or ext == '.tiff':
            result = downsample_helper([src_dir + f], img_width, img_height,
                                       mean_proj_bins, max_proj_bins)
            tifffile.imsave(dst_dir + f, result.squeeze())

        # downsample folders with videos split into smaller time chunks
        elif os.path.isdir(src_dir + f):
            sub_videos = [src_dir + add_pathsep(f) + v for v in os.listdir(src_dir + f)
                          if (os.path.splitext(v)[1].lower() == '.tif' or
                              os.path.splitext(v)[1].lower() == '.tiff')]
            result = downsample_helper(sub_videos, img_width, img_height, mean_proj_bins, max_proj_bins)
            tifffile.imsave(dst_dir + f + '.tif', result.squeeze())


def time_equalize(src_dir, dst_dir, img_width, img_height, new_time_depth):
    '''Make image stacks in src_dir have the same number of frames.  Place results in dst_dir'''
    do_copy = src_dir != dst_dir
    for f in os.listdir(src_dir):
        ext = os.path.splitext(f)[1].lower()

        # copy zip roi files without modification
        if ext == '.zip' and do_copy:
            shutil.copy(src_dir + f, dst_dir)

        # time equalize indvidual videos
        elif ext == '.tif' or ext == '.tiff':
            data = load_stack(src_dir + f)
            resized = zoom(data, (float(new_time_depth)/data.shape[0], 1, 1))
            tifffile.imsave(dst_dir + f, resized.squeeze())


def remove_ds_store(file_list):
    '''Remove OSX .DS_Store file from list'''
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass


def get_labeled_split(already_split_dir):
    '''Builds and returns a dict of the train/test/val split given in already_split_dir
    Keys of dict are fnames, values are one of ['training', 'test', 'val']'''
    already_split_dir = add_pathsep(already_split_dir)
    split_dict = dict()
    subdir_list = ['training', 'test', 'validation']
    for subdir in subdir_list:
        files = os.listdir(already_split_dir + subdir)
        remove_ds_store(files)
        split_dict.update(dict.fromkeys(files, subdir))
    return split_dict


def split_labeled_directory(split_dict, dir_to_split, is_ROI_tif, is_post_process):
    '''Moves files from dir_to_split into newly created train, test,
    and validation subdirectories in dir_to_split based on split_dict dictionary.
    is_ROI_tif should be true when ROI labels in directory are tif files, not zip files'''
    dir_to_split = add_pathsep(dir_to_split)
            
    for subdir in split_dict.itervalues():
        subdir_path = dir_to_split + subdir
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    for fname, subdir in split_dict.items():
        for f in os.listdir(dir_to_split):
            if not os.path.isfile(dir_to_split + f): continue
            base = os.path.splitext(fname)[0]
            if base in f:
                os.rename(dir_to_split + f, dir_to_split + subdir + os.sep + f)
    '''
        if is_post_process and ".zip" in fname:
            continue
        if is_ROI_tif:
            fname = fname.replace(".zip", "_ROI.tif")
        try:
            # move fname into new subdir
            os.rename(dir_to_split + fname, dir_to_split + subdir + '/' + fname)
            if is_post_process:
                fname = fname.split('.')[0] + '.npz'
                os.rename(dir_to_split + fname, dir_to_split + subdir + '/' + fname)
        except AssertionError:
            print fname, ' was not found in ', dir_to_split, ' while attempting to maintain training/test/validation split'
    '''

def put_labeled_at_end_of_path_if_not_there(fpath):
    if fpath[-1] == '/':
        fpath = fpath[:-1]
    if not is_labeled(fpath):
        return fpath + '/labeled'
    else:
        return fpath


def main(main_config_fpath='../data/example/main_config.cfg'):
    '''Get user-specified information from main_config.cfg'''
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath, 'r'))

    # get directory paths
    data_dir = add_pathsep(cfg_parser.get('general', 'data_dir'))
    downsample_dir = data_dir[0:-1] + "_downsampled" + os.sep
    preprocess_dir = data_dir[0:-1] + "_preprocessed" + os.sep
    ttv_list = ['training' + os.sep, 'validation' + os.sep, 'test' + os.sep]

    # ensure directories exist
    if not os.path.isdir(data_dir):
        sys.exit("Specified data directory " + data_dir + " does not exist.")
    for ttv in ttv_list if is_labeled(data_dir) else ['']:
        if not os.path.isdir(downsample_dir + ttv):
            os.makedirs(downsample_dir + ttv)
        if not os.path.isdir(preprocess_dir + ttv):
            os.makedirs(preprocess_dir + ttv)

    # get remaining preprocessing parameters
    img_width = cfg_parser.getint('general', 'img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    mean_proj_bins = cfg_parser.getint('preprocessing', 'mean_proj_bin')
    max_proj_bins = cfg_parser.getint('preprocessing', 'max_proj_bin')
    new_time_depth = cfg_parser.getint('preprocessing', 'time_equalize')
    upper_contrast = cfg_parser.getfloat('preprocessing', 'upper_contrast')
    lower_contrast = cfg_parser.getfloat('preprocessing', 'lower_contrast')
    centroid_radius = cfg_parser.getint('preprocessing', 'centroid_radius')

    # run preprocessing
    for ttv in ttv_list if is_labeled(data_dir) else ['']:
        if cfg_parser.getboolean('general', 'do_downsample'):
            downsample(data_dir + ttv, downsample_dir + ttv,
                       img_width, img_height, mean_proj_bins, max_proj_bins)
            time_equalize(downsample_dir + ttv, downsample_dir + ttv,
                          img_width, img_height, new_time_depth)
        else:
            time_equalize(data_dir + ttv, downsample_dir + ttv,
                          img_width, img_height, new_time_depth)

        if is_labeled(data_dir):    
            stks, rois, file_names = load_data(downsample_dir + ttv, img_width, img_height)
            stks = improve_contrast(stks, upper_contrast, lower_contrast)
            rois = get_centroids(rois, centroid_radius, img_width, img_height)
            save_image_tifs(stks, file_names, preprocess_dir + ttv)
            save_roi_tifs(rois, file_names, preprocess_dir + ttv)
        else:
            stks, file_names = load_data(downsample_dir + ttv, img_width, img_height, no_rois=True)
            stks = improve_contrast(stks, upper_contrast, lower_contrast)
            save_image_tifs(stks, file_names, preprocess_dir + ttv)
            

if __name__ == "__main__":
    main()
