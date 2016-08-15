#!/usr/bin/env python

'''
 Python pipeline for creating ZNN files
 
 Author: Alexander Riordan
      
 Description: script that reads parameters from
    configuration file and creates ZNN configuration,
    network, and dataset.spec files accordingly
'''

import os, sys
import ConfigParser


'''create_dataset_spec
 N.B. this method will cause issues if any non-ROI filenames contain '_ROI.'
 will also cause issues if files aren't .tif 
'''

def create_dataset_spec(fname, preprocess_directory):
    f = open(fname,'w+') 
    
    files = os.listdir(preprocess_directory) # get files in preprocess_dir
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    
    files = [x for x in files if "_ROI." not in x] # remove all filenames with '_ROI.'
    files = [x.replace('.tif','') for x in files] # remove .tif file extension
    
    docker_directory = update_path_for_Docker_mount(preprocess_directory)
    
    s = ''
    #Iterate over remaining list, and use entries for stk and ROI fnames. Write dataset.spec file.
    for i in range(len(files)) :
        s = write_one_section_dataset_spec(s, i+1, '' + docker_directory + '/' + files[i] + '.tif', docker_directory + '/' + files[i] + '_ROI.tif')
    f.write(s)
    f.close()


'''
 write_one_section_dataset_spec
 Helper method for create_dataset_spec
 Given a unique stk filename and its corresponding ROI file, creates one section of a ZNN dataset.spec file
'''
def write_one_section_dataset_spec(s, section_num, stk_path, roi_path):
    s += '[image' + str(section_num) +']\n'
    s += 'fnames = ' + str(stk_path) + '\n'
    s += 'pp_types = standard3D\n'
    s += 'is_auto_crop = yes\n\n'

    s += '[label' + str(section_num) +']\n'
    s += 'fnames = ' + str(roi_path) + '\n'
    s += 'pp_types = auto\n'
    s += 'is_auto_crop = yes\n\n'
    
    s += '[sample' + str(section_num) +']\n'
    s += 'input = ' + str(section_num) + '\n'
    s += 'output = ' + str(section_num) + '\n\n'
    
    return s

'''
 update_path_for_Docker_mount
 Helper method for create_dataset_spec
 Rewrites user-provided path to point to mounted directory in Docker container
 Assumes ConvnetCellDetection directory is mounted in znn-release directory in Docker container
 N.B. Assumes that no subfolders of ConvnetCellDetection are named 'ConvnetCellDetection'
 Assumes that user is using unix-like path conventions, i.e. forward slash and NOT backslash
 Assumes that user provides absolute paths in main config
'''
def update_path_for_Docker_mount(user_path):
    s = user_path.split('ConvnetCellDetection')
    new_path = '../ConvnetCellDetection' + s[1]
    return new_path

'''
'''
def create_config_file():
    pass

if __name__ == "__main__":
    '''Get user-specified information from main_config.cfg'''
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open('../main_config_ar.cfg', 'r'))
    preprocess_directory = cfg_parser.get('general', 'preprocess_dir')
    img_width = cfg_parser.getint('general', 'img_width')
    img_height = cfg_parser.getint('general', 'img_height')
    
    '''Create znn files'''
    dataspec_path = '../celldetection_znn/dataset.spec'
    create_dataset_spec(dataspec_path, preprocess_directory)