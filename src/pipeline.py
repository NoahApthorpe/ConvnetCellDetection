#!/usr/bin/env python

'''
 Master file for running ConvNet pipeline 
 
 Author: Alex Riordan
      
 Description: TBA
'''

import os, sys, subprocess, ConfigParser
import preprocess, create_znn_files, run_znn_docker, postprocess, score

#TODO: test this method
def complete_pipeline(main_config_fpath):
    preprocessing(main_config_fpath)
    train(main_config_fpath)
    forward_pass(main_config_fpath)
    postprocessing(main_config_fpath)
    score_labeled_data(main_config_fpath)

#TODO: test this method
def create_expt_dir(dir_name):
    if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    else:
        raise AssertionError('Directory already exists.\n Choose a different name for your new experiment directory. ', dir_name)
    os.makedirs(dir_name + '/labeled/test')
    os.makedirs(dir_name + '/labeled/training')
    os.makedirs(dir_name + '/labeled/validation')
    print 'new experiment directory ', dir_name, ' created.' 
    #Should autowrite main_config file?
    #Then move this method to its own file?
    #main_config file can be autowritten based on expt_name (e.g. "v1_final")
    
    
#TODO: rewrite all files as callable with main_config_fpath as parameter 
def preprocessing(main_config_fpath):
    #Get paths of MATLAB and preprocessing.m 
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath, 'r'))
    matlab_path = cfg_parser.get('general', 'matlab_path')
    src_path = os.path.dirname(os.path.abspath(__file__))
    cmd_mlab = matlab_path + ' -nodesktop -nosplash -r '
    cmd_cd = '\"cd(\'' + os.path.dirname(os.path.abspath(__file__)) + '\'); '
    cmd_path = 'path(path, \'' + os.path.dirname(os.path.abspath(__file__)) + '\'); '
    cmd_preprocess = 'preprocess(\'' + main_config_fpath + '\')\"' 
    cmd = cmd_mlab + cmd_cd + cmd_path + cmd_preprocess
    
    print 'Running initial preprocessing steps in MATLAB...'
    process = subprocess.Popen(cmd, shell=True) 
    process.communicate()
    
    print 'Running final preprocessing in Python...'
    preprocess.main(main_config_fpath)

def train(main_config_fpath):
    run_type = 'training'
    print 'Creating ZNN files for training...'
    create_znn_files.main(main_config_fpath, run_type)
    print 'Preparing to run ZNN in Docker for training...'
    run_znn_docker.main(main_config_fpath, run_type)

def forward_pass(main_config_fpath):
    run_type = 'forward'
    print 'Creating ZNN files for forward pass...'
    create_znn_files.main(main_config_fpath, run_type)
    print 'Preparing to run ZNN in Docker for forward pass...'
    run_znn_docker.main(main_config_fpath, run_type)

def postprocessing(main_config_fpath):
    print 'Postprocessing results of forward pass...'
    postprocess.main(main_config_fpath)
    
def score_labeled_data(main_config_fpath):
    score.main(main_config_fpath)

def is_main_config(main_config_fpath):
    pass

if __name__ == "__main__":
    pass
    #User should point to main_config file for all methods but create_expt_dir
    #Args string should contain a keyword (e.g. 'complete' or 'forward') that specifies what user wants to do
    #can use a switch statement to deal with this
    # 
    # print 'Number of arguments: ', len(sys.argv), ' arguments'
    # print 'Argument list: ', str(sys.argv)
    
    cmd = sys.argv[1]
    param = sys.argv[2]
    run_dict = {'complete': complete_pipeline,
                'new_experiment': create_expt_dir,
                'preprocess': preprocessing,
                'train': train,
                'forward': forward_pass,
                'postprocess': postprocessing,
                'score': score_labeled_data,
                }
    run_dict[cmd](param)
