###############################################################
#
# Master file for running ConvNet pipeline
#
# Author: Alex Riordan & Noah Apthorpe
#
# Description: Interface to run ConvNet pipeline
#
# Usage: python pipeline.py create <experiment name>
#        python pipeline.py <pipeline step> <config file path>
#
#        pipeline step options:
#           * complete (run entire pipeline)
#           * preprocess
#           * train
#           * postprocess
#           * forward
#           * score
#
##############################################################

import sys
import create_experiment_dir
import preprocess
import create_znn_files
import run_znn_docker
import postprocess
import score


def complete_pipeline(main_config_fpath):
    '''Run entire pipeline'''
    preprocessing(main_config_fpath)
    train(main_config_fpath)
    forward_pass(main_config_fpath)
    postprocessing(main_config_fpath)
    score_labeled_data(main_config_fpath)


def create_expt_dir(experiment_name):
    '''Create directory structure and default config file for new experiment'''
    create_experiment_dir.main(experiment_name)


def preprocessing(main_config_fpath):
    '''Run preprocessing'''
    print 'Running preprocessing...'
    preprocess.main(main_config_fpath)


def train(main_config_fpath):
    '''Train convnet using ZNN'''
    run_type = 'training'
    print 'Creating ZNN files for training...'
    create_znn_files.main(main_config_fpath, run_type)
    print 'Preparing to run ZNN in Docker for training...'
    run_znn_docker.main(main_config_fpath, run_type)


def forward_pass(main_config_fpath):
    '''Run a forward pass using existing trained network'''
    run_type = 'forward'
    print 'Creating ZNN files for forward pass...'
    create_znn_files.main(main_config_fpath, run_type)
    print 'Preparing to run ZNN in Docker for forward pass...'
    run_znn_docker.main(main_config_fpath, run_type)


def postprocessing(main_config_fpath):
    '''Run postprocessing'''
    print 'Postprocessing results of forward pass...'
    postprocess.main(main_config_fpath)


def score_labeled_data(main_config_fpath):
    '''Score ConvNet precision and accuracy on labeled data'''
    score.main(main_config_fpath)


if __name__ == "__main__":
    cmd = sys.argv[1]
    param = sys.argv[2]
    run_dict = {'complete': complete_pipeline,
                'create': create_expt_dir,
                'preprocess': preprocessing,
                'train': train,
                'forward': forward_pass,
                'postprocess': postprocessing,
                'score': score_labeled_data}
    run_dict[cmd](param)
