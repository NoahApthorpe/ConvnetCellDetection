##########################################################
#
# Module to run ZNN commands in a Docker container
#
# Author: Alex Riordan
#
# Description: Python interface to ZNN running in Docker
#
##########################################################

import subprocess
import ConfigParser
import os
import signal
from create_znn_files import dockerize_path
from preprocess import remove_ds_store


def start_docker_machine(memory):
    '''Input: memory allocates memory (in MB) to docker-machine'''
    cmd = ''
    cmd += 'docker-machine create -d virtualbox --virtualbox-memory ' + memory + ' default; '
    cmd += 'docker-machine start default; '  # TODO: make default a randomly-generated name
    cmd += 'eval $(docker-machine env)'
    # cmd += ' ;docker run hello-world' #for testing
    return cmd


def start_znn_container(dir_to_mount, container_name):
    cmd = ''
    cmd += 'docker run -v ' + dir_to_mount + ':/opt/znn-release/ConvnetCellDetection '
    cmd += '--name ' + container_name + ' -it jpwu/znn:v0.1.4 ' + '/bin/bash -c '
    return cmd


def train_network(output_dir):
    output_dir = dockerize_path(output_dir)
    cmd = ''
    cmd += '"cd opt/znn-release/python; sudo ldconfig; python train.py -c ' + output_dir + '/znn_config.cfg"'
    return cmd


def forward_pass(output_dir):
    cmd = ''
    cmd += '"cd opt/znn-release/python; sudo ldconfig; python forward.py -c ' + output_dir + '/znn_config.cfg"'
    return cmd


def remove_znn_container(container_name):
    cmd = ''
    cmd += '; docker stop ' + container_name + '; docker rm ' + container_name + '; docker-machine stop'
    # TODO: change default docker machine to user-given name
    return cmd


def rename_output_files(cfg_parser, main_config_fpath, forward_output_dir):
    '''Maps ZNN output fnames back to user-given fnames'''
    dict_list = cfg_parser.items('fnames')
    for item in dict_list:
        number = item[0]
        fname = item[1]
        old_fname = forward_output_dir + '/_sample' + str(number)
        new_fname = forward_output_dir + '/' + fname.split('/')[-1]
        os.rename(old_fname + '_output.h5', new_fname + '_output.h5')
        os.rename(old_fname + '_output_0.tif', new_fname + '_output_0.tif')
        os.rename(old_fname + '_output_1.tif', new_fname + '_output_1.tif')


def main(main_config_fpath='../main_config_ar.cfg', run_type='forward'):
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath, 'r'))
    memory = cfg_parser.get('docker', 'memory')
    container_name = cfg_parser.get('docker', 'container_name')
    training_output_dir = cfg_parser.get('training', 'training_output_dir')
    forward_output_dir = cfg_parser.get('forward', 'forward_output_dir')

    dir_to_mount = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Mounts ConvnetCellDetection directory
    print dir_to_mount

    cmd = ''
    cmd += start_docker_machine(memory)
    cmd += '; '
    cmd += start_znn_container(dir_to_mount, container_name)

    if run_type == 'training':
        cmd += train_network(dockerize_path(training_output_dir)) + remove_znn_container(container_name)
    elif run_type == 'forward':
        cmd += forward_pass(dockerize_path(forward_output_dir)) + remove_znn_container(container_name)
    else:
        cmd += remove_znn_container(container_name)
        raise ValueError('run_type variable should be one of "forward" or "training"', run_type)

    process = subprocess.Popen(cmd, shell=True)
    process.communicate()

    if run_type == 'forward':
        rename_output_files(cfg_parser, main_config_fpath, forward_output_dir)


if __name__ == "__main__":
    main()
