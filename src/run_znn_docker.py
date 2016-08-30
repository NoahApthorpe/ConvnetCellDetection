#!/usr/bin/env python

import subprocess, ConfigParser
import signal
from create_znn_files import dockerize_path

'''
 Module to run ZNN commands in a Docker container 
 
 Author: Alex Riordan
      
 Description: TBA
'''


'''
Input: memory allocates memory (in MB) to docker-machine
'''
def start_docker_machine(memory):
    cmd = ''
    cmd += 'docker-machine create -d virtualbox --virtualbox-memory '+ memory + ' default; '
    cmd += 'docker-machine start default; ' #TODO: make default a randomly-generated name
    cmd += 'eval $(docker-machine env); '
    cmd += 'docker run hello-world'
    return cmd
    
    
def start_znn_container(dir_to_mount):
    cmd = ''
    cmd += 'docker run -v ' + dir_to_mount + ':/opt/znn-release/ConvnetCellDetection -it jpwu/znn:v0.1.4 '
    cmd += '/bin/bash -c ' 
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

#TODO: write function, maps sample numbers back to original image fnames 
def rename_output_files():
    pass


if __name__ == "__main__":
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open('../main_config_ar.cfg', 'r'))
    memory = cfg_parser.get('docker', 'memory')
    training_output_dir = cfg_parser.get('training', 'training_output_dir')
    forward_output_dir = cfg_parser.get('forward', 'forward_output_dir')
    
    run_type = 'forward' #this will be set in main_config or pipeline.py
    
    cmd = ''
    cmd += start_docker_machine(memory)
    cmd += '; '
    cmd += start_znn_container('/Users/sergiomartinez/Documents/ConvnetCellDetection')
    
    #TODO: may need to keep ZNN container running and have a separate command for training/forward pass
    # OR: insert in "command" parameter in 'start_znn_container?'
    if run_type == 'training':
        cmd += train_network(dockerize_path(training_output_dir))
    elif run_type == 'forward':
        cmd += forward_pass(dockerize_path(forward_output_dir))
    else:
        raise ValueError('run_type variable should be one of "forward" or "training"', run_type)
    
    print cmd
    process = subprocess.Popen(cmd, shell=True)
    process.communicate() #cleanly shuts down process 
    