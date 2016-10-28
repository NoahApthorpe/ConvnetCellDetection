###############################################################
# Python pipeline for creating ZNN files
#
# Author: Alex Riordan
#
# Description: script that reads parameters from main
#    configuration file and creates ZNN configuration,
#    network, and dataset.spec files accordingly
###############################################################

import os
import sys
import shutil
import ConfigParser
from preprocess import (is_labeled, get_labeled_split, split_labeled_directory,
                           put_labeled_at_end_of_path_if_not_there, remove_ds_store, add_pathsep)


def create_dataset_spec(input_dir, output_dir, file_dict, cfg_parser, main_config_fpath, run_type):
    '''Writes dataset.spec file and returns number of image/label pairs referenced therein.
    N.B. this method will cause issues if any non-ROI filenames contain '_ROI.'
    will also cause issues if files aren't .tif'''
    f = open(output_dir + 'dataset.spec', 'w+')

    # Build a list of absolute file paths, with index given by file_dict
    files = range(len(file_dict.keys()))
    for fname, value in file_dict.items():
        if value[1] == '':
            files[value[0] - 1] = (input_dir + fname)
        else:
            files[value[0] - 1] = (input_dir + value[1] + os.sep + fname)
    # remove all filepaths with '_ROI.'
    files = [x for x in files if "_ROI." not in x]
    # remove .tif file extension
    files = [x.replace('.tif', '') for x in files]
    files = [dockerize_path(x) for x in files]

    s = ''
    # Iterate over remaining list. Write dataset.spec file. Sample number of file corresponds to index in file_dict.
    # TODO: Forward pass: for .tif files without corresponding ROI files, should just have label be equal to orig. .tif file
    if not cfg_parser.has_section('fnames'):
        cfg_parser.add_section('fnames')
    for fname in files:
        print fname
        section_num = file_dict[fname.split(os.sep)[-1] + '.tif'][0]
        s = write_one_section_dataset_spec(s, section_num, '' + fname + '.tif', fname + '_ROI.tif')
        # write section number (key) and fname (value) to config file
        cfg_parser.set('fnames', str(section_num), fname)
    f.write(s)
    f.close()

    if run_type == 'forward':
        with open(main_config_fpath, 'wb') as configfile:
            cfg_parser.write(configfile)

    return len(files)


def write_one_section_dataset_spec(s, section_num, stk_path, roi_path):
    ''' Helper method for create_dataset_spec.
    Given a unique stk filename and its corresponding ROI file,
    creates one section of a ZNN dataset.spec file'''
    s += '[image' + str(section_num) + ']\n'
    s += 'fnames = ' + str(stk_path) + '\n'
    s += 'pp_types = standard3D\n'
    s += 'is_auto_crop = yes\n\n'

    s += '[label' + str(section_num) + ']\n'
    s += 'fnames = ' + str(roi_path) + '\n'
    s += 'pp_types = auto\n'
    s += 'is_auto_crop = yes\n\n'

    s += '[sample' + str(section_num) + ']\n'
    s += 'input = ' + str(section_num) + '\n'
    s += 'output = ' + str(section_num) + '\n\n'

    return s


def dockerize_path(user_path):
    '''Rewrites user-provided path to point to mounted directory in Docker container
    Assumes ConvnetCellDetection directory is mounted in 'znn-release' directory in Docker container
    N.B. Assumes that no subfolders of ConvnetCellDetection are named 'ConvnetCellDetection' '''
    if '../' in user_path:
        s = user_path.split('../')
    else:
        s = user_path.split('ConvnetCellDetection/')
    new_path = '../ConvnetCellDetection/' + s[-1]
    return new_path


def create_znn_config_file(output_dir, train_indices, val_indices, forward_indices, new_net_fpath,
                           train_net_prefix, train_patch_size, learning_rate, momentum, num_iter_per_save,
                           max_iter, forward_net, forward_outsz, num_file_pairs):
    # copy default_znn_config.cfg from src to output_dir
    src_path = add_pathsep(os.path.dirname(os.path.abspath(__file__)))
    znn_config_path = output_dir + 'znn_config.cfg'
    shutil.copy(src_path + 'default_znn_config.cfg', znn_config_path)

    # use configParser to modify fields in new config file
    znn_cfg_parser = ConfigParser.SafeConfigParser()
    znn_cfg_parser.readfp(open(znn_config_path, 'r'))

    znn_cfg_parser.set('parameters', 'fnet_spec', dockerize_path(new_net_fpath))
    znn_cfg_parser.set('parameters', 'fdata_spec', dockerize_path(output_dir + 'dataset.spec'))
    znn_cfg_parser.set('parameters', 'train_net_prefix', dockerize_path(train_net_prefix))
    znn_cfg_parser.set('parameters', 'train_range', train_indices)
    znn_cfg_parser.set('parameters', 'test_range', val_indices)
    znn_cfg_parser.set('parameters', 'train_outsz', train_patch_size)
    znn_cfg_parser.set('parameters', 'eta', learning_rate)
    znn_cfg_parser.set('parameters', 'momentum', momentum)
    znn_cfg_parser.set('parameters', 'Num_iter_per_save', num_iter_per_save)
    znn_cfg_parser.set('parameters', 'max_iter', max_iter)
    znn_cfg_parser.set('parameters', 'forward_range', forward_indices)  # autoset as everything in input_dir
    znn_cfg_parser.set('parameters', 'forward_net', dockerize_path(forward_net))
    znn_cfg_parser.set('parameters', 'forward_outsz', forward_outsz)  # TODO: calculate forward_outsz automatically, based on field of view
    znn_cfg_parser.set('parameters', 'output_prefix', dockerize_path(output_dir))
    with open(znn_config_path, 'wb') as configfile:
        znn_cfg_parser.write(configfile)


def get_train_val_forward_split_indices_as_str(file_dict):
    '''Input: file_dict dictionary with keys = input_tif_file_names, values = (index, subdir for train/val/test split if applicable)
    Output: string of comma-separated indices for training set, validation set, and files upon which to perform
    a forward pass (for znn_config.cfg)'''
    train_indices = ''
    val_indices = ''
    forward_indices = ''
    no_subdirs = True
    for key, values in file_dict.items():
        if values[1] == 'training' and "_ROI." not in key:
            train_indices += str(values[0]) + ','
            no_subdirs = False
        if values[1] == 'validation' and "_ROI." not in key:
            val_indices += str(values[0]) + ','
            no_subdirs = False
        if "_ROI." not in key:
            forward_indices += str(values[0]) + ','
    if no_subdirs:
        train_indices = '1'
        val_indices = '1'
    if train_indices[-1] == ',':
        train_indices = train_indices[:-1]
    if val_indices[-1] == ',':
        val_indices = val_indices[:-1]
    if forward_indices[-1] == ',':
        forward_indices = forward_indices[:-1]
    return train_indices, val_indices, forward_indices


def copy_net_and_set_conv_filter_size(net_arch_fpath, new_net_fpath, filter_size):
    '''write each line from orig. network to new_net_fpath, changing only lines with conv filter'''
    with open(net_arch_fpath) as orig_net, open(new_net_fpath, 'w') as new_net:
        is_conv = False
        past_fcedges = False
        for line in orig_net:
            if 'fcedges' in line:
                past_fcedges = True
            if 'type conv' in line and not past_fcedges:
                is_conv = True
            if 'size' in line and is_conv and not past_fcedges:
                new_net.write('size 1,' + filter_size + ',' + filter_size + '\n')
                is_conv = False
            else:
                new_net.write(line)


def set_network_3D_to_2D_squashing_filter_size(new_net_fpath, filter_size):
    '''Changes z size and stride of final max filter in a network file'''
    with open(new_net_fpath, 'r') as net_file:
        all_lines = net_file.readlines()
        # find index of file line that contains 'type max_filter'
        for i in range(len(all_lines) - 1, -1, -1):
            if 'type max_filter' in all_lines[i]:
                start = i
        # change filter size and stride only in lines immediately following 'type max_filter'
        counter = 2
        for i in range(start, len(all_lines)):
            line = all_lines[i]
            if counter == 0:
                break
            if 'size' in all_lines[i]:
                counter -= 1
                all_lines[i] = 'size ' + filter_size + ',1,1\n'
            elif 'stride' in all_lines[i]:
                counter -= 1
                all_lines[i] = 'stride ' + filter_size + ',1,1\n'
    # write changed file lines to new_net_fpath
    with open(new_net_fpath, 'w') as net_file:
        net_file.writelines(all_lines)


def build_unlabeled_file_dict(file_dir):
    '''Returns dict where keys are file names of file_dir and values are the empty string '' '''
    files = os.listdir(file_dir)
    remove_ds_store(files)
    file_dict = dict()
    file_dict.update(dict.fromkeys(files, ''))
    return file_dict


def add_indices_to_dict(fname_dict):
    '''Updates values in a dictionary to be a tuple with (integer_index, original_value)'''
    index = 1
    for fname, orig_value in fname_dict.items():
        fname_dict[fname] = (index, orig_value)
        index += 1


def get_io_dirs(run_type, cfg_parser):
    '''Gets input/output directories for either training data or forward pass'''
    if run_type != 'forward' and run_type != 'training':
        raise ValueError('run_type variable should be one of "forward" or "training"', run_type)
    input_dir = add_pathsep(cfg_parser.get(run_type, run_type + '_input_dir'))
    output_dir = add_pathsep(cfg_parser.get(run_type, run_type + '_output_dir'))
    return input_dir, output_dir


def check_tif_depth():
    '''Checks that depth of tif files in dataset is same across files and matches depth of network.'''
    pass


def main(main_config_fpath='main_config.cfg', run_type='forward'):
    '''Get user-specified information from main_config.cfg'''
    cfg_parser = ConfigParser.SafeConfigParser()
    cfg_parser.readfp(open(main_config_fpath, 'r'))
    img_width = cfg_parser.get('general', 'img_width')
    img_height = cfg_parser.get('general', 'img_height')
    net_arch_fpath = cfg_parser.get('network', 'net_arch_fpath')
    train_net_prefix = cfg_parser.get('training', 'training_net_prefix')
    train_patch_size = cfg_parser.get('training', 'patch_size')
    learning_rate = cfg_parser.get('training', 'learning_rate')
    momentum = cfg_parser.get('training', 'momentum')
    num_iter_per_save = cfg_parser.get('training', 'num_iter_per_save')
    max_iter = cfg_parser.get('training', 'max_iter')
    forward_net = cfg_parser.get('forward', 'forward_net')
    forward_outsz = cfg_parser.get('forward', 'forward_outsz')
    filter_size = cfg_parser.get('network', 'filter_size')
    is_squashing = cfg_parser.get('network', 'is_squashing')
    time_equalize = cfg_parser.get('preprocessing', 'time_equalize')

    # Get and make user-specified input/output directories
    input_dir, output_dir = get_io_dirs(run_type, cfg_parser)
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #docker_dir = dockerize_path(output_dir)

    # Build file name dictionary
    if is_labeled(input_dir):
        file_dict = get_labeled_split(input_dir)
    else:
        file_dict = build_unlabeled_file_dict(input_dir)
    add_indices_to_dict(file_dict)
    train_indices, val_indices, forward_indices = get_train_val_forward_split_indices_as_str(file_dict)

    # Create znn files and save to output_dir
    new_net_fpath = output_dir + net_arch_fpath.split(os.sep)[-1]
    num_file_pairs = create_dataset_spec(input_dir, output_dir, file_dict, cfg_parser, main_config_fpath, run_type)
    create_znn_config_file(output_dir, train_indices, val_indices, forward_indices, new_net_fpath,
                           train_net_prefix, train_patch_size, learning_rate, momentum, num_iter_per_save,
                           max_iter, forward_net, forward_outsz, num_file_pairs)

    copy_net_and_set_conv_filter_size(net_arch_fpath, new_net_fpath, filter_size)
    if is_squashing:
        set_network_3D_to_2D_squashing_filter_size(new_net_fpath, time_equalize)

if __name__ == "__main__":
    main()
