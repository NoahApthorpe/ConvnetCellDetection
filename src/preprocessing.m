%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab preprocessing pipeline
%
% Author: Noah Apthorpe
% 
% Description: script that reads parameters from
%     configuration file and runs the downsample_tif
%     and time_equalize functions on all specified data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cfg_file = './main_config.cfg';
cfg_parameters = cfg2struct(cfg_file);
data_dir = cfg_parameters.general.data_dir;
img_width = cfg_parameters.general.img_width;
img_height = cfg_parameters.general.img_height;
new_time_depth = cfg_parameters.preprocessing.time_equalize;
mean_proj_bin = cfg_parameters.preprocessing.mean_proj_bin;
max_proj_bin = cfg_parameters.preprocessing.max_proj_bin;

do_downsample = cfg_parameters.general.do_downsample;

if do_downsample
    if data_dir(end) == '/' 
        data_dir = data_dir(1:end-1);
    end
    downsampled_dir = strcat(data_dir, '_preprocessed/');
    if ~exist(downsampled_dir, 'dir')
        mkdir(downsampled_dir);
    end
    downsample_tif(data_dir, downsampled_dir, img_width, img_height,  mean_proj_bin, max_proj_bin);
    time_equalize(downsampled_dir, img_width, img_height, new_time_depth);

else
    if ~(data_dir(end-1) == '/')
        data_dir = strcat(data_dir,'/');
    end
    time_equalize(data_dir, img_width, img_height, new_time_depth);
end










