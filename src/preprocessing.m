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
downsampled_dir = cfg_parameters.general.downsample_dir;
do_downsample = cfg_parameters.general.do_downsample;

if data_dir(end-1) ~= '/'
    data_dir = strcat(data_dir,'/');
end
if downsampled_dir(end-1) ~= '/'
    downsampled_dir = strcat(downsampled_dir, '/');
end

if ~exist(downsampled_dir, 'dir')
    mkdir(downsampled_dir);
end

if do_downsample
    downsample_tif(data_dir, downsampled_dir, img_width, img_height,  mean_proj_bin, max_proj_bin);
else
    copyfile(data_dir, downsampled_dir);
end

time_equalize(downsampled_dir, img_width, img_height, new_time_depth);









