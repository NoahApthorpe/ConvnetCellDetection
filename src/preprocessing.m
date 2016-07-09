%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab preprocessing pipeline
%
% Author: Noah Apthorpe
% 
% Description: script that reads parameters from
%     configuration file and runs the downsample_tif
%     and time_equalize functions on all specified data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cfg_file = 'config.cfg';
cfg_parameters = cfg2struct(cfg_file);

data_dir = cfg_parameters.general.data_dir;
downsampled_dir = cfg_parameters.general.downsampled_dir;
img_width = cfg_parameters.general.img_width;
img_height = cfg_parameters.general.img_height;
mean_proj_bins = cfg_parameters.preprocessing.mean_proj_bins;
max_proj_bins = cfg_parameters.preprocessing.max_proj_bins;

downsample_tif(data_dir, downsampled_dir, img_width, img_height,  mean_proj_bins, max_proj_bins);

% add parameters and function call for time_equalize

