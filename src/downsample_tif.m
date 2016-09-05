%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample raw tif video
%
% Author: Noah Apthorpe
% 
% Description: downsamples a raw .tif video that has 
%  been split into multiple files by mean-projecting
%  consecutive sets of frames over time and then
%  max-projecting resulting consecutive sets of 
%  frames over time. 
%
%  Parameters:
%     1) data_dir: Outer directory containing folders for individual videos
%                  Each video folder should have .tif files from a
%                  single video (named in alphabetic or numeric order
%                  e.g. vid_sec_01.tif, vid_sec_02.tif)
%     2) downscaled_dir: Destination directory of downscaled videos
%                        Each downscaled video will have same name as 
%                        directory containing its original .tif files
%     3) img_width: width of videos in pixels
%     4) img_height: height of videos in pixels
%     5) mean_proj_bins: number of frames per mean-projection
%     6) max_proj_bins: number of frames per max-projection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = downsample_tif(data_dir, downsampled_dir, img_width, img_height,  mean_proj_bins, max_proj_bins)

% loop over video directories in data directories
orig_dir = cd(data_dir);
tl_files = dir;
tl_files = {tl_files.name};
for tl_file=tl_files;
    tlf_name = char(tl_file)
    
    [~, ~, ext] = fileparts(tlf_name);
    if strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        downsample_tif_helper([data_dir + tlf_name], img_width, img_height, mean_proj_bins, max_proj_bins, downsampled_dir + tlf_name + '.tif');
    
    elseif strcmpi(ext, '.zip') 
        output_name = strcat(downsampled_dir, tlf_name, '.zip');
        copyfile(tlf_name, output_name);
    
    elseif isdir(tlf_name) && ~strcmp(tlf_name, '.') && ~strcmp(tlf_name, '..') 
        data_dir_old = cd(tlf_name);
        d = dir;
        d = {d.name};
        imgs = [];
        for fname=d
            [~, ~, ext] = fileparts(fname);
            if strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
                imgs = [imgs, fname];
            elseif strcmpi(ext, '.zip')
                output_name = strcat(downsampled_dir, tlf_name, '.zip');
                copyfile(fname, output_name);
            end
        end
        downsample_tif_helper(imgs, img_width, img_height, mean_proj_bins, max_proj_bins, downsampled_dir + fname + '.tif');
        cd(data_dir_old);
    end
end
cd(orig_dir);
end