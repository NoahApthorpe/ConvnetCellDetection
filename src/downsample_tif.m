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
video_dirs = dir;
video_dirs = {video_dirs.name};
for vdir=video_dirs;
    vdir_name = char(vdir)
    
    % ignore self and parent links and '.DS_Store' file
    % NOTE: make this if statement more exclusionary if 
    %       directory contains additional non-video folders
    if strcmp(vdir_name, '.') || strcmp(vdir_name, '..') || strcmp(vdir_name, '.DS_Store') || strcmp(vdir_name, 'ref')
        continue
    end
    
    % create arrays to store results
    full_stack = zeros(img_width,img_height,0);
    mean_stack = zeros(img_width,img_height,0);
    
    % loop over all files in current video directory
    data_dir_old = cd(vdir_name);
    d = dir;
    d = {d.name};
    for f=d;
        fname = char(f)
        
        % ignore files without >=3 character extensions
        if length(fname) < 4
            continue
        end
        
        % copy roi zip file to output directory
        if strcmpi(fname(end-3:end), '.zip')
            output_name = strcat(downsampled_dir, vdir_name, '.zip');
            copyfile(fname, output_name);
            continue
        end
        
        % ignore other non-tiff files
        if ~strcmpi(fname(end-4:end), '.tiff') && ~strcmpi(fname(end-3:end), '.tif')
            continue
        end

        % read all video frames of current file
        info = imfinfo(fname);
        num_images = numel(info);
        for k = 1:num_images
            A = imread(fname, k, 'Info', info);
            full_stack = cat(3, full_stack, A);
        end
        size(full_stack)

        % mean-project frames 
        while size(full_stack,3) > mean_proj_bins
            section = full_stack(:,:,1:mean_proj_bins);    
            mean_section = mean(section,3);
            mean_stack = cat(3, mean_stack, mean_section);
            full_stack = full_stack(:,:,mean_proj_bins+1:end);
        end
        size(full_stack)
        size(mean_stack)
    end
    
    % mean-project remaining frames (# < mean_proj_bins)
    mean_stack = cat(3, mean_stack, mean(full_stack, 3));

    % max-project frames
    m = fix(size(mean_stack,3)/max_proj_bins)*max_proj_bins;
    mean_stack_even = mean_stack(:,:,1:m);
    max_stack_even = reshape(mean_stack_even, img_width, img_height, [], max_proj_bins);
    max_stack_even = max(max_stack_even,[],4);
    if mod(size(mean_stack,3),max_proj_bins) ~= 0
        mean_stack_rem = mean_stack(:,:,m:end);
        max_stack_rem = max(mean_stack_rem,[],3);
        max_stack_final = cat(3, max_stack_even, max_stack_rem);
    else
        max_stack_final = max_stack_even;
    end

    % save resultant downsampled video as a .tif file
    output_name = strcat(downsampled_dir, vdir_name, '.tif');
    t = Tiff(output_name,'w');
    tagStruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagStruct.BitsPerSample = 32;
    tagStruct.SamplesPerPixel = 1;
    tagStruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagStruct.ImageLength = img_height;
    tagStruct.ImageWidth = img_width;
    tagStruct.RowsPerStrip = 256;
    tagStruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagStruct.Compression = 1;
    t.setTag(tagStruct);
    for i = 1:size(max_stack_final,3)
        if (i == 1)
            t.write(max_stack_final(:,:,i));
        else
            t.writeDirectory()
            t.setTag(tagStruct);
            t.write(max_stack_final(:,:,i));
        end
    end
    t.close();
    cd(data_dir_old);
end
cd(orig_dir);
end

