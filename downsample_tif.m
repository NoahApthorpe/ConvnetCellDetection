%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample raw tif video
%
% Author: Noah Apthorpe
% 
% Description: downsamples a raw .tif video that has 
%  been split into multiple files by mean-projecting
%  consecutive sets of 167 frames over time and then
%  max-projecting resulting consecutive sets of 
%  6 frames over time. 
%
% Usage: 
%  1) add this .m file to Matlab path
%  2) open the directory containing *only* .tif files 
%     from a single video (named in alphabetic or numeric order
%     e.g. vid_sec_01.tif, vid_sec_02.tif)
%     as the "current folder" in Matlab
%  3) adjust the image_size variable below to n 
%     for nxn video frames (default frame 512x512 pixels)
%  4) run the script; intermediate progress updates 
%     will print to the command window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% width and height of video frames in pixels
image_size = 512; 

% create arrays to store results
full_stack = zeros(image_size,image_size,0);
mean_stack = zeros(image_size,image_size,0);

% loop over all files in current folder
d = dir;
d = {d.name};
for f=d
    fname = char(f)
    
    % ignore self and parent links and '.DS_Store' file
    % NOTE: make this if statement more exclusionary if 
    %       directory contains additional non-video files
    if strcmp(fname, '.') || strcmp(fname, '..') || strcmp(fname, '.DS_Store') || strcmp(fname, 'ref')
        continue
    end
    
    % read all video frames of current file
    info = imfinfo(fname);
    num_images = numel(info)
    for k = 1:num_images
        A = imread(fname, k, 'Info', info);
        full_stack = cat(3, full_stack, A);
    end
    size(full_stack)
    
    % mean-project frames 
    while size(full_stack,3) > 167
        section = full_stack(:,:,1:167);    
        mean_section = mean(section,3);
        mean_stack = cat(3, mean_stack, mean_section);
        full_stack = full_stack(:,:,168:end);
    end
    size(full_stack)
    size(mean_stack)
end
% mean-project remaining frames (# < 167)
mean_stack = cat(3, mean_stack, mean(full_stack, 3));

% max-project frames
m = fix(size(mean_stack,3)/6)*6
mean_stack_even = mean_stack(:,:,1:m);
max_stack_even = reshape(mean_stack_even, image_size, image_size, [], 6);
max_stack_even = max(max_stack_even,[],4);
if mod(size(mean_stack,3),6) ~= 0
    mean_stack_rem = mean_stack(:,:,m:end);
    max_stack_rem = max(mean_stack_rem,[],3);
    max_stack_final = cat(3, max_stack_even, max_stack_rem);
else
    max_stack_final = max_stack_even;
end

% save resultant downsampled video as a .tif file
t = Tiff('preprocessed.tif','w');
tagStruct.Photometric = Tiff.Photometric.MinIsBlack;
tagStruct.BitsPerSample = 32;
tagStruct.SamplesPerPixel = 1;
tagStruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagStruct.ImageLength = image_size;
tagStruct.ImageWidth = image_size;
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

