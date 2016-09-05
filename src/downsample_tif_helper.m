function downsample_tif_helper(imgs, img_width, img_height,  mean_proj_bins, max_proj_bins, output_name)

% create arrays to store results
full_stack = zeros(img_width,img_height,0);
mean_stack = zeros(img_width,img_height,0);

% loop over all images
for f=imgs;
    fname = char(f);

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
max_stack_final = single(max_stack_final);
% save resultant downsampled video as a .tif file
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

end