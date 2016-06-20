full_stack = zeros(512,512,0);
mean_stack = zeros(512,512,0);
d = dir;
d = {d.name};
for f=d
    fname = char(f)
    if strcmp(fname, '.') || strcmp(fname, '..') || strcmp(fname, '.DS_Store') || strcmp(fname, 'ref')
        continue
    end
    info = imfinfo(fname);
    num_images = numel(info)
    for k = 1:num_images
        A = imread(fname, k, 'Info', info);
        full_stack = cat(3, full_stack, A);
    end
    size(full_stack)
    while size(full_stack,3) > 167
        section = full_stack(:,:,1:167);    
        mean_section = mean(section,3);
        mean_stack = cat(3, mean_stack, mean_section);
        full_stack = full_stack(:,:,168:end);
    end
    size(full_stack)
    size(mean_stack)
end
mean_stack = cat(3, mean_stack, mean(full_stack, 3));

m = fix(size(mean_stack,3)/6)*6
mean_stack_even = mean_stack(:,:,1:m);
max_stack_even = reshape(mean_stack_even, 512, 512, [], 6);
max_stack_even = max(max_stack_even,[],4);
if mod(size(mean_stack,3),6) ~= 0
    mean_stack_rem = mean_stack(:,:,m:end);
    max_stack_rem = max(mean_stack_rem,[],3);
    max_stack_final = cat(3, max_stack_even, max_stack_rem);
else
    max_stack_final = max_stack_even;
end

t = Tiff('preprocessed.tif','w');
tagStruct.Photometric = Tiff.Photometric.MinIsBlack;
tagStruct.BitsPerSample = 32;
tagStruct.SamplesPerPixel = 1;
tagStruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagStruct.ImageLength = 512;
tagStruct.ImageWidth = 512;
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

