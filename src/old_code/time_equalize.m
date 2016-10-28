%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resizes all .tif stacks in a folder to the same height, 
% width, and time-depth. 
%
% Authors: Alexander Riordan, Noah Apthorpe
%
% Description: Resizes all .tif stacks in a folder to the same height, 
% width, and time-depth. Height and width are user-specified, and
% time-depth is set to the lower of a user-specified input or the minimum 
% time-depth of all stacks. 
% Can handle arbitrary upsizing and downsizing, using bicubic interpolation 
% and averaging, respectively. 
%
% All input variables should be strings. 
% 
% Uses MIJ MATLAB-ImageJ interface 
% http://imagej.net/Miji
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = time_equalize(preprocessed_data_folder,...
    new_width, new_height, new_time_depth)

%adds MIJ script to MATLAB path
addpath('/Applications/Fiji.app/scripts');
%opens MIJ without GUI
Miji(false);

% %for each file in preprocessed data folder: 
% fname = 'AMG1_exp1_new_001_all_condensed.tif';
% fpath = strcat(preprocessed_data_folder,fname);
% time_equalize_one_stk(fpath,new_width,new_height,new_depth);
% MIJ.close();

% find minimum time depth of all stacks
d = dir(preprocessed_data_folder);
d = {d.name};
min_time_depth = new_time_depth;
for f=d
    fname = char(f);

    % ignore files without >=3 character extensions
    if length(fname) < 4
        continue
    end
    % ignore other non-tiff files
    if ~strcmpi(fname(end-4:end), '.tiff') && ~strcmpi(fname(end-3:end), '.tif')
        continue
    end

    % append folder to fname for absolute path 
    fpath = strcat(preprocessed_data_folder,fname);
    
    % get time depth of file, and store if smallest observed
    time_depth = length(imfinfo(fpath));
    if time_depth < min_time_depth
        min_time_depth = time_depth;
    end
end

% resize all stacks
% loop over all files in current folder
for f=d
    fname = char(f);
    
    % ignore files without >=3 character extensions
    if length(fname) < 4
        continue
    end
    % ignore other non-tiff files
    if ~strcmpi(fname(end-4:end), '.tiff') && ~strcmpi(fname(end-3:end), '.tif')
        continue
    end
    
    % append folder to fname for absolute path 
    fpath = strcat(preprocessed_data_folder,fname);
    
    % resize one stk
    time_equalize_one_stk(fpath,num2str(new_width),num2str(new_height),num2str(min_time_depth));
    MIJ.close();
end

return; 