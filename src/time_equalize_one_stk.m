%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resizes one .tif stack in space and time
% width, and time-depth. 
%
% Authors: Alexander Riordan, Noah Apthorpe
%
% Description: Resizes one .tif stack using downsampling and bicubic
% interpolation if necessary. All input variables should be strings. 
% 
% Uses MIJ MATLAB-ImageJ interface 
% http://imagej.net/Miji
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = time_equalize_one_stk(fname,new_width,new_height,new_depth)


    MIJ.run('Open...', strcat('path=[', fname, ']') );
    MIJ.run('Size...', strcat('width=',new_width,' height=',new_height,' depth=',new_depth,' constrain average interpolation=Bicubic'));
    MIJ.run('Save', strcat('save=',fname));
    MIJ.close();
end

