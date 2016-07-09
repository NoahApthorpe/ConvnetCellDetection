%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to read configuration file into structure
%
% Author: Noah Apthorpe
%
% Description: reads configuration file at "filepath" parameter
%     and returns structure containing configuration variables
%     in format cfg_struct.[section_name].[key] = value
%     values not enclosed in quotes are automatically
%     converted to numbers if possible
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function cfg_struct = cfg2struct(filepath)

cfg_file = fopen(filepath, 'r');
section = '';
while ~feof(cfg_file)
    s = strtrim(fgetl(cfg_file));
    if isempty(s) || isempty(strtrim(strtok(s,'#')))
        continue
    end
    
    if s(1)=='['
        section = matlab.lang.makeValidName(strtok(s(2:end), ']'));
        cfg_struct.(section) = [];
        continue
    end
    
    [key, value] = strtok(s, ':');
    value = strtrim(value(2:end));
    
    if value(1) == '"'
        value = strtok(value(2:end),'"');
    elseif value(1) == ''''
        value = strtok(value(2:end), '''');
    else 
        value = strtok(value, '#');
        value = strtrim(value);
    end
        
    [numvalue, can_num_convert] = str2num(value);
    if can_num_convert
        value = numvalue;
    end
    
    if isempty(section)
        cfg_struct.(matlab.lang.makeValidName(key)) = value;
    else
        cfg_struct.(section).(matlab.lang.makeValidName(key)) = value;
    end
end
fclose(cfg_file);
end
    