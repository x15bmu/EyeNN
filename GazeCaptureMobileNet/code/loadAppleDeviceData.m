% loadAppleDeviceData.m
%
% Loads the apple_device_data.csv file into the workspace as variables
% prefixed with "device."

%% Import data from text file.
% Script for importing data from apple_device_data.csv.
% Auto-generated by MATLAB and tweaked.

%% If the variables are already in the workspace, terminate early.
if exist('deviceName','var') && ...
exist('deviceCameraToScreenXMm','var') && ...
exist('deviceCameraToScreenYMm','var') && ...
exist('deviceCameraXMm','var') && ...
exist('deviceCameraYMm','var') && ...
exist('devicePixelsPerInch','var') && ...
exist('deviceScreenXMm','var') && ...
exist('deviceScreenYMm','var') && ...
exist('deviceScreenWidthMm','var') && ...
exist('deviceScreenWidthPoints','var') && ...
exist('deviceScreenWidthPointsZoomed','var') && ...
exist('deviceScreenHeightMm','var') && ...
exist('deviceScreenHeightPoints','var') && ...
exist('deviceScreenHeightPointsZoomed','var');
    return;
end

%% Initialize variables.
filename = 'apple_device_data.csv';
delimiter = ',';
startRow = 2;

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%q%q%q%q%q%q%q%q%q%q%q%q%q%q%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[2,3,4,5,6,7,8,9,10,11,12,13,14]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Split data into numeric and cell columns.
rawNumericColumns = raw(:, [2,3,4,5,6,7,8,9,10,11,12,13,14]);
rawCellColumns = raw(:, 1);


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),rawNumericColumns); % Find non-numeric cells
rawNumericColumns(R) = {NaN}; % Replace non-numeric cells

%% Allocate imported array to column variable names
deviceName = rawCellColumns(:, 1);
deviceCameraToScreenXMm = cell2mat(rawNumericColumns(:, 1));
deviceCameraToScreenYMm = cell2mat(rawNumericColumns(:, 2));
deviceCameraXMm = cell2mat(rawNumericColumns(:, 3));
deviceCameraYMm = cell2mat(rawNumericColumns(:, 4));
devicePixelsPerInch = cell2mat(rawNumericColumns(:, 5));
deviceScreenXMm = cell2mat(rawNumericColumns(:, 6));
deviceScreenYMm = cell2mat(rawNumericColumns(:, 7));
deviceScreenWidthMm = cell2mat(rawNumericColumns(:, 8));
deviceScreenWidthPoints = cell2mat(rawNumericColumns(:, 9));
deviceScreenWidthPointsZoomed = cell2mat(rawNumericColumns(:, 10));
deviceScreenHeightMm = cell2mat(rawNumericColumns(:, 11));
deviceScreenHeightPoints = cell2mat(rawNumericColumns(:, 12));
deviceScreenHeightPointsZoomed = cell2mat(rawNumericColumns(:, 13));


%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me rawNumericColumns rawCellColumns R;