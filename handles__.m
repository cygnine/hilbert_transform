function[hs,pathadditions,name] = handles__()
% handles__ -- constructs path tree for the hilbert_transform module
%
% [hs,pathadditions,name] = handles__()
%
%     Returns directory pointers for common module in hs. pathadditions is a
%     cell array with a string in each element indicated paths to add to the
%     global path structure. name is a string dictating the name of the module.

name = 'hilbert_transform';

% This is by default
hs.base = fileparts(mfilename('fullpath'));

pathadditions = cell(0);
