clear all
close all

PS1('>> ');
format long;

% Load data dari file
data = load('data.txt');
x  = data(:,1);
ei = data(:,2:7);

num_input = 20;

%normalisasi
%mi = min(x);
%ma = max(x);
%x = (x .- mi)./(ma-mi);

%mi = min(min(ei));
%ma = max(max(ei));
%for i=1:size(ei, 2)
%    ei(:,i) = (ei(:,i) .- mi) ./ (ma-mi);
%end

clear mi ma

%bikin input buat visible node
xi = zeros(size(x) - num_input, num_input);
[r c] = size(xi);

for i=1:num_input
    akhir = r+i-1;
    xi(:, i) = x(i:akhir, 1);
end

%vi = [xi ei(1:r,:)];
vi = xi

clear r c i akhir

maxepoch = 1000; % maximum number of epochs
numhid = 10;   % number of hidden units 
batchdata = zeros(1, size(vi, 2), size(vi,1)); % the data that is divided into batches (numcases numdims numbatches)
restart = 1;  % set to 1 if learning starts from beginning 

for i=1:size(batchdata, 3)
    data = vi(i,:);
    batchdata(1, :, i) = data';
end

rbm
