clear all
close all

PS1('>> ');
format long;

% Load data dari file
data = load('data2.txt');
x  = data(:,1);
%ei = data(:,2:7);

num_input = 5;

%normalisasi
mi = min(x);
ma = max(x);
x = (x .- mi)./(ma-mi);

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
vi = xi(1:700,:);
tes = x(1:700);

clear r c i akhir

addpath(genpath('../DeepLearnToolbox'));

rand('state',0)
%train dbn
dbn.sizes = [30 1];
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, vi, opts);
dbn = dbntrain(dbn, vi, opts);

%figure; visualize(dbn.rbm{1}.W');

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  100;
opts.batchsize = 100;
nn = nntrain(nn, vi, tes, opts);
[er, bad] = nntest(nn, xi(701:720,:), x(701:720));

er
assert(er < 0.10, 'Too big error');
