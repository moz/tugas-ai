clear all
close all
warning('off', 'Octave:broadcast');
addpath(genpath('../DeepLearnToolbox'));

PS1('>> ');
format long;

% Load data dari file
data = load('data2.txt');
x  = data(:,1);
%ei = data(:,2:7);

num_input = 20;

%normalisasi
mi = min(x);
ma = max(x);
x = (x .- mi)./(ma-mi);

clear mi ma

%bikin input buat visible node
xi = zeros(size(x) - num_input, num_input);
[r c] = size(xi);

for i=1:num_input
    akhir = r+i-1;
    xi(:, i) = x(i:akhir, 1);
end

%vi = [xi ei(1:r,:)];
vi = xi(2:101,:);
tes = x(1:100);

clear r c i akhir


rand('state',0)
%train dbn
dbn.sizes = [50 1];
opts.numepochs =   100;
opts.batchsize = 10;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, vi, opts);
dbn = dbntrain(dbn, vi, opts);

%figure; visualize(dbn.rbm{1}.W');

%unfold dbn to nn
nn = dbnunfoldtonn(dbn);
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.output = 'linear'

%train nn
opts.numepochs =  100;
opts.batchsize = 10;
nn = nntrain(nn, vi, tes, opts);
nnpredict(nn, xi(110:115,:));
%xi(109:115)'
nn.a{end}

%nn.a{end} .- xi(109:115,)'
%[er, bad] = nntest(nn, xi(110:120,:), x(109:119));

%er
%assert(er < 0.10, 'Too big error');
