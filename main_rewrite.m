clear all;
close all;
warning('off', 'Octave:broadcast');

addpath(genpath('../DeepLearnToolbox'));

PS1('>> ');
format short g;

data = load('data2.txt');
num_input = 20;
num_hidden = 20;

mi = min(data);
di = max(data) - mi;
data_normal = (data .- mi) ./ di;

space_count = num_input + 1;
input_space = zeros(size(data_normal) - space_count, space_count);
[r c] = size(input_space);

for i=1:space_count
    akhir = r+i-1;
    input_space(:, i) = data_normal(i:akhir, 1);
end

pemisah = r - ((floor(r/10) - 3) * 10);
train_input_space = input_space(pemisah+1:end, :);
test_input_space = input_space(1:pemisah, :);

train_x = train_input_space(:, 2:end);
train_y = train_input_space(:, 1);

test_x = test_input_space(:, 2:end);
test_y = test_input_space(:, 1);

dbn.sizes = [num_hidden 1];
opts.numepochs = 100;
opts.batchsize = 10;
opts.momentum = 0.5;
opts.alpha = 0.1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

nn = dbnunfoldtonn(dbn);
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.output = 'linear';

opts.numepochs = 100;
opts.batchsize = 1;
nn = nntrain(nn, train_x, train_y, opts);

hasil = [];
nn.testing = 1;
for i=1:size(test_y,1)
    nn = nnff(nn, test_x(i, :), zeros(1,1));
    hasil = [hasil; nn.a{3}];
end
nn.testing = 0;

hasil_angka = (hasil .* di) .+ mi;
y_angka = (test_y .* di) .+ mi;

t = 1:size(hasil,1);
t = t';
plot(t, flipud(hasil_angka), '-+;prediction;', t, flipud(y_angka), '-+;real;');

mse = sum(power(hasil_angka - y_angka, 2)) / size(hasil_angka,1)
mse1 = sum(power(hasil - test_y, 2)) / size(hasil,1)
