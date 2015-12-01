clear all;
close all;
warning('off', 'all');

%oldpager = PAGER('/dev/null');
%oldpso = page_screen_output(1);
%oldpoi = page_output_immediately(1);

function disp(x)
end

addpath(genpath('DeepLearnToolbox'));

PS1('>> ');
format short g;

data = load('data2.txt');
num_input = str2num(argv(){1});
num_hidden = str2num(argv(){2});
learning_rate_rbm = str2num(argv(){3});
learning_rate_bp = str2num(argv(){4});
epoch = 100;

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

%pemisah = r - ((floor(r/10) - 3) * 10);
pemisah = 30;
akhir = floor(r/10) * 10;
train_input_space = input_space(pemisah+1:akhir, :);
test_input_space = input_space(1:pemisah, :);

train_x = train_input_space(:, 2:end);
train_y = train_input_space(:, 1);

test_x = test_input_space(:, 2:end);
test_y = test_input_space(:, 1);

dbn.sizes = [num_hidden 1];
opts.numepochs = epoch;
opts.batchsize = 10;
opts.momentum = 0.5;
opts.alpha = learning_rate_rbm;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

nn = dbnunfoldtonn(dbn);
nn.activation_function = 'sigm';
nn.learningRate = learning_rate_bp;
nn.output = 'linear';

opts.numepochs = epoch;
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

%t = 1:size(hasil,1);
%t = t';
%plot(t, flipud(hasil_angka), '-+;prediction;', t, flipud(y_angka), '-*;real;');

mse = sum(power(hasil_angka - y_angka, 2)) / size(hasil_angka,1);
mse1 = sum(power(hasil - test_y, 2)) / size(hasil,1);
mse1 = mse1 * -1;
%PAGER(oldpager);
%page_screen_output(oldpso);
%page_output_immediately(oldpoi);

printf('%.15f\n', mse1);
