clear all
close all

PS1('>> ');
format long;

data = load('data.txt');
x  = data(:,1);
ei = data(:,2:7);

mi = min(x);
ma = max(x);
x = (x .- mi)./(ma-mi);

mi = min(min(ei));
ma = max(max(ei));
for i=1:size(ei, 2)
    ei(:,i) = (ei(:,i) .- mi) ./ (ma-mi);
end

clear mi ma

