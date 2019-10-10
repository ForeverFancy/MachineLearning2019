clear; close all; clc

%% Load Data

n = 16087;

d = 10013;

raw_data = load("hw2.mat");

full_x = full(raw_data.X);

X = [ones(n,1) full_x];

%% Normalize the data

for i=2:d+1,
    x_max = max(X(:, i));
    x_min = min(X(:, i));
    X(:, i) = (X(:, i) - x_min)/(x_max - x_min);
end

%% Compute closed form solution.

w = inv(X' * X) * X' * (raw_data.y);

J = computeCost(X, raw_data.y, w)

save res_w1.txt w -ascii;

printf done\n
