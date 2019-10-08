clear; close all; clc

%% Load Data

n = 16087;

d = 10013;

ws = 0;

alpha = 0.3;

raw_data = load("hw2.mat");

full_x = full(raw_data.X);

X = [ones(n) full_x];

cost_history = [0];

iter = 0;

%% Compute using the gradient descent algorithm

while ((w - ws)' * (w - ws) >= 1E-8)
    w = w - 2 * alpha * X' * (X * w - Y) / n;
    iter = iter + 1;
    cost_history(iter) = computeCost(X, Y, w);
endwhile

i = [0:1:iter];

plot(cost_history,i, 'rx', 'MarkerSize', 10);