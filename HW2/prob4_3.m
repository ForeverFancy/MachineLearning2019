clear; close all; clc

%% Load Data

n = 16087;

d = 10013;

alpha = 0.6;

J_min = 0.063810;

raw_data = load("hw2.mat");

full_x = full(raw_data.X);

X = [ones(n,1) full_x];

Y = raw_data.y;

cost_history = [0];

iter = 0;

w = zeros(d+1,1);

%% Normalize the data

for i=2:d+1,
    x_max = max(X(:, i));
    x_min = min(X(:, i));
    X(:, i) = (X(:, i) - x_min)/(x_max - x_min);
end

%% Compute using the gradient descent algorithm

J = computeCost(X, Y, w)

temp1 = 2*alpha*X'*X;
temp2 = 2*alpha*X'*Y;

while ((J - J_min) >= 0.1)
    % w = w - 2 * alpha * X' * (X * w - Y) / n;
    w = w - (temp1*w - temp2)/n; 
    iter = iter + 1
    J = computeCost(X, Y, w)
    cost_history(iter) = J;
endwhile

printf iterdone\n

save res_w2.txt w -ascii;
save cost_history.txt cost_history -ascii; 

