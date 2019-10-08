clear; close all; clc

%% Load Data

n = 16087;

d = 10013;

raw_data = load("hw2.mat");

full_x = full(raw_data.X);

X = [ones(n) full_x];

%% Compute closed form solution.

w = zeros(d);

w = inv(X' * X) * X' * raw_data.Y;

