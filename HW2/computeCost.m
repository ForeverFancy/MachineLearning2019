function J = computeCost(X, Y, w)

n = length(Y);

J = 0;

J = (Y - X * w)' * (Y - X * w)/n;

end