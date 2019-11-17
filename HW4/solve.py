import numpy as np
from cvxopt import matrix, solvers

def solve():
    x1 = np.array([2, 3])
    x2 = np.array([1, 2])
    x3 = np.array([1, 3])
    x4 = np.array([2, 2])
    x = [x1, x2, x3, x4]
    y = [1, 1, -1, -1]
    P = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            print(x[i]*x[j])
            P[i][j] = np.sum(x[i] * x[j]) * y[i] * y[j]
    print(P)
    P = matrix(P.transpose())
    q = matrix([-1.0, -1.0, -1.0, -1.0])
    G = matrix([[-1.0, 0, 0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, 1, 0, 0], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, -1, 0, 0, 0, 1]])
    A = matrix([1, 1, -1, -1.0],(1,4))
    b = matrix([0.0])
    h = matrix([0, 0, 0, 0, 10.0, 10.0, 10.0, 10.0])
    print(P, q, G, A, b, h)
    result = solvers.qp(P, q, G, h, A, b)
    print("Result:", result['x'])

if __name__ == "__main__":
    solve()