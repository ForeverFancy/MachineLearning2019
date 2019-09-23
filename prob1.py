import numpy as np
import matplotlib.pyplot as plt


class Regression(object):

    def __init__(self, path="data", length=30):
        self.path = path
        self.length = length
        self.data = []

    def read_data(self):
        with open(self.path, "r+") as f:
            raw_data = f.readlines()
            for item in raw_data:
                self.data.append([float(i) for i in item.strip().split('\t')])
        self.x = np.array([i[0] for i in self.data])
        self.y = np.array([i[1] for i in self.data])
        # print(self.data)

    def linear_regression(self):
        sum_x, sum_y = self.x.sum(), self.y.sum()
        sum_x_square, sum_x_mutiply_y = ((self.x) ** 2).sum(), ((self.x) * (self.y)).sum()
        self.w1 = (self.length * sum_x_mutiply_y - sum_x * sum_y) / \
            (self.length * sum_x_square - sum_x ** 2)
        temp = (self.y-self.w1*self.x).sum()
        self.w0 = temp / self.length
        self.E_linear_regression = ((self.w0 + self.w1 * self.x - self.y)**2).sum()
        # print(self.w0, self.w1, self.E_linear_regression)

    def square_regression(self):
        sum_x, sum_y = self.x.sum(), self.y.sum()
        sum_x_square, sum_x_cube, sum_x_power_four = (
            self.x ** 2).sum(), (self.x ** 3).sum(), (self.x ** 4).sum()
        sum_x_y, sum_x_square_y = (self.x * self.y).sum(), (self.x * self.x * self.y).sum()

        A = np.array([[self.length, sum_x, sum_x_square],
                      [sum_x, sum_x_square, sum_x_cube],
                      [sum_x_square, sum_x_cube, sum_x_power_four]])
        B = np.array([[sum_y], [sum_x_y], [sum_x_square_y]])

        w = np.linalg.solve(A, B)
        # print(w)
        self.E_square_regression = (
            (w[0] + w[1] * self.x + w[2] * self.x ** 2 - self.y) ** 2).sum()
        self.w = w
        # print(self.E_square_regression)

    def plot(self):
        t = np.arange(-1, 1,0.1)
        l1, = plt.plot(self.x, self.y, 'o',)
        l2, = plt.plot(self.x, self.w0 + self.w1 * self.x, 'r',)
        l3, = plt.plot(t, self.w[0] + self.w[1] * t + self.w[2] * t ** 2, 'g--')
        plt.legend(handles=[l1, l2, l3], labels=['sample', 'linear', 'square'], loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def output(self):
        self.read_data()
        self.linear_regression()
        self.square_regression()
        if self.E_linear_regression < self.E_square_regression:
            print("Linear is better: w0 = %f, w1 = %f" % {self.w0, self.w1})
        else:
            w0, w1, w2 = self.w[0,0], self.w[1,0], self.w[2,0]
            print("Square is better: w0 = {}, w1 = {}, w2 = {}".format(w0,w1,w2))
        self.plot()


if __name__ == "__main__":
    reg = Regression()
    reg.output()
