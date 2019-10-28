import numpy as np
import scipy.io
import math


def read_data():
    train_imgs = scipy.io.loadmat('./hw3_lr/train_imgs.mat')
    train_labels = scipy.io.loadmat('./hw3_lr/train_labels.mat')
    test_imgs = scipy.io.loadmat('./hw3_lr/test_imgs.mat')
    test_labels = scipy.io.loadmat('./hw3_lr/test_labels.mat')

    train_imgs = np.matrix(train_imgs)[0, 0]['train_img'].toarray()
    train_labels = np.matrix(train_labels)[0, 0]['train_label'].toarray()

    test_imgs = np.matrix(test_imgs)[0, 0]['test_img'].toarray()
    test_labels = np.matrix(test_labels)[0, 0]['test_label'].toarray()

    return train_imgs, train_labels, test_imgs, test_labels


def preprocess(imgs):
    num = imgs.shape[0]
    for i in range(num):
        max_val = imgs[i, :].max()
        min_val = imgs[i, :].min()
        if max_val == min_val:
            continue
        imgs[i, :] = (imgs[i, :] - min_val) / (max_val - min_val)

    return imgs


def train_model(train_imgs, train_labels, learning_rate=0.5, max_iters=500, verbose=10):
    theta = np.full(
        shape=(train_imgs.shape[1]+1, 1), fill_value=0, dtype=np.float64)

    temp = np.ones((train_imgs.shape[0], 1))

    X = np.c_[temp, train_imgs].astype(np.float64)

    y = train_labels.transpose()

    count = 0
    while True:
        count += 1
        J, gradient = loss_function_and_gradients(X, theta, y)
        if count % verbose == 0:
            print("Iter: ", count, " J = ", J)
            print("gradient:", math.sqrt(sum(np.power(gradient, 2))))
        theta = theta - learning_rate * gradient
        if count >= max_iters:
            break
    return theta


def loss_function_and_gradients(X: np.matrix, theta: np.matrix, y: np.matrix):
    m = y.shape[0]
    hx = np.exp(-X @ theta)

    J = np.sum(np.log(1 / (1 + hx)) * (-y) -
               (1 - y) * np.log(hx / (1 + hx))) / m

    gradient = (X.transpose() @ (1 / (1 + hx) - y)) / m
    return J, gradient


def predict(theta, test_imgs, test_labels):
    m = test_imgs.shape[0]

    temp = np.ones((test_imgs.shape[0], 1))

    test_imgs = np.c_[temp, test_imgs].astype(np.float64)

    res = (test_imgs @ theta > 0)

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(m):
        if res[i] == True:
            if test_labels[0][i] == 2:
                tp += 1
            else:
                fp += 1
        else:
            if test_labels[0][i] == 2:
                fn += 1
            else:
                tn += 1

    P = tp/(tp+fp)
    R = tp/(tp+fn)
    F1 = 2/(1/P + 1/R)

    print("tp: %d\nfp: %d\nfn: %d\ntn: %d" % (tp, fp, fn, tn))
    print("P:", P)
    print("R:", R)
    print("F1 score:", F1)


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels = read_data()
    train_imgs = preprocess(train_imgs.astype(np.float64))
    test_imgs = preprocess(test_imgs.astype(np.float64))
    theta = train_model(train_imgs, train_labels - 1, 6, 100, 10)  # to make the labels to become 0/1
    predict(theta, test_imgs, test_labels)
