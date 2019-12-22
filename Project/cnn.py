import numpy as np
from PIL import Image


class CNN(object):
    def __init__(self,):
        pass


class Convolution(object):
    def __init__(self, img, in_channels=3, out_channels=3, kernel_size=3, stride=1):
        self.img = img
        self.shape = img.shape[:2]
        self.in_channels = in_channels          # Normally RGB.
        # Kernel total size = kernel_size * kernel_size.
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels        # Filter num.
        self.filters = 0.01 * np.random.normal(
            0, size=(self.kernel_size, self.kernel_size, out_channels))
        self.biases = np.zeros(self.out_channels)

    def conv_forward(self):
        # Padding.
        if (self.shape[0] - self.kernel_size) % self.stride != 0:
            self.img = np.lib.pad(self.img, ((0, self.kernel_size -
                                              (self.shape[0] - self.kernel_size) % self.stride), (0, 0), (0, 0)),
                                  'constant')
        if (self.shape[1] - self.kernel_size) % self.stride != 0:
            self.img = np.lib.pad(self.img, ((0, 0), (0, self.kernel_size -
                                                      (self.shape[0] - self.kernel_size) % self.stride), (0, 0)),
                                  'constant')
        self.shape = self.img.shape[:2]
        # Initialize feature_maps, the num of feature_map is out_channels.
        feature_maps = np.zeros(
            ((self.shape[0] - self.kernel_size) // self.stride + 1, (self.shape[1] - self.kernel_size) // self.stride + 1, self.out_channels))

        # Do convolution.
        for i in range(self.out_channels):
            row, col = 0, 0
            for j in range((self.shape[0] - self.kernel_size) // self.stride + 1):
                col = 0
                for k in range((self.shape[1] - self.kernel_size) // self.stride + 1):
                    for channel in range(self.in_channels):
                        # Add each input channel.
                        # print(row,col)
                        feature_maps[j, k, i] += np.sum(np.dot(
                            self.filters[:, :, i], self.img[row: row + self.kernel_size, col: col + self.kernel_size, channel]))
                    col += self.stride
                row += self.stride
        feature_maps += self.biases
        print(feature_maps.shape)
        # Image.fromarray(np.uint8(feature_maps)).show()
        # TODO: implement im2col.
        return feature_maps

    def backward(self):
        # TODO: implement backward.
        pass


class Pooling(object):
    def __init__(self, feature_maps, pool_size=2, stride=2):
        self.feature_maps = feature_maps
        self.shape = feature_maps.shape[:2]
        self.out_channels = feature_maps.shape[-1]
        self.pool_size = pool_size
        self.stride = stride
        self.index = []

    def max_pooling_forward(self):
        # Drop out excess elements.
        pool_output = np.zeros(((self.shape[0] - self.pool_size) // self.stride + 1,
                                (self.shape[1] - self.pool_size) // self.stride + 1, self.out_channels))
        print(pool_output.shape)
        for i in range(self.out_channels):
            row = 0
            for j in range((self.shape[0] - self.pool_size) // self.stride + 1):
                col = 0
                for k in range((self.shape[1] - self.pool_size) // self.stride + 1):
                    pool_output[j, k, i] = np.max(
                        self.feature_maps[row: row + self.pool_size, col: col + self.pool_size, i])

                    self.index.append([(j, k), np.argmax(
                        self.feature_maps[row: row + self.pool_size, col: col + self.pool_size, i])])
                    col += self.stride
                row += self.stride
        # Image.fromarray(np.uint8(pool_output)).show()
        print(pool_output.shape)
        return pool_output

    def backward(self, eta):
        self.gradient = np.zeros((self.shape[0], self.shape[1], out_channels))
        for ind in self.index:
            self.gradient[ind[1]] = eta[ind[0]]
        return self.gradient
        # TODO: Not check.


class Relu(object):
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps
        self.shape = feature_maps.shape[:2]
        self.out_channels = feature_maps.shape[-1]
        self.gradient = np.ones(
            (self.shape[0], self.shape[1], self.out_channels))

    def forward(self):
        # Initialize relu function.
        relu = np.zeros((self.shape[0], self.shape[1], self.out_channels))

        for i in range(self.out_channels):
            for j in range(self.shape[0]):
                for k in range(self.shape[1]):
                    relu[j, k, i] = np.max([self.feature_maps[j, k, i], 0])
        return relu

    def backward(self):
        self.gradient[self.feature_maps < 0] = 0
        return self.gradient
        # TODO: Not check.


if __name__ == "__main__":
    img = Image.open(b"../../LFW/match pairs/0001/Aaron_Peirsol_0001.jpg")
    print(img.size)
    img = np.array(img, dtype=np.float)
    print(img.shape)
    Conv = Convolution(img, stride=2)
    con1 = Conv.conv_forward()
    pooling = Pooling(con1).max_pooling_forward()
    relu = Relu(pooling).forward()
    # Image.fromarray(np.uint8(relu)).show()
