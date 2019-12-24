import numpy as np
from PIL import Image
import losses


class CNN(object):
    def __init__(self,):
        pass


class Convolution(object):
    def __init__(self, input_shape, out_channels=3, kernel_size=3, stride=1, learning_rate=0.0006):
        self.input_shape = input_shape      # (batch_size, in_channels, h, w)
        self.in_channels = input_shape[-1]  # Normally RGB.
        self.batch_size = input_shape[0]
        # Kernel total size = kernel_size * kernel_size.
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels  # Filter num.
        # input channels -> output channels
        self.filters = 0.01 * np.random.normal(
            0, size=(self.kernel_size, self.kernel_size, self.in_channels, out_channels)).astype(np.float32)
        self.biases = np.zeros(self.out_channels)
        self.w_gradient = np.zeros(self.filters.shape)
        self.eta_forward = np.zeros(self.input_shape[1:])  # TODO: Check this.
        self.learning_rate = learning_rate

    def conv_forward(self, img):
        # Update img.
        self.img = img
        self.padding_img = self.img
        # Padding.
        if (self.input_shape[2] - self.kernel_size) % self.stride != 0:
            self.padding_img = np.lib.pad(self.img, ((0, 0), (0, self.kernel_size -
                                                              (self.input_shape[2] - self.kernel_size) % self.stride), (0, 0), (0, 0)),
                                          'constant')
        if (self.input_shape[3] - self.kernel_size) % self.stride != 0:
            self.padding_img = np.lib.pad(self.img, ((0, 0), (0, 0), (0, self.kernel_size -
                                                                      (self.input_shape[3] - self.kernel_size) % self.stride), (0, 0)),
                                          'constant')
        print(self.padding_img.shape)
        self.input_shape = self.padding_img.shape
        # Initialize feature_maps, the num of feature_map is out_channels.
        feature_maps = np.zeros(
            (self.batch_size, (self.input_shape[1] - self.kernel_size) // self.stride + 1, (self.input_shape[2] - self.kernel_size) // self.stride + 1, self.out_channels))

        # Do convolution.
        col_img = self.im2col(self.padding_img.shape, self.kernel_size, self.padding_img)
        feature_maps = np.tensordot(col_img, self.filters, axes=[(3, 4, 5), (0, 1, 2)]) + self.biases
        # print(feature_maps.shape)
        # Image.fromarray(np.uint8(feature_maps)).show()
        return feature_maps

    def gradient(self, eta):
        # eta shape = (n, h, w, c).
        h,w = eta.shape[1], eta.shape[2]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.w_gradient[i, j, :, :] = np.tensordot(eta, self.padding_img[:, i: i + h, j: j + w, :], axes=([0, 1, 2], [0, 1, 2]))

        # print(self.w_gradient)
        self.b_gradient = np.array([np.sum(eta[:, :, :, i])
                                    for i in range(self.out_channels)])
        flip_filters = np.flip(self.filters, (0, 1))
        padding_eta = np.lib.pad(eta, ((0,0), (self.kernel_size - 1, self.kernel_size - 1),
                                       (self.kernel_size - 1, self.kernel_size - 1), (0, 0)), 'constant')
        col_eta = self.im2col(padding_eta.shape, self.kernel_size, padding_eta)
        self.eta_forward = np.tensordot(
            col_eta, flip_filters, axes=[(3, 4, 5), (0, 1, 2)])

        
        return self.eta_forward

    def backward(self):
        self.filters -= self.learning_rate * self.w_gradient
        self.biases -= self.learning_rate * self.b_gradient

    def im2col(self, shape, ks, image):
        N, H, W, C = shape
        out_h = (H - ks) // self.stride + 1
        out_w = (W - ks) // self.stride + 1
        out_shape = (N, out_h, out_w, ks, ks, C)
        strides = (image.strides[0], image.strides[1]*self.stride, image.strides[2]*self.stride, *image.strides[1:])
        col_img = np.lib.stride_tricks.as_strided(image, shape=out_shape, strides=strides)
        return col_img


class Pooling(object):
    def __init__(self, feature_maps, pool_size=3, stride=1):
        self.feature_maps = feature_maps
        self.shape = feature_maps.shape[:2]
        self.out_channels = feature_maps.shape[-1]
        self.pool_size = pool_size
        self.stride = stride
        self.index = []
        self.gradient_out = np.zeros(
            (self.shape[0], self.shape[1], self.out_channels))

    def max_pooling_forward(self, feature_maps):
        self.feature_maps = feature_maps
        # Drop out excess elements.
        pool_output = np.zeros(((self.shape[0] - self.pool_size) // self.stride + 1,
                                (self.shape[1] - self.pool_size) // self.stride + 1, self.out_channels))
        # print(pool_output.shape)
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
        # print(pool_output.shape)
        return pool_output

    def gradient(self, eta):
        for ind in self.index:
            self.gradient_out[ind[1]] = eta[ind[0]]
        return self.gradient_out


class Relu(object):
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps
        self.shape = feature_maps.shape[:2]
        self.out_channels = feature_maps.shape[-1]
        self.gradient = np.ones(
            (self.shape[0], self.shape[1], self.out_channels))

    def forward(self, feature_maps):
        self.feature_maps = feature_maps
        # Initialize relu function.
        relu = np.zeros((self.shape[0], self.shape[1], self.out_channels))

        for i in range(self.out_channels):
            for j in range(self.shape[0]):
                for k in range(self.shape[1]):
                    relu[j, k, i] = np.max([self.feature_maps[j, k, i], 0])
        return relu

    def gradient(self):
        self.gradient[self.feature_maps < 0] = 0
        return self.gradient
        # TODO: Not check.


if __name__ == "__main__":
    img = Image.open(b"../../LFW/match pairs/0001/Aaron_Peirsol_0001.jpg")
    # print(img.size)
    img = np.array(img, dtype=np.float32)
    # print(img.shape)
    z = np.random.rand(10, 50, 50, 3).astype(np.float32)
    # print(z)
    Conv1 = Convolution(z.shape, kernel_size=2, stride=1, learning_rate=0.00001)
    out1 = Conv1.conv_forward(z)
    # pooling1 = Pooling(out1)
    # out2 = pooling1.max_pooling_forward(out1)
    Conv2 = Convolution(out1.shape, kernel_size=2, stride=1, learning_rate=0.00001)
    y_pred = Conv2.conv_forward(out1)
    print(y_pred.shape)
    y_true = np.ones((10, 48, 48, 3)).astype(np.float32)
    for i in range(10):
        error, dy = losses.mean_squared_error(y_pred, y_true)
        print(error)
        print("------")
        eta2to1 = Conv2.gradient(dy)
        Conv2.backward()
        # eta2to1 = pooling1.gradient(eta3to2)
        _ = Conv1.gradient(eta2to1)
        Conv1.backward()
        out1 = Conv1.conv_forward(z)
        # out2 = pooling1.max_pooling_forward(out1)
        y_pred = Conv2.conv_forward(out1)
