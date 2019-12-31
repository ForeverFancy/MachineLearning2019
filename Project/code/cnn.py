import numpy as np
from PIL import Image
import losses
import other


class CNN(object):
    def __init__(self,):
        pass

def im2col_enhanced(im: torch.Tensor, kernel_size, stride, inner_stride=(1, 1)) -> torch.Tensor:
    kh, kw = kernel_size
    sh, sw = stride
    ish, isw = inner_stride
    b, h, w, c = im.shape
    assert (h - kh * re) % sh == 0
    assert (w - kw * isw) % sw == 0
    out_h = (h - kh * ish) // sh + 1
    out_w = (w - kw * isw) // sw + 1
    out_size = (b, out_h, out_w, kh, kw, c)
    s = im.stride()
    out_stride = (s[0], s[1] * sh, s[2] * sw, s[1] * ish, s[2] * isw, s[3])
    col_img = im.as_strided(size=out_size, stride=out_stride)
    return col_img

class Convolution(object):
    def __init__(self, input_shape, out_channel, kernel_size, stride, learning_rate, activate_func: str = None):
        self.in_h, self.in_w, self.in_channel = input_shape
        self.learning_rate = learning_rate
        self.out_channel = out_channel
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        # ignore padding
        assert (self.in_h - self.kernel_h) % self.stride_h == 0
        assert (self.in_w - self.kernel_w) % self.stride_w == 0
        self.out_h = (self.in_h - self.kernel_h) // self.stride_h + 1
        self.out_w = (self.in_w - self.kernel_w) // self.stride_w + 1

        self.filters = torch.randn(
            (self.kernel_h, self.kernel_w, self.in_channel, out_channel),
            dtype=floatX)
        self.biases = torch.randn((self.out_channel,), dtype=floatX)
        self.filters_gradient = torch.empty(
            (self.kernel_h, self.kernel_w, self.in_channel, out_channel),
            dtype=floatX)

        if activate_func == 'relu':
            self.activation = other.Relu()
        elif activate_func == 'sigmoid':
            self.activation = other.Sigmoid(100)
        elif activate_func == 'tanh':
            self.activation = other.Tanh(100)
        else:
            self.activation = None

    # 已通过测试
    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, self.in_h, self.in_w, self.in_channel)
        self.in_data = in_data
        col_img = im2col_enhanced(in_data, (self.kernel_h, self.kernel_w), (self.stride_h, self.stride_w))
        feature_maps = torch.tensordot(col_img, self.filters, dims=[(3, 4, 5), (0, 1, 2)]) \
                       + self.biases.reshape((1, 1, 1, self.out_channel))
        if self.activation is not None:
            feature_maps = self.activation.forward(feature_maps)
        return feature_maps

    # 测试通过
    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.shape == (self.batch, self.out_h, self.out_w, self.out_channel)
        if self.activation is not None:
            eta = self.activation.backward(eta)
        # filters 梯度
        col_img = im2col_enhanced(self.in_data, (self.kernel_h, self.kernel_w), (self.stride_h, self.stride_w))
        self.filters_gradient[:, :, :, :] = 0
        for b in range(self.batch):
            self.filters_gradient += torch.tensordot(
                col_img[b], eta[b], dims=[(0, 1), (0, 1)]
            )
        # biases 梯度
        biases_gradient = eta.sum(dim=(0, 1, 2))
        # in_data 梯度
        # 这部分的实现参照 PPT
        padding_eta = torch.zeros(
            (self.batch,
             2 * (self.kernel_h - 1) + (self.out_h - 1) * self.stride_h + 1,
             2 * (self.kernel_w - 1) + (self.out_w - 1) * self.stride_w + 1,
             self.out_channel), dtype=floatX)
        pad_h = self.kernel_h - 1
        pad_w = self.kernel_w - 1
        padding_eta[:, pad_h:-pad_h:self.stride_h, pad_w:-pad_w:self.stride_w, :] = eta  # padding_eta 其他部分为0
        filters_flip = self.filters.flip(dims=(0, 1))
        # 进行卷积运算
        col_eta = im2col_enhanced(padding_eta, (self.kernel_h, self.kernel_w), (1, 1))
        assert col_eta.shape == (self.batch, self.in_h, self.in_w, self.kernel_h, self.kernel_w, self.out_channel)
        next_eta = torch.tensordot(col_eta, filters_flip, dims=[(3, 4, 5), (0, 1, 3)])
        assert next_eta.shape == (self.batch, self.in_h, self.in_w, self.in_channel)
        # 更新
        self.filters -= self.learning_rate * self.filters_gradient
        self.biases -= self.learning_rate * biases_gradient
        return next_eta


class Pooling(object):
    def __init__(self, input_shape, pool_size=3, stride=1):               
        self.batch_size = input_shape[0]                   # (n, h, w, c)
        self.shape = input_shape[1:3]
        self.out_channels = input_shape[-1]
        self.pool_size = pool_size
        self.stride = stride
        self.gradient_out = np.zeros(
            (self.batch_size, self.shape[0], self.shape[1], self.out_channels))

    def forward(self, feature_maps):
        self.feature_maps = feature_maps
        print(feature_maps.shape)
        pool_output = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1] // self.pool_size, self.pool_size, feature_maps.shape[2] // self.pool_size, self.pool_size, feature_maps.shape[3]).max(axis=(2, 4))
        self.index = pool_output.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2) == feature_maps
        return pool_output

    def gradient(self, eta):
        self.gradient_out = eta.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2) * self.index
        return self.gradient_out


class Relu(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.gradient_out = np.ones(input_shape)

    def forward(self, feature_maps):
        self.feature_maps = feature_maps
        relu = np.maximum(feature_maps, 0)
        print("relu", relu.shape)
        return relu

    def gradient(self):
        print("gradient",self.gradient_out.shape)
        self.gradient_out[self.feature_maps < 0] = 0
        return self.gradient_out
        # TODO: Not check.


if __name__ == "__main__":
    img = Image.open(b"../../../LFW/match pairs/0001/Aaron_Peirsol_0001.jpg")
    # print(img.size)
    img = np.array(img, dtype=np.float32)
    # print(img.shape)
    z = np.random.rand(10, 51, 51, 3).astype(np.float32)
    # print(z)
    Conv1 = Convolution(z.shape, out_channels=4, kernel_size=2, stride=1, learning_rate=0.00001)
    out1 = Conv1.forward(z)
    pooling1 = Pooling(out1.shape, pool_size=2)
    out2 = pooling1.forward(out1)
    Conv2 = Convolution(out2.shape, out_channels=5 ,kernel_size=2, stride=1, learning_rate=0.00001)
    y_pred = Conv2.forward(out2)
    print(y_pred.shape)
    y_true = np.ones((10, 24, 24, 5)).astype(np.float32)
    for i in range(10):
        error, dy = losses.mean_squared_error(y_pred, y_true)
        print(error)
        print("------")
        eta3to2 = Conv2.gradient(dy)
        Conv2.backward()
        eta2to1 = pooling1.gradient(eta3to2)
        _ = Conv1.gradient(eta2to1)
        Conv1.backward()
        out1 = Conv1.forward(z)
        out2 = pooling1.forward(out1)
        y_pred = Conv2.forward(out2)
