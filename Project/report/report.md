## 卷积高效实现

如果使用传统的四层循环进行卷积计算，需要多次重复进行小规模矩阵乘法运算，难以充分利用计算资源。于是我们进行调研后决定采用 im2col 方法对卷积运算进行加速。这种方法的大体思想是将待卷积的矩阵进行一定的变换，之后和卷积核进行一次矩阵相乘操作即可，能够极大加速卷积的运算过程。下面用例子说明 im2col 的思想。

假设现在有一个 3*3 的待卷积矩阵 $X$，2\*2 的卷积核 $Y$，如下： 
$$
X=\left[\begin{array}{lll}
{1} & {2} & {3} \\
{4} & {5} & {6} \\
{7} & {8} & {9}
\end{array}\right]\\
Y=\left[\begin{array}{ll}
{10} & {11} \\
{12} & {13} \\
\end{array}\right]
$$
假设卷积步长为 1，我们希望将 $X$ 转换为如下形式：
$$
X'=\begin{bmatrix}
\begin{bmatrix}
1 \ 2\\
4 \ 5
\end{bmatrix}&
\begin{bmatrix}
2 \ 3\\
5 \ 6
\end{bmatrix}\\
\begin{bmatrix}
4 \ 5\\
7 \ 8
\end{bmatrix}&
\begin{bmatrix}
5 \ 6\\
8 \ 9
\end{bmatrix}
\end{bmatrix}
$$
这样 $X'$ 的每个元素只需要与卷积核进行点乘（使用 `tensordot` 进行张量点乘进行快速运算）即可得到最后的卷积结果。

为了方便，我们使用 `as_strided` 函数进行快速变换，下面举例说明该函数的使用方法。

`as_strided` 函数需要指定 `shape, strides` 这两个参数。`shape` 为输出矩阵的形状，可以看出这时期望输出的 $X$ 的形状为 `((img_h - kernel_h) // stride + 1, (img_w - kernel_w) // stride + 1, kernel_h, kernel_w)` 。`strides` 参数为输出矩阵跨越数组各个维度在原来内存中经过的元素数，以 $X$ 为例，$X$ 的 `stride` 为 `(1, 3)`，对于 $X'$ 来说，$X'$ 的 `strides` 为 `(1, 3, 1, 3)` ，容易看出分别为 `(X.stride[0], X.stride[1], X.stride[0], X.stride[1])`。同理可以推知，如果 `stride` 不是 1，那么只需要将上面的 `strides` 改为 `(X.stride[0] * stride, X.stride[1] * stride, X.stride[0], X.stride[1])`，如果同时有多个输入矩阵，每个输入矩阵有多个 channel，那么同理只需要将 `shape` 改为 `(n, (img_h - kernel_h) // stride + 1, (img_w - kernel_w) // stride + 1, kernel_h, kernel_w, c)`，`strides` 改为 `(X.stride[0], X.stride[1] * stride, X.stride[2] * stride, X.stride[1], X.stride[2], s[3])`即可。代码如下：

```python
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
```

## 前向传播

使用 im2col 处理输入之后，将输入与卷积核进行点乘操作即可。

## 反向传播

反向传播分为两部分：

### 对本层卷积层的梯度

本层卷积核权重的梯度为使用前面一层传入的梯度对本层的输入矩阵进行卷积得到的结果，可以类似使用 im2col 等方法求解。

本层卷积层偏移 (bias) 的梯度即为对应的每个前一层输入梯度求和。

### 向前传播的梯度

向前传播的梯度需要使用全卷积实现，将输入梯度上下左右分别填充 `kernel_size - 1` 个 0，并且相邻元素之间填充 `stride - 1` 个 0，之后使用 180° 翻转之后的卷积核对其进行卷积操作，得到对向前传播的梯度。

## 参考文献

[1. 卷积算法的另一种高效实现。](https://zhuanlan.zhihu.com/p/64933417)

