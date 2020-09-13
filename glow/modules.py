import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn


class ActNorm2d(nn.Cell):
    def __init__(self, num_features, pixels, scale=1., init=False, reverse=False):
        super(ActNorm2d, self).__init__()
        # register mean and scale
        self.pixels = Tensor(np.array([[[[pixels]]]]).astype(np.float32))
        self.scale = Tensor(np.array([[[[scale]]]]).astype(np.float32))
        self.reverse = reverse
        size = [1, num_features, 1, 1]
        self.bias = Parameter((initializer("zeros", size)), name="bias")
        self.logs = Parameter((initializer("zeros", size)), name="logs")
        # mindspore operator
        self.exp = P.Exp()
        self.sum = P.ReduceSum(True)
        self.sum_axis = (0, 1, 2, 3)
        self.add = P.TensorAdd()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.neg = P.Neg()
        self.init = init
        self.mean = P.ReduceMean(True)
        self.mean_axis = (0, 2, 3)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.div = P.RealDiv()
        self.epsilon = Tensor(np.array([[[[1e-6]]]]).astype(np.float32))
        self.log = P.Log()
        self.assign = P.AssignAdd()

    def construct(self, input, logdet):
        if self.init:
            vars = self.neg(self.mean(input, self.mean_axis))
            self.assign(self.bias, vars)
            h = self.square(self.add(input, vars))
            h = self.mean(h, self.mean_axis)
            h = self.add(self.sqrt(h), self.epsilon)
            h = self.div(self.scale, h)
            h = self.log(h)
            self.assign(self.logs, h)
            input = self.add(input, vars)
            h = self.exp(h)
            input = self.mul(input, h)
            h = self.sum(h, self.sum_axis)
            dlogdet = self.mul(h, self.pixels)
            logdet = self.add(logdet, dlogdet)
            return input, logdet
        else:
            if not self.reverse:
                input = self.add(input, self.bias)
                h = self.exp(self.logs)
                input = self.mul(input, h)
                h = self.sum(self.logs, self.sum_axis)
                dlogdet = self.mul(h, self.pixels)
                logdet = self.add(logdet, dlogdet)
                return input, logdet
            else:
                input = self.mul(input, self.exp(self.neg(self.logs)))
                dlogdet = self.mul(self.sum(self.logs, self.sum_axis), self.pixels)
                dlogdet = self.neg(dlogdet)
                logdet = self.add(logdet, dlogdet)
                input = self.sub(input, self.bias)
                return input, logdet


class ConvActNorm2d(nn.Cell):
    def __init__(self, num_features, pixels, scale=1., init=False, reverse=False):
        super(ConvActNorm2d, self).__init__()
        # register mean and scale
        self.pixels = Tensor(np.array([[[[pixels]]]]).astype(np.float32))
        self.scale = Tensor(np.array([[[[scale]]]]).astype(np.float32))
        self.reverse = reverse
        size = [1, num_features, 1, 1]
        self.bias = Parameter((initializer("zeros", size)), name="bias")
        self.logs = Parameter((initializer("zeros", size)), name="logs")
        # mindspore operator
        self.exp = P.Exp()
        self.sum = P.ReduceSum(True)
        self.sum_axis = (0, 1, 2, 3)
        self.add = P.TensorAdd()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.neg = P.Neg()
        self.init = init
        self.mean = P.ReduceMean(True)
        self.mean_axis = (0, 2, 3)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.div = P.RealDiv()
        self.epsilon = Tensor(np.array([[[[1e-6]]]]).astype(np.float32))
        self.log = P.Log()
        self.assign = P.AssignAdd()

    def construct(self, input):
        if self.init:
            vars = self.neg(self.mean(input, self.mean_axis))
            self.assign(self.bias, vars)
            h = self.square(self.add(input, vars))
            h = self.mean(h, self.mean_axis)
            h = self.add(self.sqrt(h), self.epsilon)
            h = self.div(self.scale, h)
            h = self.log(h)
            self.assign(self.logs, h)
            input = self.add(input, vars)
            h = self.exp(h)
            input = self.mul(input, h)
            return input
        else:
            if not self.reverse:
                input = self.add(input, self.bias)
                h = self.exp(self.logs)
                input = self.mul(input, h)
                return input
            else:
                input = self.mul(input, self.exp(self.neg(self.logs)))
                input = self.sub(input, self.bias)
                return input


class LinearZeros(nn.Cell):
    def __init__(self, in_channels, out_channels, logscale_factor=3.0):
        super(LinearZeros, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.logscale_factor = Tensor((np.ones([1, out_channels]) * logscale_factor).astype(np.float32))
        self.weight = Parameter(initializer("zeros", [out_channels, in_channels]), name="weight")
        self.bias = Parameter(initializer("zeros", [1, out_channels]), name="bias")
        self.logs = Parameter(initializer("zeros", [1, out_channels]), name="logs")
        # mindspore operators
        self.matmul = P.MatMul()
        self.mul = P.Mul()
        self.exp = P.Exp()
        self.add = P.TensorAdd()
        self.transpose = P.Transpose()

    def construct(self, input):
        h = self.transpose(self.weight, (1, 0))
        h = self.matmul(input, h)
        output = self.add(h, self.bias)
        h = self.mul(self.logs, self.logscale_factor)
        h = self.exp(h)
        h = self.mul(output, h)
        return h


class Conv2d(nn.Cell):
    pad_dict = {
        "pad": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, pixels, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 pad_mode='pad', do_actnorm=True, weight_std=0.05, init=False, reverse=False):
        padding = Conv2d.get_padding(pad_mode, kernel_size, stride)
        kernel_size = tuple(kernel_size)
        padding = tuple(padding)
        stride = tuple(stride)
        super(Conv2d, self).__init__()
        self.pixels = pixels
        self.reverse = reverse
        self.init = init
        self.actnorm = ConvActNorm2d(out_channels, pixels=self.pixels, init=self.init, reverse=self.reverse)
        self.weight = initializer(Tensor(np.random.normal(0, weight_std, [out_channels, in_channels, kernel_size[0], kernel_size[1]]).astype(np.float32)),
                                  [out_channels, in_channels, kernel_size[0], kernel_size[1]])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride[0], pad_mode,
                                padding[0], has_bias=(not do_actnorm), weight_init=self.weight)

    def construct(self, input):
        x = self.conv2d(input)
        x = self.actnorm(x)
        return x


class Conv2dZeros(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], pad_mode="pad", logscale_factor=3.0):
        padding = Conv2d.get_padding(pad_mode, kernel_size, stride)
        kernel_size = tuple(kernel_size)
        padding = tuple(padding)
        stride = tuple(stride)
        super(Conv2dZeros, self).__init__()
        # logscale_factor
        self.logscale_factor = Tensor((np.ones([1, out_channels, 1, 1]) * logscale_factor).astype(np.float32))
        self.logs = Parameter((initializer("zeros", [1, out_channels, 1, 1])), name="logs")
        self.weight = initializer("zeros", [out_channels, in_channels, kernel_size[0], kernel_size[1]])
        self.bias = Parameter((initializer("zeros", [1, out_channels, 1, 1])), name="bias")
        # mindspore operators
        self.exp = P.Exp()
        self.mul = P.Mul()
        self.add = P.TensorAdd()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride[0], pad_mode, padding[0], weight_init=self.weight)

    def construct(self, input):
        conv_output = self.conv(input)
        conv_output = self.add(conv_output, self.bias)
        h = self.mul(self.logs, self.logscale_factor)
        h = self.exp(h)
        output = self.mul(conv_output, h)
        return output


class Permute2d(nn.Cell):
    def __init__(self, input_shape, num_channels, shuffle, reverse=False):
        super(Permute2d, self).__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.reverse = reverse
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.int32)
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.int32)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()
        self.shuffled_indices = Parameter(Tensor(self.indices), name="shuffled_indices", requires_grad=False)
        self.inversed_indices = Parameter(Tensor(self.indices_inverse), name="inversed_indices", requires_grad=False)
        self.perm = (1, 0, 2, 3)
        self.transpose = P.Transpose()
        self.gather = P.GatherV2()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def construct(self, input):
        if not self.reverse:
            h = self.transpose(input, self.perm)
            h = self.gather(h, self.shuffled_indices, 0)
            output = self.transpose(h, self.perm)
        else:
            h = self.transpose(input, self.perm)
            h = self.gather(h, self.inversed_indices, 0)
            output = self.transpose(h, self.perm)
        return output


class GaussianDiagLogp(nn.Cell):
    def __init__(self, Channels):
        super(GaussianDiagLogp, self).__init__()
        self.log2pi = Tensor((np.ones([1, 1, 1, 1]) * np.log(2 * np.pi)).astype(np.float32))
        # mindspore operators
        self.exp = P.Exp()
        self.sum = P.ReduceSum(True)
        self.sum_axis = (1, 2, 3)
        self.mul = P.Mul()
        self.add = P.TensorAdd()
        self.div = P.RealDiv()
        self.sub = P.Sub()
        self.square = P.Square()
        self.const_1 = Tensor((np.ones([1, 1, 1, 1]) * (-0.5)).astype(np.float32))
        self.const_2f = Tensor((np.ones([1, 1, 1, 1]) * 2).astype(np.float32))

    def construct(self, mean, logs, x):
        if mean is None and logs is None:
            h = self.square(x)
            h = self.add(h, self.log2pi)
            h = self.mul(self.const_1, h)
            h = self.sum(h, self.sum_axis)
            likelihood = h
        else:
            logs_2 = self.mul(logs, self.const_2f)
            x_mean = self.sub(x, mean)
            x_mean_2 = self.square(x_mean)
            exp_h = self.exp(logs_2)
            div_h = self.div(x_mean_2, exp_h)
            h = self.add(logs_2, div_h)
            h = self.add(h, self.log2pi)
            h = self.mul(self.const_1, h)
            h = self.sum(h, self.sum_axis)
            likelihood = h
        return likelihood


class GaussianDiagSample(nn.Cell):
    def __init__(self, shape, eps_std=None):
        super(GaussianDiagSample, self).__init__()
        self.shape = shape
        self.eps_std = eps_std or 1
        # mindspore operators
        self.random_normal = P.RandomNormal(0)
        self.exp = P.Exp()
        self.mul = P.Mul()
        self.add = P.TensorAdd()
        self.size = Tensor(np.array(self.shape).astype(np.int32))
        self.mean = Tensor(np.zeros(self.shape).astype(np.float32))
        self.stddev = Tensor(self.eps_std * np.ones(self.shape).astype(np.float32))

    def construct(self, mean, logs):
        eps = self.random_normal(self.size, self.mean, self.stddev)
        return self.add(mean, self.mul(self.exp(logs), eps))


class Split2d(nn.Cell):
    def __init__(self, input_shape, num_channels, eps_std=None, reverse=False):
        super(Split2d, self).__init__()
        self.input_shape = tuple(input_shape)
        self.num_channels = num_channels
        self.reverse = reverse
        self.c_slice = self.num_channels // 2
        self.conv2dzeros = Conv2dZeros(num_channels // 2, num_channels)
        if not self.reverse:
            self.gauss_logp = GaussianDiagLogp(num_channels // 2)
        else:
            self.eps_std = eps_std
            self.gauss_sample = GaussianDiagSample((input_shape[0], input_shape[1] // 2, input_shape[2], input_shape[3]), self.eps_std)
        # mindspore operators
        self.cat = P.Concat(axis=1)
        self.add = P.TensorAdd()
        self.slice_z1_begin = (0, 0, 0, 0)
        self.slice_z1_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z1_slice = Slice(self.slice_z1_begin, self.slice_z1_size)
        self.slice_z2_begin = (0, self.input_shape[1] // 2, 0, 0)
        self.slice_z2_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z2_slice = Slice(self.slice_z2_begin, self.slice_z2_size)
        self.strided_slice_mean_begin = (0, 0, 0, 0)
        self.strided_slice_mean_end = self.input_shape
        self.strided_slice_mean_stride = (1, 2, 1, 1)
        self.mean_strided_slice = StridedSlice(self.strided_slice_mean_begin, self.strided_slice_mean_end, self.strided_slice_mean_stride)
        self.strided_slice_logs_begin = (0, 1, 0, 0)
        self.strided_slice_logs_end = self.input_shape
        self.strided_slice_logs_stride = (1, 2, 1, 1)
        self.logs_strided_slice = StridedSlice(self.strided_slice_logs_begin, self.strided_slice_logs_end, self.strided_slice_logs_stride)

    def construct(self, input, logdet):
        if not self.reverse:
            z1 = self.z1_slice(input)
            z2 = self.z2_slice(input)
            h = self.conv2dzeros(z1)
            mean = self.mean_strided_slice(h)
            logs = self.logs_strided_slice(h)
            h = self.gauss_logp(mean, logs, z2)
            logdet = self.add(h, logdet)
            return z1, logdet
        else:
            z1 = input
            h = self.conv2dzeros(z1)
            mean = self.mean_strided_slice(h)
            logs = self.logs_strided_slice(h)
            z2 = self.gauss_sample(mean, logs)
            z = self.cat((z1, z2))
            return z, logdet


class SqueezeLayer(nn.Cell):
    def __init__(self, input_shape, factor=2, reverse=False):
        super(SqueezeLayer, self).__init__()
        # input_shape must be NCHW
        self.input_shape = input_shape
        self.factor = factor
        self.factor2 = self.factor ** 2
        self.reverse = reverse
        self.B = self.input_shape[0]
        self.C = self.input_shape[1]
        self.H = self.input_shape[2]
        self.W = self.input_shape[3]
        self.H_div_factor = self.H // self.factor
        self.W_div_factor = self.W // self.factor
        self.C_factor_factor = self.C * self.factor * self.factor
        self.C_div_factor2 = self.C // self.factor2
        self.H_mul_factor = self.H * self.factor
        self.W_mul_factor = self.W * self.factor
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, input, logdet):
        if not self.reverse:
            x = self.reshape(input, (self.B, self.C, self.H_div_factor, self.factor, self.W_div_factor, self.factor))
            x = self.transpose(x, (0, 1, 3, 5, 2, 4))
            x = self.reshape(x, (self.B, self.C_factor_factor, self.H_div_factor, self.W_div_factor))
            output = x
        else:
            x = self.reshape(input, (self.B, self.C_div_factor2, self.factor, self.factor, self.H, self.W))
            x = self.transpose(x, (0, 1, 4, 2, 5, 3))
            x = self.reshape(x, (self.B, self.C_div_factor2, self.H_mul_factor, self.W_mul_factor))
            output = x
        return output, logdet


class Slice(nn.Cell):
    def __init__(self, begin, size):
        super(Slice, self).__init__()
        self.slice = P.Slice()
        self.begin = begin
        self.size = size

    def construct(self, x):
        h = self.slice(x, self.begin, self.size)
        return h


class StridedSlice(nn.Cell):
    def __init__(self, begin, end, stride):
        super(StridedSlice, self).__init__()
        self.stride_slice = P.StridedSlice()
        self.begin = begin
        self.end = end
        self.stride = stride

    def construct(self, x):
        h = self.stride_slice(x, self.begin, self.end, self.stride)
        return h
