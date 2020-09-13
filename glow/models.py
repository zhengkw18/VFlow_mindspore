import numpy as np
from . import modules
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor


class ConvNet(nn.Cell):
    def __init__(self, pixels, in_channels, out_channels, hidden_channels, init=False, reverse=False):
        super(ConvNet, self).__init__()
        self.conv1 = modules.Conv2d(pixels, in_channels, hidden_channels, init=init, reverse=reverse)
        self.relu1 = P.ReLU()
        self.conv2 = modules.Conv2d(pixels, hidden_channels, hidden_channels, kernel_size=[1, 1], init=init, reverse=reverse)
        self.relu2 = P.ReLU()
        self.conv3 = modules.Conv2dZeros(hidden_channels, out_channels)

    def construct(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        return h


class ConvNetConditional(nn.Cell):
    def __init__(self, pixels, in_channels, out_channels, hidden_channels, init=False, reverse=False):
        super(ConvNetConditional, self).__init__()
        self.conv1 = modules.Conv2d(pixels, in_channels, hidden_channels, init=init, reverse=reverse)
        self.relu1 = P.ReLU()
        self.conv2 = modules.Conv2d(pixels, hidden_channels, hidden_channels, kernel_size=[1, 1], init=init, reverse=reverse)
        self.relu2 = P.ReLU()
        self.conv3 = modules.Conv2dZeros(hidden_channels, out_channels)
        self.add = P.TensorAdd()

    def construct(self, x, a):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.add(h, a)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        return h


class FlowStep(nn.Cell):
    def __init__(self, batch_size, in_channels, hidden_channels, img_shape, actnorm_scale=1.0, flow_permutation="shuffle",
                 flow_coupling="affine", init=False, reverse=False):
        super(FlowStep, self).__init__()
        self.param_init = init
        self.param_reverse = reverse
        self.img_shape = img_shape
        self.pixels = img_shape[0] * img_shape[1]
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, self.pixels, actnorm_scale, self.param_init, self.param_reverse)
        # 2. permute
        self.input_shape = [batch_size, in_channels, img_shape[0], img_shape[1]]
        if flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(self.input_shape, in_channels, shuffle=True, reverse=self.param_reverse)
            self.SHUFFLE_PERMU = 0
            self.flow_permutation = self.SHUFFLE_PERMU
        else:
            self.reverse = modules.Permute2d(self.input_shape, in_channels, shuffle=False, reverse=self.param_reverse)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = ConvNet(self.pixels, in_channels // 2, in_channels // 2, hidden_channels, self.param_init)
            self.ADDITIVE_COUPLING = 0
            self.flow_coupling = self.ADDITIVE_COUPLING
        elif flow_coupling == "affine":
            self.f = ConvNet(self.pixels, in_channels // 2, in_channels, hidden_channels, self.param_init)
            self.AFFINE_COUPLING = 1
            self.flow_coupling = self.AFFINE_COUPLING
        # MindSpore operators
        self.log = P.Log()
        self.sum = P.ReduceSum(keep_dims=True)
        self.sum_axis = (1, 2, 3)
        self.cat = P.Concat(axis=1)
        self.sigmoid = P.Sigmoid()
        self.add = P.TensorAdd()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.div = P.RealDiv()
        self.slice_z1_begin = (0, 0, 0, 0)
        self.slice_z1_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z1_slice = modules.Slice(self.slice_z1_begin, self.slice_z1_size)
        self.slice_z2_begin = (0, self.input_shape[1] // 2, 0, 0)
        self.slice_z2_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z2_slice = modules.Slice(self.slice_z2_begin, self.slice_z2_size)
        self.strided_slice_shift_begin = (0, 0, 0, 0)
        self.strided_slice_shift_end = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        self.strided_slice_shift_stride = (1, 2, 1, 1)
        self.shift_strided_slice = modules.StridedSlice(self.strided_slice_shift_begin, self.strided_slice_shift_end, self.strided_slice_shift_stride)
        self.strided_slice_scale_begin = (0, 1, 0, 0)
        self.strided_slice_scale_end = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        self.strided_slice_scale_stride = (1, 2, 1, 1)
        self.scale_strided_slice = modules.StridedSlice(self.strided_slice_scale_begin, self.strided_slice_scale_end, self.strided_slice_scale_stride)
        self.const_tensor_2 = Tensor((np.ones([self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3]]) * 2.).astype(np.float32))

    def construct(self, input, logdet):
        if not self.param_reverse:
            # 1. actnorm
            z, logdet = self.actnorm(input, logdet)
            # 2. permute
            if self.flow_permutation == self.SHUFFLE_PERMU:
                z, logdet = (self.shuffle(z), logdet)
            else:
                z, logdet = (self.reverse(z), logdet)
            # 3. coupling
            z1 = self.z1_slice(z)
            z2 = self.z2_slice(z)
            if self.flow_coupling == self.ADDITIVE_COUPLING:
                h = self.f(z1)
                z2 = self.add(z2, h)
            else:
                h = self.f(z1)
                shift = self.shift_strided_slice(h)
                scale = self.scale_strided_slice(h)
                h = self.add(scale, self.const_tensor_2)
                scale = self.sigmoid(h)
                z2 = self.add(z2, shift)
                z2 = self.mul(z2, scale)
                log_scale = self.log(scale)
                h = self.sum(log_scale, self.sum_axis)
                logdet = self.add(h, logdet)
            z = self.cat((z1, z2))
            return z, logdet
        else:
            z1 = self.z1_slice(input)
            z2 = self.z2_slice(input)
            if self.flow_coupling == self.ADDITIVE_COUPLING:
                h = self.f(z1)
                z2 = self.sub(z2, h)
            else:
                h = self.f(z1)
                shift = self.shift_strided_slice(h)
                scale = self.scale_strided_slice(h)
                scale = self.sigmoid(self.add(scale, self.const_tensor_2))
                z2 = self.div(z2, scale)
                z2 = self.sub(z2, shift)
                logdet = self.sub(logdet, self.sum(self.log(self.add(scale, self.const_tensor_2)), self.sum_axis))
            z = self.cat((z1, z2))
            # 2. permute
            if self.flow_permutation == self.SHUFFLE_PERMU:
                z, logdet = (self.shuffle(z), logdet)
            else:
                z, logdet = (self.reverse(z), logdet)
            z, logdet = self.actnorm(z, logdet)
            return z, logdet


class FlowNet(nn.Cell):
    def __init__(self, batch_size, image_shape, hidden_channels, K, L,
                 actnorm_scale=1.0, flow_permutation="shuffle",
                 flow_coupling="affine", init=False, reverse=False,
                 load_pretrained_model=False, eps_std=None):
        super(FlowNet, self).__init__()
        self.batch_size = batch_size
        self.init = init
        self.reverse = reverse
        self.output_shapes = []
        self.K = K
        self.L = L
        self.zero_logdet = Tensor(np.zeros([self.batch_size, 1, 1, 1]).astype(np.float32))
        H, W, C = image_shape
        squeeze_c = C
        unsqueeze_c = C * 4
        unsqueeze_H = H
        unsqueeze_W = W
        if self.reverse is True:
            self.eps_std = eps_std
        else:
            self.eps_std = None
        self.layers = []
        for i in range(L):
            # 1. Squeeze
            if not self.reverse:
                squeeze_input_shape = [self.batch_size, squeeze_c, H, W]
            else:
                unsqueeze_H //= 2
                unsqueeze_W //= 2
                squeeze_input_shape = [self.batch_size, unsqueeze_c, unsqueeze_H, unsqueeze_W]
            self.layers.append(modules.SqueezeLayer(squeeze_input_shape, factor=2, reverse=self.reverse))
            squeeze_c *= 2
            unsqueeze_c *= 2
            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(batch_size=self.batch_size, in_channels=C, hidden_channels=hidden_channels, img_shape=[H, W], actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation, flow_coupling=flow_coupling, init=init, reverse=reverse))
                self.output_shapes.append([-1, C, H, W])
            # 3. Split2d
            if i < L - 1:
                self.layers.append(modules.Split2d(input_shape=[batch_size, C, H, W], num_channels=C, eps_std=self.eps_std, reverse=self.reverse))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2
        for i in range(len(self.layers)):
            exec("self.layer_" + str(i) + "= self.layers[" + str(i) + "]")
        if self.reverse is True:
            self.reversed_layer = []
            if load_pretrained_model is False:
                for layer in reversed(self.layers):
                    self.reversed_layer.append(layer)
                for i in range(len(self.reversed_layer)):
                    exec("self.reversed_layer_" + str(i) + "= self.reversed_layer[" + str(i) + "]")

    def construct(self, z, logdet):
        if not self.reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
            return z, logdet
        else:
            for layer in self.reversed_layer:
                z, logdet = layer(z, self.zero_logdet)
        return z


class Glow(nn.Cell):
    def __init__(self, hparams, init=False, reverse=False):
        super(Glow, self).__init__()
        self.init = init
        self.reverse = reverse
        H, W, C = hparams.Glow.image_shape
        self.reconstruct = hparams.Infer.reconstruct
        load_pretrained_model = hparams.Infer.load_ms_checkpoint or hparams.Infer.load_pt_checkpoint
        self.eps_std = hparams.Glow.eps_std
        self.flow = FlowNet(batch_size=hparams.Train.batch_size, image_shape=[H, W, C * 2],
                            hidden_channels=hparams.Glow.hidden_channels, K=hparams.Glow.K, L=hparams.Glow.L,
                            actnorm_scale=hparams.Glow.actnorm_scale, flow_permutation=hparams.Glow.flow_permutation,
                            flow_coupling=hparams.Glow.flow_coupling, init=self.init, reverse=self.reverse,
                            load_pretrained_model=load_pretrained_model, eps_std=self.eps_std)
        self.augment = Augment(batch_size=hparams.Train.batch_size, image_shape=[H, W, C],
                               hidden_channels=hparams.Glow.augment_hidden_channels,
                               num_augment_steps=hparams.Glow.augment_steps, init=self.init)
        self.batch_size = hparams.Train.batch_size
        self.raw_channel = hparams.Glow.image_shape[2]
        self.x_shape = (hparams.Train.batch_size, hparams.Glow.image_shape[2] * 2,
                        hparams.Glow.image_shape[0], hparams.Glow.image_shape[1])
        self.pixels = int(self.x_shape[2] * self.x_shape[3])
        self.dimensions = self.raw_channel * self.pixels
        # self.init_logdet = Tensor((np.zeros([self.batch_size, 1, 1, 1]) + np.ones([self.batch_size, 1, 1, 1]) * (-np.log(256.) * self.pixels)).astype(np.float32))
        self.init_logdet = Tensor((np.zeros([self.batch_size, 1, 1, 1])).astype(np.float32))
        if not self.reverse:
            self.gaussian_logp = modules.GaussianDiagLogp(self.flow.output_shapes[-1][1])
        else:
            self.eps_std_shape = (self.batch_size // 1, self.flow.output_shapes[-1][1],
                                  self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3])
            self.gaussian_sample = modules.GaussianDiagSample(self.eps_std_shape, self.eps_std)
        self.h = Tensor(np.zeros([self.batch_size, self.flow.output_shapes[-1][1] * 2,
                                  self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3]]).astype(np.float32))
        C = self.flow.output_shapes[-1][1]
        self.param_learn_top = True
        self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        self.project_ycond = modules.LinearZeros(hparams.Glow.y_classes, 2 * C)
        if hparams.Glow.enable_project_class:
            self.enable_project_class = True
            self.project_class = modules.LinearZeros(C, hparams.Glow.y_classes)
        else:
            self.enable_project_class = False
        self.B, self.C = self.batch_size, self.flow.output_shapes[-1][1] * 2
        self.h_sz_c = self.flow.output_shapes[-1][1] * 2
        self.c_slice = self.h_sz_c // 2
        self.div = P.RealDiv()
        self.bit_x = Tensor((np.ones([self.batch_size, 1, 1, 1]) * (np.log(2.))).astype(np.float32))
        self.add = P.TensorAdd()
        self.neg = P.Neg()
        self.drop_begin = (0, 0, 0, 0)
        self.drop_size = (self.batch_size, self.raw_channel, self.x_shape[2], self.x_shape[3])
        self.slice_drop = modules.Slice(self.drop_begin, self.drop_size)
        self.slice_h1_begin = (0, 0, 0, 0)
        self.slice_h1_size = (self.h.shape()[0], self.h.shape()[1] // 2, self.h.shape()[2], self.h.shape()[3])
        self.slice_h1 = modules.Slice(self.slice_h1_begin, self.slice_h1_size)
        self.slice_h2_begin = (0, self.h.shape()[1] // 2, 0, 0)
        self.slice_h2_size = (self.h.shape()[0], self.h.shape()[1] // 2, self.h.shape()[2], self.h.shape()[3])
        self.slice_h2 = modules.Slice(self.slice_h2_begin, self.slice_h2_size)
        self.reduce_mean_1 = P.ReduceMean(keep_dims=False)
        self.reduce_mean_2 = P.ReduceMean(keep_dims=False)
        self.cast = P.Cast()
        self.type_float32 = mstype.float32
        self.reshape = P.Reshape()
        self.mean = Tensor((np.zeros(self.x_shape)).astype(np.float32))
        self.stddev = Tensor((np.ones(self.x_shape) * (1 / 256)).astype(np.float32))
        self.ones = Tensor(np.array([[[[1.]]]]).astype(np.float32))
        self.twos = Tensor(np.array([[[[2.]]]]).astype(np.float32))
        self.bounds = Tensor(np.array([[[[0.9]]]]).astype(np.float32))
        self.mul = P.Mul()
        self.concat = P.Concat(axis=1)
        self.sub = P.Sub()
        self.shape = P.Shape()
        self.sigmoid = P.Sigmoid()
        self.log = P.Log()

    def prior(self, y_onehot):
        h = self.learn_top(self.h)
        yp = self.project_ycond(y_onehot)
        yp = self.reshape(yp, (self.B, self.C, 1, 1))
        h = self.add(h, yp)
        h1 = self.slice_h1(h)
        h2 = self.slice_h2(h)
        return h1, h2

    def construct(self, x, y_onehot):
        if not self.reverse:
            return self.normal_flow(x, y_onehot)
        elif self.reconstruct:
            return self.reconstruct_reverse_flow(x, y_onehot)
        else:
            return self.reverse_flow(y_onehot)

    def normal_flow(self, x, y_onehot):
        x = self.mul(x, self.twos)
        x = self.sub(x, self.ones)
        x = self.mul(x, self.bounds)
        x = self.add(x, self.ones)
        x = self.div(x, self.twos)
        x1 = x
        x1 = self.log(x1)
        x2 = self.sub(self.ones, x)
        x2 = self.log(x2)
        x = self.sub(x1, x2)
        x_augmentation, _logdet = self.augment(x)
        x = self.concat((x, x_augmentation))
        z, objective = self.flow(x, self.init_logdet)
        mean, logs = self.prior(y_onehot)
        gauss_sample = self.gaussian_logp(mean, logs, z)
        objective = self.add(objective, gauss_sample)
        objective = self.sub(objective, _logdet)
        reduced_z = self.reduce_mean_1(z, 2)
        reduced_z = self.reduce_mean_2(reduced_z, 2)
        if self.enable_project_class:
            y_logits = self.project_class(z)
        else:
            y_logits = y_onehot
        h = self.neg(objective)
        nll = self.div(h, self.bit_x)
        return z, nll, y_logits, self.dimensions

    def reconstruct_reverse_flow(self, x, y_onehot):
        x = self.flow(x, self.init_logdet)
        return x

    def reverse_flow(self, y_onehot):
        mean, logs = self.prior(y_onehot)
        z = self.gaussian_sample(mean, logs)
        x = self.flow(z, self.init_logdet)
        x = self.slice_drop(x)
        x = self.sigmoid(x)
        return x


class AugmentStep(nn.Cell):
    def __init__(self, batch_size, in_channels, hidden_channels, img_shape, actnorm_scale=1.0, flow_permutation="shuffle",
                 flow_coupling="affine", init=False):
        super(AugmentStep, self).__init__()
        self.param_init = init
        self.param_reverse = False
        self.img_shape = img_shape
        self.pixels = img_shape[0] * img_shape[1]
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, self.pixels, actnorm_scale, self.param_init, self.param_reverse)
        # 2. permute
        self.input_shape = [batch_size, in_channels, img_shape[0], img_shape[1]]
        if flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(self.input_shape, in_channels, shuffle=True, reverse=self.param_reverse)
            self.SHUFFLE_PERMU = 0
            self.flow_permutation = self.SHUFFLE_PERMU
        else:
            self.reverse = modules.Permute2d(self.input_shape, in_channels, shuffle=False, reverse=self.param_reverse)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = ConvNetConditional(self.pixels, in_channels // 2, in_channels // 2, hidden_channels, self.param_init)
            self.ADDITIVE_COUPLING = 0
            self.flow_coupling = self.ADDITIVE_COUPLING
        elif flow_coupling == "affine":
            self.f = ConvNetConditional(self.pixels, in_channels // 2, in_channels, hidden_channels, self.param_init)
            self.AFFINE_COUPLING = 1
            self.flow_coupling = self.AFFINE_COUPLING
        # MindSpore operators
        self.log = P.Log()
        self.sum = P.ReduceSum(keep_dims=True)
        self.sum_axis = (1, 2, 3)
        self.cat = P.Concat(axis=1)
        self.sigmoid = P.Sigmoid()
        self.add = P.TensorAdd()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.div = P.RealDiv()
        self.slice_z1_begin = (0, 0, 0, 0)
        self.slice_z1_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z1_slice = modules.Slice(self.slice_z1_begin, self.slice_z1_size)
        self.slice_z2_begin = (0, self.input_shape[1] // 2, 0, 0)
        self.slice_z2_size = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3])
        self.z2_slice = modules.Slice(self.slice_z2_begin, self.slice_z2_size)
        self.strided_slice_shift_begin = (0, 0, 0, 0)
        self.strided_slice_shift_end = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        self.strided_slice_shift_stride = (1, 2, 1, 1)
        self.shift_strided_slice = modules.StridedSlice(self.strided_slice_shift_begin, self.strided_slice_shift_end, self.strided_slice_shift_stride)
        self.strided_slice_scale_begin = (0, 1, 0, 0)
        self.strided_slice_scale_end = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        self.strided_slice_scale_stride = (1, 2, 1, 1)
        self.scale_strided_slice = modules.StridedSlice(self.strided_slice_scale_begin, self.strided_slice_scale_end, self.strided_slice_scale_stride)
        self.const_tensor_2 = Tensor((np.ones([self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2], self.input_shape[3]]) * 2.).astype(np.float32))

    def construct(self, input, logdet, a):
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet)
        # 2. permute
        if self.flow_permutation == self.SHUFFLE_PERMU:
            z, logdet = (self.shuffle(z), logdet)
        else:
            z, logdet = (self.reverse(z), logdet)
        # 3. coupling
        z1 = self.z1_slice(z)
        z2 = self.z2_slice(z)
        if self.flow_coupling == self.ADDITIVE_COUPLING:
            h = self.f(z1, a)
            z2 = self.add(z2, h)
        else:
            h = self.f(z1, a)
            shift = self.shift_strided_slice(h)
            scale = self.scale_strided_slice(h)
            h = self.add(scale, self.const_tensor_2)
            scale = self.sigmoid(h)
            z2 = self.add(z2, shift)
            z2 = self.mul(z2, scale)
            log_scale = self.log(scale)
            h = self.sum(log_scale, self.sum_axis)
            logdet = self.add(h, logdet)
        z = self.cat((z1, z2))
        return z, logdet


class Augment(nn.Cell):
    def __init__(self, batch_size, image_shape, hidden_channels, num_augment_steps, init=False):
        super(Augment, self).__init__()
        self.param_init = init
        self.batch_size = batch_size
        self.H, self.W, self.C = image_shape
        self.layers = []
        self.squeeze = modules.SqueezeLayer([self.batch_size, self.C, self.H, self.W], factor=2, reverse=False)
        self.shallow = ConvNet(None, self.C * 4, hidden_channels, hidden_channels, self.param_init)
        for i in range(num_augment_steps):
            self.layers.append(AugmentStep(batch_size=batch_size, in_channels=self.C * 4, hidden_channels=hidden_channels, img_shape=[self.H // 2, self.W // 2, self.C * 4]))
        self.unsqueeze = modules.SqueezeLayer([self.batch_size, self.C * 4, self.H // 2, self.W // 2], factor=2, reverse=True)
        self.h = Tensor(np.zeros([self.batch_size, self.C * 8, self.H // 2, self.W // 2]).astype(np.float32))
        self.learn_top = modules.Conv2dZeros(self.C * 8, self.C * 8)
        self.logp = modules.GaussianDiagLogp(self.C * 4)
        self.eps_std_shape = (self.batch_size, self.C * 4, self.H // 2, self.W // 2)
        self.gaussian_sample = modules.GaussianDiagSample(self.eps_std_shape)
        self.add = P.TensorAdd()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.div = P.RealDiv()
        self.slice_h1_begin = (0, 0, 0, 0)
        self.slice_h1_size = (self.h.shape()[0], self.h.shape()[1] // 2, self.h.shape()[2], self.h.shape()[3])
        self.slice_h1 = modules.Slice(self.slice_h1_begin, self.slice_h1_size)
        self.slice_h2_begin = (0, self.h.shape()[1] // 2, 0, 0)
        self.slice_h2_size = (self.h.shape()[0], self.h.shape()[1] // 2, self.h.shape()[2], self.h.shape()[3])
        self.slice_h2 = modules.Slice(self.slice_h2_begin, self.slice_h2_size)
        self.init_logdet = Tensor((np.zeros([self.batch_size, 1, 1, 1])).astype(np.float32))

    def construct(self, x):
        logdet = self.init_logdet
        x, logdet = self.squeeze(x, logdet)
        a = self.shallow(x)
        h = self.learn_top(self.h)
        means = self.slice_h1(h)
        logs = self.slice_h2(h)
        eps = self.gaussian_sample(means, logs)
        eps_logp = self.logp(means, logs, eps)
        z = eps
        for layer in self.layers:
            z, logdet = layer(z, logdet, a)
        z, logdet = self.unsqueeze(z, logdet)
        logdet = self.sub(eps_logp, logdet)
        return z, logdet


class GlowLoss(nn.Cell):
    def __init__(self, y_condition, y_briterion, weight_y):
        super(GlowLoss, self).__init__()
        self.y_condition = y_condition
        self.y_briterion = y_briterion
        self.weight_y = weight_y
        self.reduce_mean_1 = P.ReduceMean(keep_dims=False)
        self.reduce_mean_2 = P.ReduceMean(keep_dims=True)
        self.reduce_sum_1 = P.ReduceSum(keep_dims=False)
        self.reduce_sum_2 = P.ReduceSum(keep_dims=True)
        self.softmax_crossentropy_with_logits = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=False)
        self.add = P.TensorAdd()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.div = P.RealDiv()

    def construct(self, nll, y_logits, y_onehot, dimensions):
        loss_generative = self.reduce_sum_1(nll, (1, 2, 3))
        loss_generative = self.div(loss_generative, dimensions)
        loss_generative = self.reduce_mean_2(loss_generative, (0))
        loss_classes = self.softmax_crossentropy_with_logits(y_logits, y_onehot)
        loss_classes = self.reduce_mean_2(loss_classes, (0))
        h = self.mul(loss_classes, self.weight_y)
        loss = self.add(loss_generative, h)
        return loss
