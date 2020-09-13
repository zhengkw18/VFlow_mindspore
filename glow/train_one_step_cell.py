import numpy as np
from mindspore.ops import composite as C, functional as F, operations as P
from mindspore.common import Tensor
from mindspore.nn.cell import Cell
from mindspore.common.parameter import ParameterTuple
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import (_get_device_num, _get_mirror_mean, _get_parallel_mode)
from mindspore.train.parallel_utils import ParallelMode

_grad_norm_op = C.MultitypeFuncGraph("grad_clip_norm_op")
_grad_value_op = C.MultitypeFuncGraph("grad_clip_value_op")
judge = Tensor(np.array([[[[1.0]]]]).astype(np.float32))
@_grad_norm_op.register("Tensor")
def grad_clip_norm_clipcoef(grad):
    square_op = P.Square()
    reduce_sum_op = P.ReduceSum()
    axis = ()
    grad_square = square_op(grad)
    gradreduce = reduce_sum_op(grad_square, axis)
    return gradreduce


@_grad_norm_op.register("Tensor", "Tensor")
def tensor_grad_clip_norm(scale, grad):
    minop = P.Minimum()
    return grad * minop(scale, judge)


@_grad_value_op.register("Tensor", "Tensor")
def tensor_grad_clip_value(scale, grad):
    minop = P.Minimum()
    maxop = P.Maximum()
    negop = P.Neg()
    grad = maxop(grad, negop(scale))
    grad = minop(grad, scale)
    return grad


class TrainOneStepCell(Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_mirror_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

        self.hyper_map = C.HyperMap()
        self.addn = P.AddN()
        self.eps = Tensor(np.array([[[[1.0e-6]]]]).astype(np.float32))
        self.add = P.TensorAdd()
        self.max_norm = Tensor(np.array([[[[20]]]]).astype(np.float32))
        self.sqrt = P.Sqrt()
        self.div = P.RealDiv()
        self.judge = Tensor(np.array([[[[1.0]]]]).astype(np.float32))
        self.max_value = Tensor(np.array([1.0]).astype(np.float32))

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        grads = self.hyper_map(F.partial(_grad_value_op, self.max_value), grads)
        grads_hat = self.hyper_map(F.partial(_grad_norm_op), grads)
        add_grad = self.addn(grads_hat)
        grad_sqrt = self.sqrt(add_grad)
        gradeps = self.add(grad_sqrt, self.eps)
        clip_coef = self.div(self.max_norm, gradeps)
        grads = self.hyper_map(F.partial(_grad_norm_op, clip_coef), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
