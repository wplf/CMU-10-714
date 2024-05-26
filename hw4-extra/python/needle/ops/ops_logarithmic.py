from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)

def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Note, it is crucial to the Z_max dimension, because the elementwise sub will call broadcast. If we do not have that dimension, the process will collapse. 
        Z_max = Z.max(axis=self.axes, keepdims=True)
        Z_max_reduce = Z.max(axis=self.axes)
        ans = (Z-Z_max.broadcast_to(Z.shape)).exp().sum(axis=self.axes).log() + Z_max_reduce
        # ans = array_api.log(array_api.sum(array_api.exp(Z - Z_max.broadcast_to(Z.shape)), axis=self.axes)) + Z_max_reduce
        # breakpoint()
        return ans
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # outgrad and lower shape is smaller
        # upper shape is bigger
        z = node.inputs[0]
        max_z = z.cached_data.max(self.axes, keepdims=True).broadcast_to(z.shape)
        upper = exp(z-max_z)
        lower = upper.sum(self.axes)
        new_shape = list(z.shape)
        axes = list(z.shape) if self.axes is None else self.axes
        for x in axes:
            new_shape[x] = 1
        return (out_grad / lower).reshape(tuple(new_shape)).broadcast_to(z.shape) * upper
        

        # pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
