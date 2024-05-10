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
        # softmax will compact tensor according to self.axes, into out_grad.shape
        # So we will restore out_grad to inp.shape
        
        inp = node.inputs[0]
        X = inp.cached_data
        X_max = array_api.max(X, axis=self.axes, keepdims=True)
        softmax_X = Tensor(array_api.exp(X-X_max)) / Tensor(array_api.sum(array_api.exp(X-X_max), axis=self.axes, keepdims=True))
        
        expand_axes = list(inp.shape)
        axes = self.axes if self.axes else list(range(len(inp.shape)))
        for i in axes:
            expand_axes[i] = 1
        grad_out = out_grad.reshape(expand_axes).broadcast_to(inp.shape)
        
        return Tensor(grad_out * softmax_X ), 
        
        
        # pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
