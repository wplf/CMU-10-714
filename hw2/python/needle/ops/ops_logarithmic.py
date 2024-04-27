from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
import needle as nd

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return logsumexp(Z, axes=-1) - nd.nn.init.one_hot(Z.shape[-1], Z.shape[-1], device=Z.device)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Z
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        ans = array_api.log(array_api.sum(array_api.exp(Z - Z_max), axis=self.axes, keepdims=True)) + Z_max
        # Note, it is crucial to keepdims. 
        # Z_max = array_api.max(Z, axis=self.axes)
        # ans2 = array_api.log(array_api.sum(array_api.exp(Z - Z_max), axis=self.axes)) + Z_max
        # breakpoint()
        return ans.squeeze()
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

