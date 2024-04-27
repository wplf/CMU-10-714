"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X: N x in_feat, weight: in_feat x out_feat
        # w = self.weight.broadcast_to(X.shape)
        # b = self.bias.broadcast_to()
        
        out = X.matmul(self.weight)
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        # return ops.relu(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        cur = x
        for mod in self.modules:
            cur = mod(cur)
        return cur
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # z_y = init.one_hot()
        # breakpoint()
        log_sum_exp = ops.logsumexp(logits, axes=(-1, ))

        z_y = init.one_hot(logits.shape[1], y, device=logits.device)
        ans = log_sum_exp.sum() - (z_y * logits).sum()
        return ans / logits.shape[0]
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), requires_grad=True)
        
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)

        if self.training:
            x_mean = x.sum(0) / x.shape[0]
            expanded_mean = x_mean.broadcast_to(x.shape)
            x_var = ((x-expanded_mean) ** 2).sum(0) / x.shape[0]
            expanded_var = x_var.broadcast_to(x.shape)
            
            self.running_mean = self.running_mean * (1-self.momentum) + x_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum)+ x_var *  self.momentum
        else:
            x_mean = self.running_mean
            x_var = self.running_var
            
            expanded_mean = x_mean.broadcast_to(x.shape)
            expanded_var = x_var.broadcast_to(x.shape)
        
        
        x_std = ops.power_scalar(expanded_var + self.eps, 0.5)   
        norm = (x - expanded_mean) / x_std
        # breakpoint()
        return w * norm + b


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), requires_grad=True)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # since summation can not keep dimension, we need to reshape it
        # ndl don't support implicit broadcast, because that is not be tracked by backward.
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        
        x_mean = x.sum(1) / x.shape[-1]
        x_mean = x_mean.reshape(x_mean.shape + (1,)).broadcast_to(x.shape)

        x_var = ((x-x_mean) ** 2).sum(axes=(-1, )) / x.shape[-1] + self.eps
        x_var = x_var.reshape(x_var.shape + (1,)).broadcast_to(x.shape)
        x_std = ops.power_scalar(x_var, 0.5)
        
        norm = (x - x_mean) / x_std
        return w * norm + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p)
            return x * mask / (1 - self.p)
        return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION
