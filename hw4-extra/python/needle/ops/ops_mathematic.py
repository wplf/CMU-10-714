"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * (node.inputs[0] ** (self.scalar-1) * self.scalar ), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # if not isinstance(node.inputs[0], NDArray) or not isinstance(
        #     node.inputs[1], NDArray
        # ):
        #     raise ValueError("Both inputs must be tensors (NDArray).")
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -1 * out_grad * a / b / b
        # breakpoint()
        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        transpose_axis = [x for x in range(len(a.shape))]
        if self.axes is None:
            transpose_axis[-2], transpose_axis[-1] = \
                transpose_axis[-1], transpose_axis[-2]
        else:
            transpose_axis[self.axes[0]], transpose_axis[self.axes[1]] = \
                self.axes[1], self.axes[0]
        return a.permute(transpose_axis)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (transpose(out_grad, axes=self.axes), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.reshape(node.inputs[0].shape), )


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if len(a.shape) < len(self.shape):
            a = a.reshape (tuple((len(self.shape) - len(a.shape) )* [1] + list(a.shape)))
        return array_api.broadcast_to(a, self.shape)


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        if len(shrink_dims) == 0:
            return out_grad.reshape(ori_shape)
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # ndarray.sum 只支持一个轴或者所有轴
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        return a.sum(axis=self.axes)
        # return 
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
    ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mat_1, mat_2 = node.inputs[0], node.inputs[1]
        # breakpoint()

        grad_1 = matmul(out_grad, transpose(mat_2))
        grad_2 = matmul(transpose(mat_1), out_grad)
        
        # breakpoint()
        if grad_2.shape != mat_2.shape:
            grad_2 = summation(grad_2, axes=tuple(range(len(grad_2.shape)-len(mat_2.shape))))
        if grad_1.shape != mat_1.shape:
            grad_1 = summation(grad_1, axes=tuple(range(len(grad_1.shape)-len(mat_1.shape))))

        return grad_1, grad_2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1 * out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * (Tensor(1) / node.inputs[0]), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * exp(node.inputs[0]), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        # return a * (a>=0)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0].cached_data>=0), 
        # raise NotImplementedError()
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - array_api.tanh(node.inputs[0].cached_data) ** 2)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert all(arg.shape == args[0].shape for arg in args)
        new_shape = list(args[0].shape)
        n = len(args)
        new_shape.insert(self.axis, n)
        ans = array_api.empty(tuple(new_shape), device=args[0].device)
        slices = [slice(0, s, 1) for s in new_shape]
        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            ans[tuple(slices)] = args[i]
        return ans
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ans = []
        slices = [slice(0, s) for i, s in enumerate(A.shape)]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        for i in range(A.shape[self.axis]):
            slices[self.axis] = slice(i, i+1)
            ans.append(A[tuple(slices)].compact().reshape(new_shape))
        # breakpoint()
        return tuple(ans)
        
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # forward pass in NDarray
        return a.flip(self.axes)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # backward pass in Tensor
        # breakpoint()
        return Tensor(out_grad.cached_data.flip(self.axes),
                      device=out_grad.device,
                      requires_grad=False)
        # raise out_grad.flip(self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        step = self.dilation + 1
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] *= step
        ans = a.device.full(tuple(new_shape), 0)
        slices = []
        for i in range(len(new_shape)):
            cur_step = 1 if i not in self.axes else step
            slices.append(slice(0, new_shape[i], cur_step))
        # breakpoint()
        ans[tuple(slices)] = a
        return ans
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        step = self.dilation + 1
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] //= step
        ans = a.device.full(tuple(new_shape), 0)
        slices = []
        for i in range(len(new_shape)):
            cur_step = 1 if i not in self.axes else step
            slices.append(slice(0, a.shape[i], cur_step))
        ans = a[tuple(slices)] 
        return ans
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A, n, h, w, channel 
        # B, kernel, kernel, in_channel, out_channel
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, C_in_s = A.strides

        out_H = (H-K+1)//self.stride
        out_W = (W-K+1)//self.stride

        inner_dim = K * K * C_in
        transformed_A = A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                                     strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, C_in_s))
        # breakpoint()
        if A.device != B.device:
            breakpoint()
        out = transformed_A.compact().reshape((N * out_H * out_W, inner_dim)) @ \
                    B.compact().reshape((inner_dim, C_out))
        return out.compact().reshape( (N, out_H, out_W, C_out) )
        # naive conv implement
        # for n in range(N):
        #     for c_in in range(C_in):
        #         for c_out in range(C_out):
        #             for x in range(out_shape[1]):
        #                 for y in range(out_shape[2]):
        #                      for i in range(K):
        #                          for j in range(K):
        #                              out[n, x, y, c_out] += B[i, j, c_in, c_out] * pad_A[n, x+i*self.stride, y+j*self.stride, c_in]
        # # breakpoint()
        
        # im2col implement
        # def conv_im2col(Z, weight):
        #     N,H,W,C_in = Z.shape
        #     K,_,_,C_out = weight.shape
        #     Ns, Hs, Ws, Cs = Z.strides
            
        #     inner_dim = K * K * C_in
        #     A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in),
        #                                         strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
        #     out = A @ weight.reshape(-1, C_out)
        #     return out.reshape(N,H-K+1,W-K+1,C_out)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # X: N, H, W, c_in
        # W: K, K, c_in, c_out
        
        X, W = node.inputs[0], node.inputs[1]
        K, _, _, _ = W.shape
        # out_grad: N, (H-K+1+2P)//stride, (W-K+1+2P)//stride, c_out
        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1)
        # (N, H+2P+1-K, W+2P+1-K, C_out), (K, K, c_out, c_in) => (N, H, W, C_in)
        # H+2P+1-K + 2 * var - K + 1 = H
        # var = -P+K-1
        W_traspose = transpose(flip(W, axes=(0, 1)), axes=(2,3))
        X_grad = conv(out_grad, W_traspose, padding=K-1-self.padding)
        
        # (c_in, H, W, N), (H+2P+1-K, W+2P+1-K, N, C_out) => (c_in, K, K, c_out) 
        # H + 2 * var - (H+2P+1-K) + 1 = K  {var means padding in this conv} 
        # var = P
        X_transpose = transpose(X, (0, 3))
        out_grad_transpose = transpose(transpose(out_grad, (0,1)), (1,2))

        W_grad = conv(X_transpose, out_grad_transpose, padding=self.padding)
        W_grad = transpose(transpose(W_grad, (0,1)), (1, 2))

        return X_grad, W_grad
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
