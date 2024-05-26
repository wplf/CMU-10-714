"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_channels*kernel_size*kernel_size, 
            fan_out=out_channels*kernel_size*kernel_size, 
            shape=(kernel_size,kernel_size,in_channels,out_channels),
            device=device),device=device)
        if bias:
            self.bias = Parameter(init.rand(
                out_channels,
                low=-1/(in_channels*kernel_size**2)**0.5,
                high=1/(in_channels*kernel_size**2)**0.5,
                device=device),device=device
            )
        else:
            bias = None
        self.padding = (kernel_size-1)//2
        ### END YOUR SOLUTION
        # NHWC
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # breakpoint()
        x_NHWC = x.transpose((1,2)).transpose((2,3))
        conv_out = ops.conv(x_NHWC, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            conv_out += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(conv_out.shape)
        return conv_out.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION