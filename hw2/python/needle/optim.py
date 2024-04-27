"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay * param.detach()
            
            self.u[param] = self.u.get(param, 0) * self.momentum \
                    + grad * (1 - self.momentum)
                    
            param.cached_data -= self.lr * self.u[param].cached_data

        # breakpoint()
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        from collections import defaultdict
        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay * param.detach()
            
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad ** 2)
            # breakpoint()
            m_t1_hat = self.m[param] / (1 - self.beta1 ** (self.t))
            v_t1_hat = self.v[param] / (1 - self.beta2 ** (self.t))
            
            param.cached_data -= (self.lr * m_t1_hat / ((v_t1_hat ** 0.5) + self.eps)).cached_data
            # locals().clear()
        # self.v.clear()
        # self.m.clear()
        # self.v.clear()
    # def step(self):
    #     ### BEGIN YOUR SOLUTION
    #     self.t += 1
    #     for para in self.params:
    #         m = self.m.get(para)
    #         v = self.v.get(para)
    #         grad = para.grad.detach() + self.weight_decay * para.detach()
    #         m_new = (1 -
    #                  self.beta1) * grad + self.beta1 * (0 if m == None else m)
    #         v_new = (1 - self.beta2) * (grad**2) + self.beta2 * (0 if v == None
    #                                                              else v)
    #         self.m[para] = m_new
    #         self.v[para] = v_new

    #         m_new = (m_new / (1 - (self.beta1**self.t))).detach()
    #         v_new = (v_new / (1 - (self.beta2**self.t))).detach()
    #         para.cached_data -= self.lr * ((m_new / (
    #             (v_new**0.5) + self.eps))).cached_data

        ### END YOUR SOLUTION
            # breakpoint()
        ### END YOUR SOLUTION
