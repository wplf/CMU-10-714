"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x, device=x.device) / (1 + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        from math import sqrt
        bound = sqrt(1/hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, requires_grad=True))
        self.bias = bias
        if self.bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, requires_grad=True))
        self.nonlinear = ops.tanh if nonlinearity == 'tanh' else ops.relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        cur = X @ self.W_ih
        if h:
            cur += h @ self.W_hh
        if self.bias:
            # breakpoint()
            cur += (self.bias_ih + self.bias_hh).broadcast_to(\
                (cur.shape[0], self.hidden_size))
        return self.nonlinear(cur)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        
        self.rnn_cells = [0] * num_layers
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells[i] = RNNCell(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            nonlinearity=nonlinearity,
                                            device=device,
                                            dtype=dtype)
            else:
                self.rnn_cells[i] = RNNCell(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            nonlinearity=nonlinearity,
                                            device=device,
                                            dtype=dtype)
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        total_h0 = list(ops.split(h0, axis=0)) # each layer has one h_n
        
        layer = list(ops.split(X, axis=0))
        
        for i in range(self.num_layers):
            neuron = self.rnn_cells[i]
            for j in range(seq_len):
                total_h0[i] = neuron(layer[j], total_h0[i])
                layer[j] = total_h0[i]
        return ops.stack(layer, axis=0), ops.stack(total_h0, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        from math import sqrt
        bound = sqrt(1/hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.sigmoid = Sigmoid()
        self.tanh = ops.Tanh()
        self.bias = bias
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size, _ = X.shape
        if h is None:
            h0, c0 = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype), \
                     init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        if self.bias:
            gates_all = X @ self.W_ih + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((batch_size, 4 * self.hidden_size)) \
                        + h0 @ self.W_hh + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((batch_size, 4 * self.hidden_size))
        else:
            gates_all = X @ self.W_ih + h0 @ self.W_hh
        gates_all_split = tuple(ops.split(gates_all, axis = 1))
        gates = []
        for i in range(4):
            gates.append(ops.stack(gates_all_split[i * self.hidden_size : (i + 1) * self.hidden_size], axis = 1))
        i,f,g,o = gates
        i,f,g,o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_out = f * c0 + i * g
        h_out = o * self.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = [0] * num_layers
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells[i] = LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)
            else:
                self.lstm_cells[i] = LSTMCell(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
    
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
            
        total_h0 = list(ops.split(h0, axis=0)) # each layer has one h_n
        total_c0 = list(ops.split(c0, axis=0))
        
        layer = list(ops.split(X, axis=0))
        
        for i in range(self.num_layers):
            neuron = self.lstm_cells[i]
            for j in range(seq_len):
                total_h0[i], total_c0[i] = neuron(layer[j], (total_h0[i], total_c0[i]))
                layer[j] = total_h0[i]
        return ops.stack(layer, axis=0), (ops.stack(total_h0, axis=0), ops.stack(total_c0,axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_one_hot = init.one_hot(self.num_embeddings, x.reshape((seq_len * bs, )), device=self.device, dtype=self.dtype)
        out =  (x_one_hot.reshape((seq_len * bs, self.num_embeddings)) \
            @ self.weight)
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION