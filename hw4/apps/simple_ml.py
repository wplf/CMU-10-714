"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as f:
        image_bytes = f.read()
        magic_number, num_images, rows, cols = struct.unpack(">iiii", image_bytes[:16])
        pixels = struct.unpack("%dB" % (num_images * rows * cols), image_bytes[16:])
        pixels = np.array(pixels, dtype=np.float32).reshape(num_images,  rows * cols) / 255
    
    with gzip.open(label_filename, 'rb') as f:
        label_bytes = f.read()
        magic_number, num_labels = struct.unpack(">ii", label_bytes[:8])
        labels = struct.unpack("%dB" % num_labels, label_bytes[8:])
        labels = np.array(labels, dtype=np.uint8)
    return pixels, labels    ### END YOUR SOLUTION    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    n = Z.shape[0]
    loss = ndl.logsumexp(Z, 1).sum() - (Z-y_one_hot).sum()
    return loss / n
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, len(X), batch):
        start, end = i, min(i+batch, len(X))
        X_batch = ndl.Tensor(X[start:end])
        y_batch = ndl.Tensor(y[start:end])
        y_one_hot = np.eye(W2.shape[1])[y_batch.cached_data]
        
        z1 = ndl.relu(ndl.matmul(X_batch, W1))
        z2 = ndl.matmul(z1, W2)
        loss = softmax_loss(z2, y_one_hot)
        loss.backward()
        
        W1 = W1.numpy() - lr * W1.grad.numpy()
        W2 = W2.numpy() - lr * W2.grad.numpy()
        W1, W2 = ndl.Tensor(W1), ndl.Tensor(W2)
        print(f"current position {i} loss {loss}")

    return W1, W2    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, cnt = 0, 0
    total_loss = 0
    device= model.device
    if opt:
        model.train()
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc='Batch:', unit='step', total=111)
        for idx, (X, y) in enumerate(pbar):
            # X = X.to(model.device)
            # y = y.to(model.device)
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            opt.reset_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            total_loss += loss.detach().numpy()
            opt.step()   
            correct += (out.numpy().argmax(1) == y.numpy()).sum()  
            cnt += y.shape[0]
            pbar.set_postfix(Parameter=f"acc {(correct/cnt).item():.4f}, loss {(total_loss/cnt).item():.4f}")
            pbar.update()
    else:
        model.eval()
        for idx, (X, y) in enumerate(dataloader):
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            loss = loss_fn(out, y)
            total_loss += loss.detach().numpy()
            correct += (out.numpy().argmax(1) == y.numpy()).sum()  
            cnt += y.shape[0]

    return correct / cnt, total_loss / cnt        
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, 1 + n_epochs):
        acc, loss = epoch_general_cifar10(dataloader=dataloader,
                              model=model,
                              loss_fn=loss_fn(),
                              opt=opt)
    return acc, loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    acc, loss = epoch_general_cifar10(dataloader=dataloader,
                            model=model,
                            loss_fn=loss_fn())
    return acc, loss
    # raise NotImplementedError()
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, cnt = 0, 0
    total_loss = 0
    device= model.device
    nbatch, batch_size = data.shape
    if opt:
        model.train()
    else:
        model.eval()
        
    from tqdm import tqdm
    pbar = tqdm(data, desc='Batch:', unit='step', total=500)
    for i in range(0, nbatch - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, h = model(X)
        # breakpoint()
        loss = loss_fn(out, y)
        
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()   
            
        total_loss += loss.detach().numpy()
        correct += (out.numpy().argmax(1) == y.numpy()).sum()  
        cnt += y.shape[0]
        pbar.set_postfix(Parameter=f"acc {(correct/cnt).item():.4f}, loss {(total_loss/cnt).item():.4f}")
        pbar.update()
    
    return correct / cnt, total_loss / cnt   
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, 1 + n_epochs):
        acc, loss = epoch_general_ptb(data=data,
                              model=model,
                              loss_fn=loss_fn(),
                              opt=opt, clip=None, device=device, dtype=dtype)
    return acc, loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    acc, loss = epoch_general_ptb(data=data,
                            model=model,
                            seq_len=seq_len,
                            loss_fn=loss_fn(),
                            opt=None, clip=None, device=device, dtype=dtype)
    return acc, loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
