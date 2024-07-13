#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from nda.problems import Problem
from nda.datasets import MNIST
from nda import log

def relu(x):
    return xp.maximum(0, x)

def softmax(x):
    tmp = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return tmp / tmp.sum(axis=1, keepdims=True)

def softmax_loss(Y, score):
    return - xp.sum(Y * xp.log(score + 1e-8)) / Y.shape[0]

import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from nda.problems import Problem
from nda import log

def relu(x):
    return xp.maximum(0, x)

def softmax(x):
    tmp = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return tmp / tmp.sum(axis=1, keepdims=True)

def softmax_loss(Y, score):
    return - xp.sum(Y * xp.log(score + 1e-8)) / Y.shape[0]

class ResNet8(Problem):
    def __init__(self, n_channels=16, dataset='mnist', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

        self.n_channels = n_channels
        self.n_class = self.Y_train.shape[1]
        self.img_dim = int(xp.sqrt(self.X_train.shape[1] - 1))  # Subtracting 1 for bias term
        
        log.info(f"X_train shape: {self.X_train.shape}")
        log.info(f"Y_train shape: {self.Y_train.shape}")
        log.info(f"Image dimension: {self.img_dim}")
        
        # Define layer dimensions for MNIST data (single-channel input)
        self.layer_dims = [
            (self.n_channels, 1, 3, 3),  # Initial conv: (output channels, input channels, kernel height, kernel width)
            (self.n_channels, self.n_channels, 3, 3),  # Res block 1
            (self.n_channels, self.n_channels, 3, 3),
            (2*self.n_channels, self.n_channels, 3, 3),  # Res block 2
            (2*self.n_channels, 2*self.n_channels, 3, 3),
            (4*self.n_channels, 2*self.n_channels, 3, 3),  # Res block 3
            (4*self.n_channels, 4*self.n_channels, 3, 3),
            (4*self.n_channels * ((self.img_dim - 18)//2)**2, self.n_class)  # Final fully connected layer
        ]
        
        self.dim = sum(np.prod(dim) for dim in self.layer_dims)
        log.info(f"Total dimensions: {self.dim}")

        self.Y_train_labels = self.Y_train.argmax(axis=1)
        self.Y_test_labels = self.Y_test.argmax(axis=1)

        # Internal buffers
        self._dw = xp.zeros(self.dim)
        self._dw_layers = self.unpack_w(self._dw)

        log.info('Initialization done')
    
    def cuda(self):
        super().cuda()
        self._dw_layers = self.unpack_w(self._dw)

    def unpack_w(self, W):
        layers = []
        start = 0
        for dim in self.layer_dims:
            end = start + np.prod(dim)
            layers.append(W[start:end].reshape(dim))
            start = end
        return layers

    def pack_w(self, layers):
        return xp.concatenate([layer.reshape(-1) for layer in layers])

    def conv2d(self, X, W):
        n, c, h, w = X.shape
        f, _, kh, kw = W.shape
        oh = h - kh + 1
        ow = w - kw + 1

        X_col = xp.lib.stride_tricks.as_strided(X,
            shape=(n, c, oh, ow, kh, kw),
            strides=(X.strides[0], X.strides[1], X.strides[2], X.strides[3], X.strides[2], X.strides[3])
        )

        X_col = X_col.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * kh * kw)
        W_col = W.reshape(f, c * kh * kw).T  # Ensure correct shape

        out = xp.dot(X_col, W_col)
        out = out.reshape(n, oh, ow, f).transpose(0, 3, 1, 2)
        return out



    def forward(self, X, w):
        layers = self.unpack_w(w)
        A = X[:, :-1].reshape(-1, 1, self.img_dim, self.img_dim)  # Reshape and remove bias column
        print(f"Initial A shape: {A.shape}")
        cache = [A]

        # Initial conv
        A = relu(self.conv2d(A, layers[0]))
        print(f"After initial conv A shape: {A.shape}")
        cache.append(A)

        # Residual blocks
        for i in range(1, len(layers)-1, 2):
            identity = A
            print(f"identity shape: {identity.shape}")
            A = relu(self.conv2d(A, layers[i]))
            print(f"A after relu: {A.shape}")
            A = self.conv2d(A, layers[i+1])
            print(f"A after conv2d: {A.shape}")
            
            # Ensure identity matches A in shape
            if A.shape[2] != identity.shape[2] or A.shape[3] != identity.shape[3]:  # Spatial dimension change
                identity = identity[:, :, :A.shape[2], :A.shape[3]]  # Adjust spatial dimensions
                print(f"identity reshaped: {identity.shape}")

            if A.shape[1] != identity.shape[1]:  # Channel increase
                identity = xp.pad(identity, ((0,0),(A.shape[1]-identity.shape[1],0),(0,0),(0,0)), 'constant')
                print(f"identity reshaped: {identity.shape}")

            A = relu(A + identity)
            print(f"After res block {i//2 + 1} A shape: {A.shape}")
            cache.append(A)

        # Global average pooling
        A = A.mean(axis=(2, 3))
        print(f"After global average pooling A shape: {A.shape}")

        # Final fully connected layer
        scores = softmax(xp.dot(A, layers[-1]))
        print(f"Final scores shape: {scores.shape}")
        print("forward completed")

        return softmax_loss(self.Y_train, scores), cache, scores

        
    def forward_backward(self, X, Y, w):
        layers = self.unpack_w(w)
        loss, cache, scores = self.forward(X, w)

        grads = [xp.zeros_like(layer) for layer in layers]
        dA = scores - Y

        # Backward pass for final fully connected layer
        grads[-1] = xp.dot(cache[-1].mean(axis=(2, 3)).T, dA)

        for i in reversed(range(len(layers)-1)):
            if i % 2 == 0:  # Start of residual block
                dA = dA.reshape(cache[i+1].shape) + cache[i+1]  # Skip connection
            dA = relu(dA) * (cache[i+1] > 0)
            grads[i] = self.conv2d(cache[i].transpose(1, 0, 2, 3), dA.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
            if i > 0:
                dA = self.conv2d(dA, layers[i].transpose(1, 0, 2, 3))

        self._dw = self.pack_w(grads)

        return self._dw, loss

    def grad_h(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if w.ndim == 1:
            if type(j) is int:
                j = [j]

            if i is None and j is None:  # Return the full gradient
                return self.forward_backward(self.X_train, self.Y_train, w)[0]
            elif i is not None and j is None:  # Return the local gradient
                return self.forward_backward(self.X[i], self.Y[i], w)[0]
            elif i is None and j is not None:  # Return the stochastic gradient
                return self.forward_backward(self.X_train[j], self.Y_train[j], w)[0]
            else:  # Return the stochastic gradient
                return self.forward_backward(self.X[i][j], self.Y[i][j], w)[0]

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                return xp.array([self.forward_backward(self.X[i], self.Y[i], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            elif i is None and j is not None:  # Return the stochastic gradient
                return xp.array([self.forward_backward(self.X[i][j[i]], self.Y[i][j[i]], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            else:
                log.fatal('For distributed gradients j must be None')

        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None, split='train'):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            if w.ndim > 1 or i is not None or j is not None:
                log.fatal("Function value on test set only applies to one parameter vector")
            X = self.X_test
            Y = self.Y_test

        if i is None and j is None:  # Return the function value
            return self.forward(X, w)[0]
        elif i is not None and j is None:  # Return the function value at machine i
            return self.forward(self.X[i], w)[0]
        else:  # Return the function value at machine i
            if type(j) is int:
                j = [j]
            return self.forward(self.X[i][j], w)[0]

    def accuracy(self, w, split='test'):
        if w.ndim > 1:
            w = w.mean(axis=1)
        if split == 'train':
            X = self.X_train
            Y = self.Y_train
            labels = self.Y_train_labels
        elif split == 'test':
            X = self.X_test
            Y = self.Y_test
            labels = self.Y_test_labels
        else:
            log.fatal('Data split %s is not supported' % split)

        loss, _, scores = self.forward(X, w)
        pred = scores.argmax(axis=1)

        return sum(pred == labels) / len(pred), loss


if __name__ == '__main__':
    p = ResNet8()