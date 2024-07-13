import numpy as np
#import logging

# No need for logging module

try:
    import cupy as xp
except ImportError:
    xp = np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nda.problems import Problem
from nda.datasets import MNIST
#from nda import log  # Remove import of log module


def sigmoid(x):
    return 1 / (1 + xp.exp(-x))


def softmax(x):
    tmp = xp.exp(x)
    return tmp / tmp.sum(axis=1, keepdims=True)


def softmax_loss(Y, score):
    return - xp.sum(xp.log(score[Y != 0])) / Y.shape[0]
    # return - xp.sum(Y * xp.log(score)) / Y.shape[0]


class NN(Problem):
    def __init__(self, dataset='mnist', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        #self.debug = logging.getLogger(__name__)  # Remove logger initialization

        self.n_class = self.Y_train.shape[1]
        self.img_dim = self.X_train.shape[1]
        self.img_shape = (1, 28, 28)  # MNIST image shape

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_class)
        )

        self.dim = sum(p.numel() for p in self.model.parameters())

        self.Y_train_labels = self.Y_train.argmax(axis=1)
        self.Y_test_labels = self.Y_test.argmax(axis=1)

        self._dw = torch.zeros(self.dim)
        self.criterion = nn.CrossEntropyLoss()

        print('Initialization done')  # Print statement instead of logging

    def preprocess_input(self, X):
        # Debugging statements for input shapes
        print(f"Original shape of X: {X.shape}")
        
        # Remove the extra column if present
        if X.shape[1] == 785:
            X = X[:, :-1]
        
        # Reshape the input data
        reshaped_X = torch.tensor(X, dtype=torch.float32).reshape(-1, *self.img_shape)
        
        # Debugging statements for reshaped input
        print(f"Reshaped shape of X: {reshaped_X.shape}")
        
        return reshaped_X

    def cuda(self):
        super().cuda()
        self.model = self.model.cuda()
        self._dw = self._dw.cuda()

    def unpack_w(self, W):
        W = torch.as_tensor(W)
        params = []
        idx = 0
        for param in self.model.parameters():
            num_params = param.numel()
            params.append(W[idx:idx+num_params].reshape(param.shape))
            idx += num_params
        return params

    def pack_w(self, params):
        return torch.cat([p.flatten() for p in params])

    def grad_h(self, w, i=None, j=None):
        print(f"grad_h called with w shape: {w.shape}, i: {i}, j: {j}")
        
        if w.ndim == 1:
            if isinstance(j, int):
                j = [j]

            if i is None and j is None:
                return self.forward_backward(self.X_train, self.Y_train, w)[0]
            elif i is not None and j is None:
                return self.forward_backward(self.X[i], self.Y[i], w)[0]
            elif i is None and j is not None:
                return self.forward_backward(self.X_train[j], self.Y_train[j], w)[0]
            else:
                return self.forward_backward(self.X[i][j], self.Y[i][j], w)[0]

        elif w.ndim == 2:
            if i is None and j is None:
                print(f"wdim calling:{w.ndim}")
                return np.stack([self.forward_backward(self.X[i], self.Y[i], w[:, i])[0] for i in range(self.n_agent)]).T
            elif i is None and j is not None:
                print(f"wdim calling:{w.ndim}")
                return np.stack([self.forward_backward(self.X[i][j[i]], self.Y[i][j[i]], w[:, i])[0] for i in range(self.n_agent)]).T
            else:
                print('For distributed gradients j must be None')

        else:
            print('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None, split='train'):
        print(f"h called with w shape: {w.shape}, i: {i}, j: {j}, split: {split}")

        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            if w.ndim > 1 or i is not None or j is not None:
                print("Function value on test set only applies to one parameter vector")
            X = self.X_test
            Y = self.Y_test

        if i is None and j is None:
            return self.forward(X, Y, w)[0]
        elif i is not None and j is None:
            return self.forward(X[i], Y[i], w)[0]
        else:
            if isinstance(j, int):
                j = [j]
            return self.forward(X[i][j], Y[i][j], w)[0]

    def forward(self, X, Y, w):
        print(f"Forward pass called with X shape: {X.shape}, Y shape: {Y.shape}, w shape: {w.shape}")
        
        X = self.preprocess_input(X)
        Y = torch.tensor(Y, dtype=torch.long)
        
        params = self.unpack_w(w)
        for param, new_param in zip(self.model.parameters(), params):
            param.data.copy_(new_param)

        outputs = self.model(X)
        loss = self.criterion(outputs, Y.argmax(dim=1))
        return loss.item(), outputs

    def forward_backward(self, X, Y, w):
        print(f"Forward-backward pass called with X shape: {X.shape}, Y shape: {Y.shape}, w shape: {w.shape}")
        
        X = self.preprocess_input(X)
        Y = torch.tensor(Y, dtype=torch.long)
        
        params = self.unpack_w(w)
        for param, new_param in zip(self.model.parameters(), params):
            param.data.copy_(new_param)
            param.grad = None

        outputs = self.model(X)
        loss = self.criterion(outputs, Y.argmax(dim=1))
        loss.backward()

        grads = [param.grad.flatten() for param in self.model.parameters()]
        self._dw = torch.cat(grads)

        print(f"Gradients computed with shape: {self._dw.shape}")

        return self._dw.cpu().numpy(), loss.item()
    
    def accuracy(self, w, split='test'):
        print(f"Accuracy called with w shape: {w.shape}, split: {split}")
        
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
            print('Data split %s is not supported' % split)

        X = self.preprocess_input(X)
        Y = torch.tensor(Y, dtype=torch.long)

        loss, outputs = self.forward(X, Y, w)
        pred = outputs.argmax(dim=1)

        accuracy = (pred == torch.tensor(labels)).float().mean().item()
        print(f"Accuracy computed: {accuracy}, Loss: {loss}")

        return accuracy, loss


if __name__ == '__main__':
    p = NN()
    print("NN problem instance created.")