import numpy as np

class Conv2D:
    def __init__(self, weight, bias, stride=1, padding=0):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def pad_input(self, x):
        if self.padding == 0:
            return x
        return np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def forward(self, x):
        """
        Pure NumPy implementation of 2D convolution.
        x: (C_in, H_in, W_in)
        returns: (C_out, H_out, W_out)
        """
        C_out, C_in, kH, kW = self.weight.shape
        x_padded = self.pad_input(x)
        _, H_in, W_in = x.shape
        H_out = (H_in + 2*self.padding - kH) // self.stride + 1
        W_out = (W_in + 2*self.padding - kW) // self.stride + 1

        out = np.zeros((C_out, H_out, W_out))

        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    for ic in range(C_in):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = x_padded[ic, h_start:h_start+kH, w_start:w_start+kW]
                        out[oc, i, j] += np.sum(patch * self.weight[oc, ic])
                    out[oc, i, j] += self.bias[oc]
        return out

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class MaxPool2D:
    def forward(self, x):
        h, w = x.shape
        out = np.zeros((h // 2, w // 2))
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                out[i//2, j//2] = np.max(x[i:i+2, j:j+2])
        return out
    
class AveragePool2D:
    def forward(self, x):
        h, w = x.shape
        out = np.zeros((h // 2, w // 2))
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                out[i//2, j//2] = np.average(x[i:i+2, j:j+2])
        return out

class Flatten:
    def forward(self, x):
        return x.flatten()

class Dense:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return np.dot(self.weight, x) + self.bias