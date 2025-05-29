import numpy as np

class MySequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Conv2DScratch:
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

class ReLUScratch:
    def forward(self, x):
        return np.maximum(0, x)

class MaxPool2DScratch:
    def forward(self, x):
        # x: (C, H, W)
        C, H, W = x.shape
        out = np.zeros((C, H // 2, W // 2))
        for c in range(C):
            for i in range(0, H, 2):
                for j in range(0, W, 2):
                    out[c, i//2, j//2] = np.max(x[c, i:i+2, j:j+2])
        return out

class AveragePool2DScratch:
    def forward(self, x):
        # x: (C, H, W)
        C, H, W = x.shape
        out = np.zeros((C, H // 2, W // 2))
        for c in range(C):
            for i in range(0, H, 2):
                for j in range(0, W, 2):
                    out[c, i//2, j//2] = np.mean(x[c, i:i+2, j:j+2])
        return out

class FlattenScratch:
    def forward(self, x):
        return x.flatten()

class DenseScratch:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return np.dot(self.weight, x) + self.bias