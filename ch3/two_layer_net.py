import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = torch.normal(0, weight_init_std, (input_size, hidden_size), requires_grad=True)
        self.params['b1'] = torch.zeros(hidden_size, requires_grad=True)
        self.params['W2'] = torch.normal(0, weight_init_std, (hidden_size, output_size), requires_grad=True)
        self.params['b2'] = torch.zeros(output_size, requires_grad=True)

