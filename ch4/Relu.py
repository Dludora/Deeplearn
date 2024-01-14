import torch
import torch.nn as nn
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    y.sum().backward()
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), x.grad.detach().numpy())
    plt.legend(['relu', 'grad'])
    plt.show()

