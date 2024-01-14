import torch
import torch.nn as nn
from dataset.fashion_mnist import load_fashion_mnist
from common.util import train
from d2l import torch as d2l


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


train_iter, test_iter = load_fashion_mnist(batch_size=256)
net = nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

if __name__ == "__main__":
    lr, num_epochs = 0.9, 10
    train(net, train_iter, test_iter, num_epochs=num_epochs, lr=lr, device=d2l.try_gpu())
