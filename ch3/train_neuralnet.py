import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from common.Animator import Animator
from common.Accumulator import Accumulator
from dataset.fashion_mnist import load_fashion_mnist, get_fashion_mnist_labels
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, t):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(t.dtype) == t
    return float(cmp.type(torch.float32).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 设置为评估模式
    metric = Accumulator(2) # 正确预测的数量、预测的总数量
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确率总和、样本数
    metric = Accumulator(3)
    for x_train, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(x_train)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(x_train.shape[0])
        # 计算训练损失和训练准确率
        with torch.no_grad():
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    global train_metrics
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    train_loss_list = []
    train_acc_list = []
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        print(train_metrics, end='\n')
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss_list.append(train_metrics[0])
        train_acc_list.append(train_metrics[1])
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    x = list(range(1, num_epochs + 1))
    plt.plot(x, train_loss_list, label='train loss')
    plt.plot(x, train_acc_list, label='train acc')
    plt.legend()
    plt.show()
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    global x, y
    for x, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    train_iter, test_iter = load_fashion_mnist(batch_size=256)
    num_inputs, num_hidden, num_outputs = 784, 256, 10
    net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 200
    train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
    predict_ch3(net, test_iter)
