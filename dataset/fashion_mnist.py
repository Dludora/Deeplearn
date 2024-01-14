import torchvision
from torchvision import transforms
from torch.utils import data
import os

dataset_dir = os.path.dirname(os.path.abspath(__file__))


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def download_fashion_mnist(trans):
    data_path = dataset_dir + "/data"
    train_mnist = torchvision.datasets.FashionMNIST(
        root=data_path, train=True, transform=trans, download=True)
    test_mnist = torchvision.datasets.FashionMNIST(
        root=data_path, train=False, transform=trans, download=True)

    return train_mnist, test_mnist


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train_mnist, test_mnist = download_fashion_mnist(trans)

    return (data.DataLoader(train_mnist, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(test_mnist, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()))
