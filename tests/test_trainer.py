import pytest
import torch
from torch import nn, optim

from torchvision.datasets import MNIST
from torchvision import transforms

from src.metrics import AccumulatedAccuracyMetric
from src.networks import SimpleNet
from src.trainer import train_epoch

cuda = torch.cuda.is_available()


@pytest.fixture
def mnist_loader():
    mean, std = 0.1307, 0.3081
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_dataset = MNIST('./MNIST', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))
                          ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader


def test_train_epoch(mnist_loader):
    log_interval = 1
    lr = 1e-2
    im_size = 28
    metrics = [AccumulatedAccuracyMetric()]

    model = SimpleNet(im_chan=im_size*im_size, n_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, metrics = train_epoch(mnist_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

    assert total_loss > 0

