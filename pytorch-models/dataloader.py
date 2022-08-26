import os
from torchvision.datasets import MNIST, ImageNet
from torchvision.transforms import transforms

class Dataset:
  def __init__(self):
    self.data_root = './datasets/'

    self.candidates = {
      'mnist': self._load_mnist,
      'cifar10': self._load_cifar10,
      'imagenet': self._load_imagenet
    }
  
  def load(self, dataset: str):
    assert dataset in self.candidates, f"Available datasets are [{' '.join(self.candidates.keys())}]"
    return self.candidates[dataset]()

  def _load_mnist(self):
    path = os.path.join(self.data_root, 'mnist')
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5), (1.0))
    ])
    train = MNIST(path, transform=transform, train=True, download=True)
    valid = MNIST(path, transform=transform, train=False, download=True)
    # test = MNIST(path, transform=transform, train=False, download=True)
    # return train, valid, test
    return train, valid

  def _load_cifar10(self):
    print('cifar')

  def _load_imagenet(self):
    path = os.path.join(self.data_root, 'imagenet')
    os.mkdir(path)
    train = ImageNet(path, 'train')
    valid = ImageNet(path, 'val')
    return train, valid


if __name__ == '__main__':
  print(Dataset().load('imagenet'))