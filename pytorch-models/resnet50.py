import timm
import torch
from tqdm import tqdm
import torch.nn as nn
import dataloader


class ResNet50:
  def __init__(self, dataloader, device):
    self.init_model()
    self.dataloader = dataloader
    self.device = device
    self.loss_fn = nn.CrossEntropyLoss()

  def init_model(self):
    self.model = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=10)
    print(self.model)
    # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    # self.model.bn1 = nn.BatchNorm2d(28)
    self.model.fc = nn.Linear(in_features=2048, out_features=10)

  def train(self, epochs):
    for epoch in range(epochs):
      print(f'epoch: {epoch}')
      self.train_one_epoch()
      self.eval()

  def train_one_epoch(self):
    for image, label in tqdm(self.dataloader['train']):
      image = image.to(self.device).float()
      label = label.to(self.device)
      
      pred = self.model(image)
      loss = self.loss_fn(pred, label)
      loss.backward()

  def eval(self):
    with torch.no_grad():
      for image, label in tqdm(self.dataloader['valid']):
        image = image.to(self.device)
        label = label.to(self.device)

        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        correct = torch.argmax(pred, 1) == label
        print(f'val: {correct}, {loss}')

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = ResNet50(dataloader.load_data('mnist', 16, 4), 'cpu')
  print(device)
  model.train(epochs=10)
