import sys
import timm
import torch
from tqdm import tqdm
import torch.nn as nn
import dataloader
import init

class ResNet50:
  def __init__(self, dataloader, device):
    self.init_model()
    self.dataloader = dataloader
    self.device = device
    self.loss_fn = nn.CrossEntropyLoss()

  def init_model(self):
    self.model = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=10)

  def train(self, epochs):
    for epoch in range(epochs):
      print(f'epoch: {epoch}')
      # self.train_one_epoch()
      self.eval()

  def train_one_epoch(self):
    for image, label in tqdm(self.dataloader['train'], desc='train'):
      image = image.to(self.device).float()
      label = label.to(self.device)
      
      pred = self.model(image)
      loss = self.loss_fn(pred, label)
      loss.backward()

  def eval(self):
    with torch.no_grad():
      for image, label in tqdm(self.dataloader['valid'], desc='eval'):
        image = image.to(self.device)
        label = label.to(self.device)

        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        correct = torch.argmax(pred, 1) == label
      print(f'val: {torch.sum(correct)/len(correct)*100:.2f}%, {loss:.4f}')

if __name__ == '__main__':
  opt = init.init()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = ResNet50(dataloader.load_data('mnist', batch_size=opt.batch_size, num_workers=opt.n_workers), device)
  print(device)
  model.train(epochs=10)
