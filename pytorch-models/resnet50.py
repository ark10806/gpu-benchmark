import timm
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam

import init
import dataloader
from helper import TPS

class ResNet50:
  def __init__(self, dataloader, device):
    self.dataloader = dataloader
    self.device = device
    self.loss_fn = nn.CrossEntropyLoss()
    self.init_model()
    self.optim = Adam(self.model.parameters(), lr=1e-4)

  def statistics(func):
    def wrapper(self):
      self.tps = TPS()
      func( self )
    return wrapper

  def init_model(self):
    self.model = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=10).to(self.device)

  def train(self, epochs):
    for epoch in range(epochs):
      print(f'epoch: {epoch}')
      self.train_one_epoch()
      self.eval()

  @statistics
  def train_one_epoch(self):
    pbar = tqdm(self.dataloader['train'], desc='train')
    for image, label in pbar:
      self.tps.append(len(label))
      image = image.to(self.device).float()
      label = label.to(self.device)
      
      pred = self.model(image)
      loss = self.loss_fn(pred, label)
      loss.mean().backward()
      self.optim.step()
      pbar.set_postfix({'tps': self.tps.eval()})


  @statistics
  def eval(self):
    with torch.no_grad():
      pbar = tqdm(self.dataloader['valid'], desc='eval')
      for image, label in pbar:
        self.tps.append(len(label))
        image = image.to(self.device)
        label = label.to(self.device)

        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        correct = torch.argmax(pred, 1) == label
        pbar.set_postfix({'tps': self.tps.eval()})
      print(f'val: {torch.sum(correct)/len(correct)*100:.2f}%, {loss:.4f}')

if __name__ == '__main__':
  opt = init.init()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = ResNet50(dataloader.load_data('mnist', batch_size=opt.batch_size, num_workers=opt.n_workers), device)
  print(device)
  model.train(epochs=10)
