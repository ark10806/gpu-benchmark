import timm
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam

from .tools import args, helper
from . import dataloader

class ResNet50:
  def __init__(self, dataloader, device, n_gpu=1, verbose=False):
    self.v = verbose
    self.dataloader = dataloader
    self.device = device
    self.init_model()
    if (1 < n_gpu <= torch.cuda.device_count()) and self.device=='cuda':
      print(f'using gpu {list(range(n_gpu))}')
      self.model = nn.DataParallel(self.model, device_ids=list(range(n_gpu)))

  def statistics(func):
    def wrapper(self):
      self.tps = helper.TPS()
      func( self )
    return wrapper

  def init_model(self):
    self.loss_fn = nn.CrossEntropyLoss()
    self.model = timm.create_model('resnet50', pretrained=True, in_chans=1, num_classes=10).to(self.device)
    self.optim = Adam(self.model.parameters(), lr=1e-4)
    print(self.device)

  def train(self, epochs):
    train_tps = []
    valid_tps = []
    for epoch in range(epochs):
      if self.v: print(f'{"[epoch " + str(epoch+1)+"]":-^100}')
      self.train_one_epoch()
      train_tps.append(self.tps.eval())
      self.eval()
      valid_tps.append(self.tps.eval())
      if self.v: print('-' * 100, end='\n\n')
    return {'train': helper.avg(train_tps), 'valid': helper.avg(valid_tps)}

  @statistics
  def train_one_epoch(self):
    pbar = tqdm(self.dataloader['train'], desc='train')
    for image, label in pbar:
      self.tps.append(len(label))
      self.optim.zero_grad()
      image = image.to(self.device).float()
      label = label.to(self.device)
      
      pred = self.model(image)
      loss = self.loss_fn(pred, label)
      loss.mean().backward()
      self.optim.step()
      pbar.set_postfix({'tps': f'{self.tps.eval():.2f}'})


  @statistics
  def eval(self):
    with torch.no_grad():
      pbar = tqdm(self.dataloader['valid'], desc='eval ')
      for image, label in pbar:
        self.tps.append(len(label))
        image = image.to(self.device)
        label = label.to(self.device)

        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        correct = torch.argmax(pred, 1) == label
        pbar.set_postfix({'tps': f'{self.tps.eval():.2f}'})
      if self.v: print(f'> acc: {torch.sum(correct)/len(correct)*100:.2f}%\t loss: {loss:.4f}')

if __name__ == '__main__':
  opt = args.init()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = ResNet50(dataloader.load_data('mnist', batch_size=opt.batch_size, num_workers=opt.n_workers), device, verbose=True)
  print(model.train(epochs=opt.epochs))