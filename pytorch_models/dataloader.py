import torch
from .dataset import Dataset

def load_data(dataset, batch_size, num_workers):
  dataset = Dataset().load(dataset)
  splits = ['train', 'valid']
  shuffle = {'train': True, 'valid': False}

  dataloader = {
    x: torch.utils.data.DataLoader(
      dataset = dataset[x],
      batch_size = batch_size,
      shuffle = shuffle[x],
      num_workers = num_workers,
      drop_last = shuffle[x],
    )
    for x in splits
  }
  return dataloader