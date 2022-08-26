import torch
from dataset import Dataset

def __init__(self, dataset):
    dataset = Dataset().load(dataset)
    splits = ['train', 'test']
    shuffle = {'train': True, 'test': False}

return dataset['']