import torch
from pytorch_models.tools import args
from pytorch_models.resnet50 import ResNet50
from pytorch_models.dataloader import load_data


opt = args.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for batch_size in [16,32,64,128,256,512,1024,2048]:
  dataloader = load_data('mnist', batch_size=batch_size, num_workers=opt.n_workers)
  model = ResNet50(dataloader, device, verbose=False)
  print(model.train(epochs=opt.epochs))