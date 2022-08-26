import torch
from math import log

from pytorch_models.tools import args
from pytorch_models.resnet50 import ResNet50
from pytorch_models.dataloader import load_data


opt = args.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

output = []
if opt.batch_size == -1:
  batch_list = [(2)**x for x in range(int(log(opt.min_batch_size,2)), int(log(opt.max_batch_size,2))+1)]
else:
  batch_list = [opt.batch_size]
n_gpu = torch.cuda.device_count() if opt.n_gpu == -1 else opt.n_gpu
print(f'{n_gpu} X [{torch.cuda.get_device_name(0)}]')

for batch_size in batch_list:
  dataloader = load_data('mnist', batch_size=batch_size, num_workers=opt.n_workers)
  print(f'batch_size: {batch_size}')
  model = ResNet50(dataloader, device, n_gpu, verbose=True)
  tps_res = model.train(epochs=opt.epochs)
  output.append( (batch_size, tps_res['train']) )


print(f'{n_gpu} X [{torch.cuda.get_device_name(0)}]')
print(f'{"batch_size":<10}\t TPS')
for batch_size, tps in output:
  print(f'{batch_size:<10}:\t {tps}')