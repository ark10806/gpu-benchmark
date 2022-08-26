import torch
from pytorch_models.tools import args
from pytorch_models.resnet50 import ResNet50
import pytorch_models.dataloader as dataloader


opt = args.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50(dataloader.load_data('mnist', batch_size=opt.batch_size, num_workers=opt.n_workers), device, verbose=True)
print(model.train(epochs=opt.epochs))