import timm
from dataloader import Dataset

train, val = Dataset().load('mnist')

model = timm.create_model(model_name='resnet50', pretrained=True)
print(dir(model))