import timm

model = timm.create_model(model_name='resnet50', pretrained=True)
print(model)