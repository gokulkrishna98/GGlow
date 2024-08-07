import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, input)

traced_script_module.save("../models/resnet18.pt")