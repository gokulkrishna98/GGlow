import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, input)

traced_script_module.save("../models/mobilenetv2.pt")
    