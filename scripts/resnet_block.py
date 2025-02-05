import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # input_tensor_info = torch.tensor([1, 3, 128, 128], dtype=torch.int64)
        self.input_tensor_info = [1, 3, 128, 128]
        # self.register_buffer("input_tensor_info", input_tensor_info)
        

    def forward(self, x):
        identity = x 
        # First convolution + batch norm + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Second convolution + batch norm
        out = self.conv2(out)
        out = self.bn2(out)
        # Add the residual connection
        # out += identity
        # out = self.relu(out)
        return out


model = ResNetBlock(3, 3, 1)
traced_script_module = torch.jit.script(model)
graph = traced_script_module.graph
print(graph)

traced_script_module.save("../models/resnet_block.pt")