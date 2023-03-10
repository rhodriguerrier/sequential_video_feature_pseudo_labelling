import torch
import torch.nn as nn


class FlatMLP(nn.Module):
    def __init__(self):
        super(FlatMLP, self).__init__()
        self.rgb_linear_small = nn.Linear(5120, 8)
        self.flow_linear_small = nn.Linear(5120, 8)

    def forward(self, rgb_ft, flow_ft):
        rgb_output = self.rgb_linear_small(rgb_ft)
        flow_output = self.flow_linear_small(flow_ft)
        return rgb_output, flow_output
