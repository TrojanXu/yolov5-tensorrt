import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, size, scale, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        sh = torch.tensor(x.shape)
        return F.interpolate(x, size=(int(sh[2]*self.scale), int(sh[3]*self.scale)), mode=self.mode, align_corners=self.align_corners)