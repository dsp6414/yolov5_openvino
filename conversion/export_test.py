import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import onnx
import onnxruntime

import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scale=3, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        sh = torch.tensor(x.shape)
        return F.interpolate(x, size=(int(sh[2]*self.scale), int(sh[3]*self.scale)), mode=self.mode, align_corners=self.align_corners)

model = Upsample()
model.eval()
x = torch.rand((1, 3, 200, 320))
torch.onnx._export(model, x, "weights/test.onnx", verbose=True, opset_version=10)

with torch.no_grad():
    torch_out = model(x)

ort_session = onnxruntime.InferenceSession("weights/test.onnx")
ort_input = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
ort_out = ort_session.run(None, ort_input)[0]
