# 先用lego的试一下吧

import torch
import numpy as np


# 平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# 绕x轴的旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# 绕y轴的旋转
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, np.sin(th), 0],
    [0, 1, 0, 0],
    [-np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

print(torch.Tensor([1,1]).float())
