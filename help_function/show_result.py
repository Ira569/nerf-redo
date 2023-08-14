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


def pose_spherical(theta, phi, radius):
    """
    theta: -180 -- +180，间隔为9  传进去正负好像也只影响旋转顺序了，所以没什么问题。
    phi: 固定值 -30
    radius: 固定值 4
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w