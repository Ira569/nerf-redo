import torch
import numpy as np

def get_rays(H,W,K,c2w):
    '''
    TODO
    :param H:
    :param W:
    :param K: 相机内参
    :param c2w: 相机到世界坐标系的转换
    :return:
    '''
# j
# [0,0.....]
# [1,1,......]
# [W-1,....]
# i
# [0,..,H-1]
# [0,..,H-1]
# [0,..,H-1]

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i = i.t()
    j = j.t()
    # [400,400,3]
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dirs [400,400,3] -> [400,400,1,3]
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d [400,400,3]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 前三行，最后一列，定义了相机的平移，因此可以得到射线的原点o
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
import numpy as np
def get_rays_np(H,W,K,c2w):
    # 与上面的方法相似，这个是使用的numpy，上面是使用的torch
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

