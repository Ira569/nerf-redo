import torch
from network import *
import numpy as np
def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        # 以chunk分批进入网络，防止显存爆掉，然后在拼接
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def create_nerf():
    model = NeRF()
    grad_vars = list(model.parameters())
    model_fine = NeRF()
    grad_vars += list(model_fine.parameters())
    network_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=pos_enc,
                                                                        embeddirs_fn=view_enc,
                                                                        netchunk=1023*64)
    optimizer = torch.optim.Adam(params=grad_vars,lr = 5e-4,betas = (0.9,0.999))
    start = -1
    # netchunk 是网络中处理的点的batch_size
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn)
    render_kwargs_train = {
        # 精细网络
        'network_fine': model_fine,
        # 粗网络
        'network_fn': model,
        'network_query_fn':network_query_fn
    }
    #    if args.dataset_type != 'llff' or args.no_ndc:
    #     print('Not ndc!')
    #     render_kwargs_train['ndc'] = False
    # render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return render_kwargs_train, render_kwargs_test,start, grad_vars, optimizer


def get_rays_torch(H, W, K, c2w):
    """
    K：相机内参矩阵
    c2w: 相机到世界坐标系的转换
    """
    # j
    # [0,......]
    # [1,......]
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


def get_rays_np(H, W, K, c2w):
    # 与上面的方法相似，这个是使用的numpy，上面是使用的torch
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    bins: z_vals_mid
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    # 归一化 [bs, 62]
    # 概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 累积分布函数
    cdf = torch.cumsum(pdf, -1)
    # 在第一个位置补0
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])  # [bs,128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF

    u = u.contiguous()
    # u 是随机生成的
    # 找到对应的插入的位置
    inds = torch.searchsorted(cdf, u, right=True)
    # 前一个位置，为了防止inds中的0的前一个是-1，这里就还是0
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 最大的位置就是cdf的上限位置，防止过头，跟上面的意义相同
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # (batch, N_samples, 2)
    inds_g = torch.stack([below, above], -1)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # (batch, N_samples, 63)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # 如[1024,128,63] 提取 根据 inds_g[i][j][0] inds_g[i][j][1]
    # cdf_g [1024,128,2]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 如上, bins 是从2到6的采样点，是64个点的中间值
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 差值
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # 防止过小
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom

    # lower+线性插值
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# no use


import os
def create_log_files(basedir, expname, args):
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # 保存一份参数文件
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # 保存一份配置文件
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    return basedir, expname
# import imageio
# def run_render_only(args, images, i_test, basedir, expname, render_poses, hwf, K, render_kwargs_test, start):
#     with torch.no_grad():
#         if args.render_test:
#             # render_test switches to test poses
#             images = images[i_test]
#         else:
#             # Default is smoother render_poses path
#             images = None
#
#         testsavedir = os.path.join(basedir, expname,
#                                    'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
#         os.makedirs(testsavedir, exist_ok=True)
#         print('test poses shape', render_poses.shape)
#
#         rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
#                               savedir=testsavedir, render_factor=args.render_factor)
#         print('Done rendering', testsavedir)
#         imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)