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
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
def create_nerf(args):
    # embed_fn, input_ch = pos_enc,3
    #
    # input_ch_views = 0
    # embeddirs_fn = view_enc
    # if args.use_viewdirs:
    #     embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    #
    # # 想要=5生效，首先需要use_viewdirs=False and N_importance>0
    # output_ch = 5 if args.N_importance > 0 else 4
    # skips = [4]
    # 粗网络
    model = NeRF().to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 精细网络
        model_fine = NeRF().to(device)
        # 模型参数
        grad_vars += list(model_fine.parameters())

    # netchunk 是网络中处理的点的batch_size
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=pos_enc,
                                                                        embeddirs_fn=view_enc,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    # 优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)

    # load参数
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        # 精细网络
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        # 粗网络
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    print(model_fine)

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


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
    theta: -181 -- +180，间隔为9  传进去正负好像也只影响旋转顺序了，所以没什么问题。
    phi: 固定值 -31
    radius: 固定值 3
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

