from torch import nn
import torch.nn.functional as F
import torch

L_embed =10
L_view_embed=4
def pos_enc(x):
    # 2-60 L=10 ; 3-24 L=4
    rets = []
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.concat(rets, -1)

def view_enc(x):
    # 2-60 L=10 ; 3-24 L=4
    rets = []
    for i in range(L_view_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.concat(rets, -1)
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



def run_network(inputs,viewdirs,fn,embed_fn,embeddirs_fn,netchunk=1024*64):
    """
        被下面的create_nerf 封装到了lambda方法里面
        Prepares inputs and applies network 'fn'.
        inputs: pts，光线上的点 如 [1024,64,3]，1024条光线，一条光线上64个点
        viewdirs: 光线起点的方向
        fn: 神经网络模型 粗糙网络或者精细网络
        embed_fn:
        embeddirs_fn:
        netchunk:
        """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N_rand*64,3]
    # 坐标点进行编码嵌入 [N_rand*64,63]
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rand*64,27]
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 里面经过网络 [bs*64,3]
    outputs_flat = batchify(fn, netchunk)(embedded)
    # [bs*64,4] -> [bs,64,4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=24, output_ch=4, skips=[4], use_viewdirs=False,):
        super(NeRF,self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # input_ch - w + 4个 W-W + 一个 W+input_ch -W 加三个 567 W-W，然后是输出密度 然后加上方向输出RGB
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        # 方向24 + 256 - 128
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # feature 是什么层没看懂
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)


    def forword(self,x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        alpha = self.alpha_linear(h) # 256-1
        feature = self.feature_linear(h) #256-256
        h = torch.cat([feature, input_views], -1) # 256+24

        for i, l in enumerate(self.views_linears): #256+24-128
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h) # 128-3
        outputs = torch.cat([rgb, alpha], -1)

        return outputs

from trainer import device
def create_nerf(args):
    embed_fn = pos_enc
    input_ch =3
    input_ch_view =3
    embeddirs_fn =view_enc
    output_ch =4

    model = NeRF().to(device)
    grad_vars =list(model.parameters())

    model_fine = NeRF()
    grad_vars += list(model_fine.parameters())
    # XYZ, dirs, network_fn(nerf)
    network_query_fn = lambda inputs,viewdirs,network_fn: run_network(inputs,viewdirs,network_fn,embed_fn =embed_fn,embeddirs_fn = embeddirs_fn,netchunk = args.netchunk)

    optimizer = torch.optim.Adam(params=grad_vars, lr=0.0005, betas=(0.9, 0.999))
    start = 0
    basedir = args.basedir
    expname = args.expname

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