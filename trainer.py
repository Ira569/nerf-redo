import time

from  inference import render_path
import tqdm
from tqdm import trange

from render import *

from network import *
import torch
from read_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeRF().to(device)
print(model)

grad_vars = list(model.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=0.0005, betas=(0.9, 0.999))

img2mse = lambda x, y : torch.mean((x - y) ** 2)
# img_loss = img2mse(rgb, target_s)

def create_log_files(basedir,expname,args):
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
from nerf_helper import *
def train():
    from opts import config_parser
    parser = config_parser()
    args = parser.parse_args()


    images, poses, render_poses,hwf, i_split = read_lego_data()

    i_train,i_val,i_test =i_split
    near = 2
    far = 6

    images = images[...,:3]

    H,W,focal= hwf
    H,W = int(H),int(W)
    hwf =[H,W,focal]

    K=np.array([
        [focal,0,0.5*W],
        [0,focal,0.5*H],
        [0,0,1]
    ])
    render_poses = torch.Tensor(render_poses).to(device)

    basedir = args.basedir
    expname = args.expname

    create_log_files(basedir,expname,args)

    render_kwargs_train, render_kwargs_test,start,grad_vars,optimizer = create_nerf(args)

    global_step =start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # --------------------------------------------------------------------------------------------------------

    # Short circuit if only rendering out from trained model
    # 这里会使用render_poses
    # if args.render_only:
    #     # 仅进行渲染，不进行训练
    #     print('RENDER ONLY')
    #     run_render_only(args, images, i_test, basedir, expname, render_poses, hwf, K, render_kwargs_test, start)
    #     return

    # --------------------------------------------------------------------------------------------------------

    # Prepare ray batch tensor if batching random rays
    N_rand = args.N_rand

    use_batching = not args.no_batching

    # 统一一个时刻放入cuda
    # Move training data to GPU

    poses = torch.Tensor(poses).to(device)

    # --------------------------------------------------------------------------------------------------------

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # 训练部分的代码
    # 两万次迭代
    # 可能是强迫症，不想在保存文件的时候，出现19999这种数字
    N_iters = 200000 + 1
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]  # [400,400,3] 图像内容
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        if N_rand is not None: #2d -> 3d
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            # precrop_iters: number of steps to train on central crops
            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(# H=400，coords取样100-300
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW), indexing='ij',
                    ), -1)
                if i == start:
                    print(
                        f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                                    torch.linspace(0, W - 1, W), indexing='ij'),
                                     -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            # 选出的像素坐标
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)  # 堆叠 o和d
            # target 也同样选出对应位置的点
            # target 用来最后的mse loss 计算
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # rgb 网络计算出的图像
        # 前三是精细网络的输出内容，其他的还保存在一个dict中，有5项
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # 计算loss
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        # 计算指标
        psnr = mse2psnr(img_loss)

        # rgb0 粗网络的输出
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 学习率衰减
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # 保存模型
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                # 运行的轮次数目
                'global_step': global_step,
                # 粗网络的权重
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 精细网络的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 优化器的状态
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 生成测试视频，使用的是render_poses (这个不等同于test数据)
        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # 360度转一圈的视频
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # 执行测试，使用测试数据
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # 用时
        dt = time.time() - time0
        # 打印log信息的频率
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt}")

        global_step += 1

    # for batch, (X, y) in enumerate(dataloader):
    #     X, y = X.to(device), y.to(device)
    #
    #     # Compute prediction error
    #     pred = model(X)
    #     # loss = loss_fn(pred, y)
    #
    #     # Backpropagation
    #     # loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()