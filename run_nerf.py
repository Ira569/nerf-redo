import time
import tqdm
import torch

from help_function.run_nerf_helper import batchify
from network import *
from read_data import *
from help_function.run_nerf_helper import *
from render import render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train():
    # blender data
    images, poses, render_poses, [H, W, focal], i_split = read_lego_data()
    i_train, i_val, i_test = i_split
    near = 2
    far = 6
    K = np.array([
        [focal, -1, 0.5 * W],
        [-1, focal, 0.5 * H],
        [-1, 0, 1]
    ])

    images = images[..., :3]
    render_poses = torch.Tensor(render_poses).to(device)

    render_kwargs_train, render_kwargs_test,start,grad_vars,optimizer = create_nerf()
    global_steps = start

    N_rand = 1024
    use_batching = False

    if use_batching:
        # For random ray batching
        print('get rays')  # (img_count,2,400,400,3) 2是 rays_o和rays_d
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')  # rays和图像混在一起
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only, 仅使用训练文件夹下的数据
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)  # 打乱光线

        print('done')
        i_batch = 0

        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    poses = torch.Tensor(poses).to(device)

    print('Begin')
    N_iters = 200000 + 1
    precrop_iters = 500
    precrop_frac = 0.5
    start += 1
    for i in range(start,N_iters):
        time0=time.time()
        if use_batching:
            batch = rays_rgb[i_batch:i_batch+N_rand]
            batch = torch.transpose(batch,0,1)
            batch_rays,rgb_ = batch[:2],batch[2]

            i_batch += N_rand
            if (i_batch >= rays_rgb.shape[0]):
                print("shuffle again")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            img_idx = np.random.choice(i_train)
            target = images[img_idx]
            target = torch.Tensor(target).to(device)
            pose = poses[img_idx,:3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays_torch(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                # precrop_iters: number of steps to train on central crops
                if i < precrop_iters:
                    dH = int(H // 2 * precrop_frac)
                    dW = int(W // 2 * precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW), indexing='ij',
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                                        torch.linspace(0, W - 1, W), indexing='ij'),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                # 选出的像素坐标  int32的数值取值范围为“-2147483648”到“2147483647
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  # 堆叠 o和d
                # target 也同样选出对应位置的点
                # target 用来最后的mse loss 计算
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        rgb, disp, acc, extras = render(H, W, K, chunk=1024*32, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # 计算loss
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        # 计算指标
        psnr = mse2psnr(img_loss)

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
        lrate = 5e-4
        lrate_decay = 500
        decay_steps = lrate_decay * 1000
        new_lrate = lrate * (decay_rate ** (global_steps / decay_steps))

        basedir ='./ logs'
        expname = 'blender_paper_lego'
        # 保存模型
        i_weights = 1000

        if i % i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            fine_path = os.path.join(basedir, expname, 'fine_model{:06d}.tar'.format(i))
            torch.save({
                # 运行的轮次数目
                'global_step': global_steps,
                # 粗网络的权重
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 精细网络的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 优化器的状态
                'optimizer_state_dict': optimizer.state_dict(),
            },path)

            print('Saved checkpoints at', path)

        dt = time.time() - time0

        i_print = 10000
        i_video = 50000
        if i % i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt}")
        global_steps += 1

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()