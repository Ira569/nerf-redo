# 先用lego的试一下吧
import torch
import numpy as np
import os
import json
import imageio
import cv2


def load_blender_data(basedir,half_res=True, testskip=1):
    '''
    load blender(artifect) data from basedir
    :param basedir: the base directory for data
    :param half_res:
    :param testskip:
    :return:
    '''
    splits = ['train','val','test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,'transforms_{0}.json'.format(s)),'r') as fp:
            metas[s] =json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # testskip 跳着选测试集数据
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    i_split = [np.arange(counts[i],counts[i+1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs,0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # a= [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]]
    # b=[pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)]
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    render_poses = None

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split