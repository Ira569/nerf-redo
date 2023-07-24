from torch.utils.data import Dataset
import cv2
import os


# 证明可以正常读图片
# img_path = 'data/nerf_synthetic/lego/test/r_0.png'
# img_cv = cv2.imread(img_path)
# num =0
# for i in range(800):
#     for j in range(800):
#         for k in range(3):
#             if img_cv[i][j][k]>0:
#                 num =num+1
# print(800*800*3)
# print(num)
class LegoImg(Dataset):
    def __init__(self,root_dir= 'data/nerf_synthetic/lego',label_dir = 'train'):


        self.root_dir =root_dir
        self.label_dir =label_dir
        self.path = os.path.join(root_dir,label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,'r_',)+'.png'
        img = cv2.imread(img_item_path)
        label = self.label_dir
        return img
    def __len__(self):
        return len(self.img_path)

train_dataset = LegoImg()
val_dataset =LegoImg(label_dir='val')

img = train_dataset[0]
print(len(train_dataset))
print(len(val_dataset))


