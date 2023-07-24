from torch import nn
import torch.nn.functional as F
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,):
        super(NeRF,self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])


        self.linear1 = nn.Linear(3,)

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

    def forword(self,x):
        x = F.relu(self.linear1(x))
