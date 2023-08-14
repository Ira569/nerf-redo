import math
from math import sin, cos

from network import *
import torch
import unittest

class TestNetwork(unittest.TestCase):
    def test_encoder(self):
        x=[-1,0,1]
        x = torch.Tensor(x)
        pos = pos_enc(x)
        view =view_enc(x)
        self.assertEqual(torch.numel(pos), 63)
        self.assertEqual(torch.numel(view), 27)


    def test_value(self):
        # sin1 = 0.84  cos1=0.54 sin0=0 cos0=1
        # [-1,0,1] -> [-1 0 1 ][-0.8415,  0.0000,  0.8415] [0.5403,  1.0000,  0.5403] ...
        # -0.8415,  0.0000,  0.8415,  0.5403,  1.0000,  0.5403,
        x=[-1,0,1]
        x = torch.Tensor(x)
        pos = pos_enc(x)
        view =view_enc(x)
        print(pos)
        self.assertTrue(math.fabs(pos[3].item() - sin(-1)) < 1e-5)
        self.assertTrue(math.fabs(pos[4].item() - sin(0)) < 1e-5)
        self.assertTrue(math.fabs(pos[5].item() - sin(1)) < 1e-5)

        self.assertTrue(math.fabs(pos[0].item() + 1) < 1e-5)
        self.assertTrue(math.fabs(pos[1].item() - 0) < 1e-5)
        self.assertTrue(math.fabs(pos[2].item() - 1) < 1e-5)

        self.assertTrue(math.fabs(view[3].item() - sin(-1)) < 1e-5)
        self.assertTrue(math.fabs(view[4].item() - sin(0)) < 1e-5)
        self.assertTrue(math.fabs(view[5].item() - sin(1)) < 1e-5)

        self.assertTrue(math.fabs(pos[0].item() + 1) < 1e-5)
        self.assertTrue(math.fabs(pos[1].item() - 0) < 1e-5)
        self.assertTrue(math.fabs(pos[2].item() - 1) < 1e-5)
    def test_NeRF(self):
        pos = [1,2,3]
        view = [4,5,6]

        pos = torch.Tensor(pos)
        view = torch.Tensor(view)
        pos = pos_enc(pos)
        view = view_enc(view)
        x =torch.cat([pos,view],dim=-1)
        print(x)
        print(x.shape)#63 + 27 = 90
        x= x[None,:]
        print(x.shape)
        print(x)
        self.assertTrue(x.shape[1] == 90)
        myModel = NeRF()
        y = myModel(x)
        print(y.shape)
        print(y)

        self.assertTrue(y.shape[1] == 4)