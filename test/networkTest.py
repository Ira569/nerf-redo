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
        self.assertEqual(torch.numel(pos), 60)
        self.assertEqual(torch.numel(view), 24)


    def test_value(self):
        # sin1 = 0.84  cos1=0.54 sin0=0 cos0=
        # [-1,0,1] -> [-0.8415,  0.0000,  0.8415] [0.5403,  1.0000,  0.5403] ...
        # -0.8415,  0.0000,  0.8415,  0.5403,  1.0000,  0.5403,
        x=[-1,0,1]
        x = torch.Tensor(x)
        pos = pos_enc(x)
        view =view_enc(x)
        self.assertTrue(math.fabs(pos[0].item() - sin(-1)) < 1e-5)
        self.assertTrue(math.fabs(pos[1].item() - sin(0)) < 1e-5)
        self.assertTrue(math.fabs(pos[2].item() - sin(1)) < 1e-5)

        self.assertTrue(math.fabs(view[0].item() - sin(-1)) < 1e-5)
        self.assertTrue(math.fabs(view[1].item() - sin(0)) < 1e-5)
        self.assertTrue(math.fabs(view[2].item() - sin(1)) < 1e-5)
