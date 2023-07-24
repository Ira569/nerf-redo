import unittest
from load_blender_data import *

lego_dir = './data/nerf_synthetic/lego'

class TestLoadBlender(unittest.TestCase):
    def test_load_blender_data(self):
        load_blender_data(lego_dir)
    def test_load2(self):
        self.assertEqual(1,1)


if __name__ == "__main__":
    unittest.main()