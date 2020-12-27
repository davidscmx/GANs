import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../building_blocks/")
import unittest

from micro_style_GAN_generator_block import MicroStyleGANGeneratorBlock

class TestMicroStyleGANGeneratorBlock(unittest.TestCase):
    def test_micro_style_GAN_generator_block(self):    
        test_stylegan_block = MicroStyleGANGeneratorBlock(in_chan=128, out_chan=64, w_dim=256, kernel_size=3, starting_size=8)
        test_x = torch.ones(1, 128, 4, 4)
        test_x[:, :, 1:3, 1:3] = 0
        test_w = torch.ones(1, 256)
        test_x = test_stylegan_block.upsample(test_x)
        
        self.assertEqual(tuple(test_x.shape), (1, 128, 8, 8))
        self.assertLess (torch.abs(test_x.mean() - 0.75), 1e-4)
        
        test_x = test_stylegan_block.conv(test_x)
        
        self.assertEqual(tuple(test_x.shape), (1, 64, 8, 8))
        
        test_x = test_stylegan_block.inject_noise(test_x)
        test_x = test_stylegan_block.activation(test_x)
        
        self.assertLess(test_x.min(), 0)
        self.assertLess( -test_x.min() / test_x.max(), 0.4)
        
        test_x = test_stylegan_block.adain(test_x, test_w) 
        foo = test_stylegan_block(torch.ones(10, 128, 4, 4), torch.ones(10, 256))



if __name__ == "__main__":
    unittest.main()