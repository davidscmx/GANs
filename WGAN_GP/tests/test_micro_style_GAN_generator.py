import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../building_blocks/")
import unittest

from micro_style_GAN_generator import MicroStyleGANGenerator
from micro_style_GAN_generator_block import MicroStyleGANGeneratorBlock
from utilities import get_truncated_noise

class TestMicroStyleGANGenerator(unittest.TestCase):
    def test_micro_style_GAN_generator(self):    
        z_dim = 128
        out_chan = 3
        truncation = 0.7

        mu_stylegan = MicroStyleGANGenerator(
            z_dim=z_dim, 
            map_hidden_dim=1024,
            w_dim=496,
            in_chan=512,
            out_chan=out_chan, 
            kernel_size=3, 
            hidden_chan=256
        )

        test_samples = 10
        test_result = mu_stylegan(get_truncated_noise(test_samples, z_dim, truncation))

        # Check if the block works
        self.assertEqual(tuple(test_result.shape), (test_samples, out_chan, 16, 16))

        # Check that the interpolation is correct
        mu_stylegan.alpha = 1.
        test_result, _, test_big =  mu_stylegan(
            get_truncated_noise(test_samples, z_dim, truncation), 
            return_intermediate=True)
        self.assertLess( torch.abs(test_result - test_big).mean(), 0.001)
        
        mu_stylegan.alpha = 0.
        test_result, test_small, _ =  mu_stylegan(
            get_truncated_noise(test_samples, z_dim, truncation), 
            return_intermediate=True)
        self.assertLess(torch.abs(test_result - test_small).mean(), 0.001)

if __name__ == "__main__":
    unittest.main()