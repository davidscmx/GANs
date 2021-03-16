import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../building_blocks/")
import unittest

from adain import AdaIn

class TestAdaIn(unittest.TestCase):

    def test_adain_1(self):    
        w_channels = 50
        image_channels = 20
        image_size = 30
        n_test = 10
    
        adain = AdaIn(image_channels, w_channels)
        test_w = torch.randn(n_test, w_channels)
        self.assertEqual(adain.style_scale_transform(test_w).shape, adain.style_shift_transform(test_w).shape)
        self.assertEqual(adain.style_scale_transform(test_w).shape[-1],image_channels)
        self.assertEqual(tuple(adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape),(n_test, image_channels, image_size, image_size))        
    
    def test_adain_2(self):    
        w_channels = 3
        image_channels = 2
        image_size = 3
        n_test = 1
        adain = AdaIn(image_channels, w_channels)

        adain.style_scale_transform.weight.data = torch.ones_like(adain.style_scale_transform.weight.data) / 4
        adain.style_scale_transform.bias.data = torch.zeros_like(adain.style_scale_transform.bias.data)
        adain.style_shift_transform.weight.data = torch.ones_like(adain.style_shift_transform.weight.data) / 5
        adain.style_shift_transform.bias.data = torch.zeros_like(adain.style_shift_transform.bias.data)
        test_input = torch.ones(n_test, image_channels, image_size, image_size)
        test_input[:, :, 0] = 0
        test_w = torch.ones(n_test, w_channels)
        test_output = adain(test_input, test_w)
        
        self.assertLess(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))), 1e-4)
        self.assertLess(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))), 1e-4)

if __name__ == "__main__":
    unittest.main()