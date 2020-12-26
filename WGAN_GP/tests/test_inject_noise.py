
import torch
from torch import nn
import unittest
import sys
sys.path.append("../building_blocks/")
import unittest

from inject_noise import InjectNoise

class TestInjectNoise(unittest.TestCase):
    def test_inject_noise(self):
        test_noise_channels = 3000
        test_noise_samples = 20
        fake_images = torch.randn(test_noise_samples, test_noise_channels, 10, 10)
        inject_noise = InjectNoise(test_noise_channels)
        self.assertLess(torch.abs(inject_noise.weight.std() - 1), 0.1)
        self.assertLess(torch.abs(inject_noise.weight.mean()), 0.1 )
        self.assertEqual(type(inject_noise.get_weight()),torch.nn.parameter.Parameter)    
        self.assertEqual(tuple(inject_noise.weight.shape), (1, test_noise_channels, 1, 1))
        inject_noise.weight = nn.Parameter(torch.ones_like(inject_noise.weight))
        # Check that something changed
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images)).mean(), 0.1)
        
        # Check that the change is per-channel
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(0)).mean(), 1e-4)
        self.assertLess(torch.abs((inject_noise(fake_images) - fake_images).std(1)).mean(), 1e-4)
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(2)).mean(), 1e-4)
        self.assertGreater(torch.abs((inject_noise(fake_images) - fake_images).std(3)).mean(), 1e-4)
        # Check that the per-channel change is roughly normal
        per_channel_change = (inject_noise(fake_images) - fake_images).mean(0).std()
      
        # assert per_channel_chan > 0.9 and per_channel_change < 1.1
        # Make sure that the weights are being used at all
        inject_noise.weight = nn.Parameter(torch.zeros_like(inject_noise.weight))
        self.assertLess( torch.abs((inject_noise(fake_images) - fake_images)).mean(), 1e-4)
        self.assertEqual(len(inject_noise.weight.shape),4)

if __name__ == '__main__':
    unittest.main()