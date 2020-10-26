import torch
from torch import nn
from losses import *
import unittest

class test_losses(unittest.TestCase):
    def test_get_gen_loss(self):
        self.assertTrue(torch.isclose(get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05))

    def test_get_critic_loss(self):
        self.assertTrue(torch.isclose(get_crit_loss(torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), 0.1),torch.tensor(-0.7)))
        self.assertTrue(torch.isclose(get_crit_loss(torch.tensor(20.), torch.tensor(-20.), torch.tensor(2.), 10),torch.tensor(60.)))

if __name__ == '__main__':
    unittest.main()