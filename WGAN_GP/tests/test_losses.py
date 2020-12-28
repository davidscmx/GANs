import torch
from torch import nn

import sys
sys.path.append("../building_blocks/")

from losses import *
import unittest

class test_losses(unittest.TestCase):
    def test_get_gen_loss(self):
        self.assertTrue(torch.isclose(get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05))

    def test_get_critic_loss(self):
        self.assertTrue(torch.isclose(get_crit_loss(torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), 0.1),torch.tensor(-0.7)))
        self.assertTrue(torch.isclose(get_crit_loss(torch.tensor(20.), torch.tensor(-20.), torch.tensor(2.), 10),torch.tensor(60.)))

    def test_get_disc_loss_cycle_gan(self):
        test_disc_X = lambda x: x * 97
        test_real_X = torch.tensor(83.)
        test_fake_X = torch.tensor(89.)
        test_adv_criterion = lambda x, y: x * 79 + y * 73

        self.assertLess(torch.abs((get_disc_loss_cycle_gan(test_real_X, test_fake_X, test_disc_X, test_adv_criterion)) - 659054.5000), 1e-6)
        test_disc_X = lambda x: x.mean(0, keepdim=True)
        test_adv_criterion = torch.nn.BCEWithLogitsLoss()
        test_input = torch.ones(20, 10)
        # If this runs, it's a pass - checks that the shapes are treated correctly
        get_disc_loss_cycle_gan(test_input, test_input, test_disc_X, test_adv_criterion)

    def test_get_gen_loss_cycle_gan(self):
        test_disc_Y = lambda x: x * 97
        test_real_X = torch.tensor(83.)
        test_gen_XY = lambda x: x * 89
        test_adv_criterion = lambda x, y: x * 79 + y * 73
        test_res = get_gen_adversarial_loss_cycle_gan(test_real_X, test_disc_Y, test_gen_XY, test_adv_criterion)
        self.assertLess(torch.abs(test_res[0] - 56606652), 1e-6)
        self.assertLess( torch.abs(test_res[1] - 7387), 1e-6)
        test_disc_Y = lambda x: x.mean(0, keepdim=True)
        test_adv_criterion = torch.nn.BCEWithLogitsLoss()
        test_input = torch.ones(20, 10)
        # If this runs, it's a pass - checks that the shapes are treated correctly
        get_gen_adversarial_loss_cycle_gan(test_input, test_disc_Y, test_gen_XY, test_adv_criterion)

if __name__ == '__main__':
    unittest.main()