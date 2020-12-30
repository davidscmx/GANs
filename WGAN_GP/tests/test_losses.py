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

    def test_identity_loss(self):
        test_real_X = torch.tensor(83.)
        test_gen_YX = lambda x: x * 89
        test_identity_criterion = lambda x, y: (x + y) * 73
        test_res = get_identity_loss(test_real_X, test_gen_YX, test_identity_criterion)
        self.assertLess(torch.abs(test_res[0] - 545310), 1e-6)
        self.assertLess(torch.abs(test_res[1] - 7387), 1e-6)

    def test_cycle_consistency_loss(self):
        test_real_X = torch.tensor(83.)
        test_fake_Y = torch.tensor(97.)
        test_gen_YX = lambda x: x * 89
        test_cycle_criterion = lambda x, y: (x + y) * 73
        test_res = get_cycle_consistency_loss(test_real_X, test_fake_Y, test_gen_YX, test_cycle_criterion)
        self.assertLess(torch.abs(test_res[1] - 8633), 1e-6)
        self.assertLess(torch.abs(test_res[0] - 636268), 1e-6)
    
    def test_get_total_gen_loss_cycle_gan(self):

        test_real_A = torch.tensor(97)
        test_real_B = torch.tensor(89)
        test_gen_AB = lambda x: x * 83
        test_gen_BA = lambda x: x * 79
        test_disc_A = lambda x: x * 47
        test_disc_B = lambda x: x * 43
        test_adv_criterion = lambda x, y: x * 73 + y * 71
        test_recon_criterion = lambda x, y: (x + y) * 61
        test_lambda_identity = 59
        test_lambda_cycle = 53
        
        test_res = get_total_gen_loss_cycle_gan(
        test_real_A, 
        test_real_B, 
        test_gen_AB, 
        test_gen_BA, 
        test_disc_A,
        test_disc_B,
        test_adv_criterion, 
        test_recon_criterion, 
        test_recon_criterion, 
        test_lambda_identity, 
        test_lambda_cycle)

        self.assertEqual(test_res[0].item(), 4048102400)
        self.assertEqual(test_res[1].item(), 7031)
        self.assertEqual(test_res[2].item(), 8051)

if __name__ == '__main__':
    unittest.main()