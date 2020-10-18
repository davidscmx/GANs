import torch
from torch import nn
from gradient_penalty import *
import unittest
from generator import Generator
from critic import Critic

class test_gradient_penalty(unittest.TestCase):

    def test_get_gradient(self):
        image_shape = (256, 1, 28, 28)
        device = 'cuda'
        z_dim = 64
        gen = Generator(z_dim).to(device)
        crit = Critic().to(device) 
        real = torch.randn(*image_shape, device=device) + 1
        fake = torch.randn(*image_shape, device=device) - 1
        epsilon_shape = [1 for _ in image_shape]
        epsilon_shape[0] = image_shape[0]
        epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
        gradient = get_gradient(crit, real, fake, epsilon)
        self.assertEqual(tuple(gradient.shape),image_shape)
        self.assertGreater(gradient.max(),0)
        self.assertLess(gradient.min(),0)

    def test_gradient_penalty(self):
        image_shape = (256, 1, 28, 28)
        device = 'cuda'
        z_dim = 64
        gen = Generator(z_dim).to(device)
        crit = Critic().to(device) 
        real = torch.randn(*image_shape, device=device) + 1
        fake = torch.randn(*image_shape, device=device) - 1
        epsilon_shape = [1 for _ in image_shape]
        epsilon_shape[0] = image_shape[0]
        epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
        
        bad_gradient = torch.zeros(*image_shape)
        bad_gradient_penalty = gradient_penalty(bad_gradient)
        image_size = torch.prod(torch.Tensor(image_shape[1:]))
        good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
        good_gradient_penalty = gradient_penalty(good_gradient)
        
        random_gradient = get_gradient(crit, real, fake, epsilon)        
        random_gradient_penalty = gradient_penalty(random_gradient)

        self.assertTrue(torch.isclose(bad_gradient_penalty, torch.tensor(1.)))
        self.assertTrue(torch.isclose(good_gradient_penalty, torch.tensor(0.)))
        
        
        #TODO 
        #self.assertLess(torch.abs(random_gradient_penalty - 1),0.1)

if __name__ == '__main__':
    unittest.main()