

from generator import Generator, get_noise
import torch
from torch import nn
import unittest
class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


class test_critic(unittest.TestCase):
    #Test your make_disc_block() function
    num_test = 100
    gen = Generator()
    critic = Critic()
    test_images = gen(get_noise(num_test, gen.z_dim))
    
    ## Test the hidden block
    def test_hidden_block(self):
          #Test your make_disc_block() function
        num_test = 100
        gen = Generator()
        critic = Critic()
        test_images = gen(get_noise(num_test, gen.z_dim))
        test_hidden = critic.make_crit_block(1, 5, kernel_size=6, stride=3)
        hidden_output = test_hidden_block(test_images)
        self.assertEqual(hidden_output.shape, (num_test, 5, 8, 8))
        
        # Because of the LeakyReLU slope
        self.assertGreater( -hidden_output.min() / hidden_output.max(), 0.15)
        self.assertLess(-hidden_output.min() / hidden_output.max(), 0.25)
        self.assertGreater(hidden_output.std(),0.5)
        self.assertLess(hidden_output.std(),1)
    
    ## Test the final block    
    def test_final_block(self):  
          #Test your make_disc_block() function
        num_test = 100
        gen = Generator()
        critic = Critic()
        test_images = gen(get_noise(num_test, gen.z_dim))  
        test_final_block = critic.make_crit_block(1, 10, kernel_size=2, stride=5, final_layer=True)
        final_output = test_final_block(test_images)

        self.assertEqual(tuple(final_output.shape),(num_test, 10, 6, 6))
        self.assertGreater(final_output.max(),1.0)
        self.assertLess(final_output.min(),-1.0)
        self.assertGreater(final_output.std(),0.3)
        self.assertLess(final_output.std(),0.6)
    
    ## Test the whole block
    def test_whole_block(self):
          #Test your make_disc_block() function
        num_test = 100
        gen = Generator()
        critic = Critic()
        test_images = gen(get_noise(num_test, gen.z_dim))
        critic_output = Critic(test_images)
        self.assertEqual(tuple(critic_output.shape) == (num_test, 1))
        self.assertGreater(critic_output.std(), 0.25)
        self.assertLess(critic_output.std(),0.5)

if __name__ == '__main__':
    unittest.main()