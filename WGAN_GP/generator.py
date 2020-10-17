
import torch
from torch import nn
import unittest

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
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
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


class test_generator(unittest.TestCase):
    gen = Generator()
    num_test = 100

    # Test the hidden block
    
    #hidden_output = test_hidden_block(test_uns_noise)
    #hidden_output = test_hidden(test_uns_noise)

    # Test the hidden block    
    def test_hidden_block(self):        
        test_hidden_noise = get_noise(test_generator.num_test, test_generator.gen.z_dim)
        test_hidden_block = test_generator.gen.make_gen_block(10, 20, kernel_size=4, stride=1)
        test_uns_noise = test_generator.gen.unsqueeze_noise(test_hidden_noise)
        hidden_output = test_hidden_block(test_uns_noise)
        test_hidden = hidden_output     

        self.assertEqual(tuple(test_hidden.shape),(test_generator.num_test, 20, 4, 4))
        self.assertGreater(test_hidden.max(),1)
        self.assertEqual(test_hidden.min(),0)
        self.assertGreater(test_hidden.std(),0.2)
        self.assertLess(test_hidden.std(),1)
        self.assertGreater(test_hidden.std(),0.5)

    def test_strides(self):
        # Check that it works with other strides
        test_hidden_noise = get_noise(test_generator.num_test, test_generator.gen.z_dim)
        test_hidden_block = test_generator.gen.make_gen_block(10, 20, kernel_size=4, stride=1)
        test_uns_noise = test_generator.gen.unsqueeze_noise(test_hidden_noise)
        hidden_output = test_hidden_block(test_uns_noise)

        test_hidden_block_stride = test_generator.gen.make_gen_block(20, 20, kernel_size=4, stride=2)
        self.assertEqual(tuple(test_hidden_block_stride(hidden_output).shape), (test_generator.num_test, 20, 10, 10))

    def test_final_block(self):
        test_final_noise = get_noise(test_generator.num_test, test_generator.gen.z_dim) * 20
        test_final_block = test_generator.gen.make_gen_block(10, 20, final_layer=True)
        test_final_uns_noise = test_generator.gen.unsqueeze_noise(test_final_noise)        

        final_output = test_final_block(test_final_uns_noise)
        self.assertEqual(final_output.max().item(), 1)
        self.assertEqual(final_output.min().item(),-1)
    
    def test_whole_block(self):
        ## Test the whole thing:
        test_gen_noise = get_noise(test_generator.num_test, test_generator.gen.z_dim)
        test_uns_gen_noise = test_generator.gen.unsqueeze_noise(test_gen_noise)
        gen_output = test_generator.gen(test_uns_gen_noise)
    
        self.assertEqual(tuple(gen_output.shape), (test_generator.num_test, 1, 28, 28))
        self.assertGreater(gen_output.std(),0.5)
        self.assertLess(gen_output.std(), 0.8)

if __name__ == '__main__':
    unittest.main()