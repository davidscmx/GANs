from generator import *

import torch
from torch import nn
import unittest

class test_generator(unittest.TestCase):
    gen = Generator()
    num_test = 100

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