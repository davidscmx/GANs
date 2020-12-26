from generator import *
from critic import *
import torch
from torch import nn
import unittest

class test_critic(unittest.TestCase):
    num_test = 100
    gen = Generator()
    critic = Critic()
    test_images = gen(get_noise(num_test, gen.z_dim))
    
    ## Test the hidden block
    def test_hidden_block(self):
        test_hidden_block = test_critic.critic.make_crit_block(1, 5, kernel_size=6, stride=3)
        hidden_output = test_hidden_block(test_critic.test_images)
        self.assertEqual(hidden_output.shape, (test_critic.num_test, 5, 8, 8))
        
        # Because of the LeakyReLU slope
        self.assertGreater( -hidden_output.min() / hidden_output.max(), 0.15)
        self.assertLess(-hidden_output.min() / hidden_output.max(), 0.25)
        self.assertGreater(hidden_output.std(),0.5)
        self.assertLess(hidden_output.std(),1)
    
    ## Test the final block    
    def test_final_block(self):  
        #Test your make_disc_block() function
        test_final_block = test_critic.critic.make_crit_block(1, 10, kernel_size=2, stride=5, final_layer=True)
        final_output = test_final_block(test_critic.test_images)

        self.assertEqual(tuple(final_output.shape),(test_critic.num_test, 10, 6, 6))
        self.assertGreater(final_output.max(),1.0)
        self.assertLess(final_output.min(),-1.0)
        self.assertGreater(final_output.std(),0.3)
        self.assertLess(final_output.std(),0.6)
    
    ## Test the whole block
    def test_whole_block(self):
        critic_output = test_critic.critic(test_critic.test_images)
        self.assertEqual(tuple(critic_output.shape),(test_critic.num_test, 1))
        self.assertGreater(critic_output.std(), 0.25)
        self.assertLess(critic_output.std(),0.5)

if __name__ == '__main__':
    unittest.main()