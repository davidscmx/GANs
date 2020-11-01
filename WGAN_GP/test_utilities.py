import torch
from torch import nn
from utilities import *
import unittest

class test_utilities(unittest.TestCase):
    def test_get_gen_loss(self):
        self.assertEqual(get_one_hot_labels(labels=torch.Tensor([[0, 2, 1]]).long(), n_classes=3).tolist(),
                        [[[1, 0, 0], [0, 0, 1], [0, 1, 0]]])

    def test_combine_vectors(self):        
        combined = combine_vectors(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]));
        # Check exact order of elements
        self.assertTrue(torch.all(combined == torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]])))
        # Tests that items are of float type
        self.assertEqual(type(combined[0][0].item()), float)
        
        # Check shapes
        combined = combine_vectors(torch.randn(1, 4, 5), torch.randn(1, 8, 5));
        self.assertEqual(tuple(combined.shape),(1, 12, 5))
        self.assertEqual(tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12).long()).shape),(1, 30, 12))
        
    def test_get_input_dimensions(self):
        gen_dim, disc_dim = get_input_dimensions(23, (12, 23, 52), 9)
        self.assertEqual(gen_dim,32)
        self.assertEqual(disc_dim,21)
        
