import torch
from torch import nn
from utilities import *
import unittest
from classifier import *

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
    
    def test_calculate_updated_noise_basic(self):
        device = 'cuda'
        label_indices = range(40)
        classifier = Classifier(n_classes=len(label_indices)).to(device)
        opt = torch.optim.Adam(classifier.parameters(), lr=0.01)
        opt.zero_grad()
        noise = torch.ones(20, 20) * 2
        noise.requires_grad_()
        fake_classes = (noise ** 2).mean()
        fake_classes.backward()
        new_noise = calculate_updated_noise(noise, 0.1)
        self.assertEqual(type(new_noise),torch.Tensor)
        self.assertEqual(tuple(new_noise.shape),(20, 20))
        self.assertEqual(new_noise.max(),2.0010)
        self.assertEqual(new_noise.min(),2.0010)
        self.assertTrue(torch.isclose(new_noise.sum(), torch.tensor(0.4) + 20 * 20 * 2))
        
    #def test_calculate_updated_noise_generated(self):
    #    # Check that it works for generated images
    #    opt.zero_grad()
    #    noise = get_noise(32, z_dim).to(device).requires_grad_()
    #    fake = gen(noise)
    #    fake_classes = classifier(fake)[:, 0]
    #    fake_classes.mean().backward()
    #    noise.data = calculate_updated_noise(noise, 0.01)
    #    fake = gen(noise)
    #    fake_classes_new = classifier(fake)[:, 0]
    #    assert torch.all(fake_classes_new > fake_classes)
#
    #def test_get_score():
    #    # UNIT TEST
    #    assert torch.isclose(
    #        get_score(torch.ones(4, 3), torch.zeros(4, 3), [0], [1, 2], 0.2), 
    #        1 - torch.sqrt(torch.tensor(2.)) * 0.2
    #    )
    #    rows = 10
    #    current_class = torch.tensor([[1] * rows, [2] * rows, [3] * rows, [4] * rows]).T.float()
    #    original_class = torch.tensor([[1] * rows, [2] * rows, [3] * rows, [4] * rows]).T.float()
    #    
    #    # Must be 3
    #    assert get_score(current_class, original_class, [1, 3] , [0, 2], 0.2).item() == 3
    #    
    #    current_class = torch.tensor([[1] * rows, [2] * rows, [3] * rows, [4] * rows]).T.float()
    #    original_class = torch.tensor([[4] * rows, [4] * rows, [2] * rows, [1] * rows]).T.float()
    #    
    #    # Must be 3 - 0.2 * sqrt(10)
    #    assert torch.isclose(get_score(current_class, original_class, [1, 3] , [0, 2], 0.2), 
    #                 -torch.sqrt(torch.tensor(10.0)) * 0.2 + 3)