# Test the mapping function

import torch
from torch import nn
import unittest
import sys
sys.path.append("../building_blocks/")
import unittest

from mapping_layers import MappingLayers

class TestMappingLayers(unittest.TestCase):
    def test_mapping_layers(self):
        map_fn = MappingLayers(10, 20, 30)
        self.assertEqual(tuple(map_fn(torch.randn(2, 10)).shape),(2, 30))
        self.assertGreater(len(map_fn.mapping), 4)
        
        outputs = map_fn(torch.randn(1000, 10))
        self.assertTrue(outputs.std() > 0.05 and outputs.std() < 0.3)
        self.assertTrue(outputs.min() > -2 and outputs.min() < 0)
        self.assertTrue(outputs.max() < 2 and outputs.max() > 0)
        layers = [str(x).replace(' ', '').replace('inplace=True', '') for x in map_fn.get_mapping()]

        self.assertEqual (layers, ['Linear(in_features=10,out_features=20,bias=True)', 
                          'ReLU()', 
                          'Linear(in_features=20,out_features=20,bias=True)', 
                          'ReLU()', 
                          'Linear(in_features=20,out_features=30,bias=True)'])


if __name__ == '__main__':
    unittest.main()