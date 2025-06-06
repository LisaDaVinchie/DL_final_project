import unittest
import torch as th

import sys
import os

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from model import DummyModel

class TestDummyModel(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.n_channels = 3
        
        self.nrows = 10
        self.ncols = 10
        
        self.model = DummyModel()
        
    def test_unet_forward(self):
        input_shape = (self.batch_size, self.n_channels, self.nrows, self.ncols)
        
        mask_size = 3
        input_tensor = th.rand(input_shape)
        mask_tensor = th.ones((self.batch_size, 1, self.nrows, self.ncols), dtype=th.bool)
        mask_tensor[:, 0, :mask_size, :mask_size] = 0
        
        means = []
        for i in range(self.n_channels):
            valid_values = input_tensor[:, i, :, :][mask_tensor[:, 0, :, :]]
            means.append(valid_values.mean())

        expected_output = input_tensor.clone()
        for i in range(self.n_channels):
            expected_output[:, i, :mask_size, :mask_size] = means[i]
        
        output_tensor, output_mask = self.model(input_tensor, mask_tensor)
        
        self.assertEqual(output_tensor.shape, (input_shape))
        self.assertEqual(output_mask.shape, mask_tensor.shape)
        
        self.assertTrue(th.allclose(output_tensor, expected_output, atol=1e-6))