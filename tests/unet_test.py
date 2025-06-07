import unittest
import torch as th

import sys
import os

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from model import UNet, UNetLite, EncoderLayer, DecoderLayer

class TestModel(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.n_channels = 3
        
        self.nrows = 64
        self.ncols = 64
        
        self.model_params = {
            "e_filters": [3, 64, 128, 256, 512, 512, 512, 512, 512],
            "d_filters": [512, 512, 512, 512, 256, 128, 64, 3, 3],
            "e_kernels": [3, 3, 3, 3, 3, 3, 3, 3],
            "d_kernels": [3, 3, 3, 3, 3, 3, 3, 3, 1],
            "e_bn" : [0, 1, 1, 1, 1, 1, 1, 1],
            "d_bn" : [1, 1, 1, 1, 1, 1, 1, 0],
            "e_strides": [2, 2, 2, 2, 2, 2, 2, 2],
            "d_strides": [2, 2, 2, 2, 2, 2, 2, 2]
        }
        
    def test_encoder_layer_output_shape(self):
        in_channels = 3
        out_channels = 64
        kernel_size = 7
        
        layer = EncoderLayer(in_filters=in_channels, out_filters=out_channels, kernel_size=kernel_size)
        
        input_tensor = th.rand(self.batch_size, in_channels, self.nrows, self.ncols)
        mask_tensor = th.ones_like(input_tensor)
        
        output_tensor, output_mask = layer(input_tensor, mask_tensor)
        
        self.assertEqual(output_tensor.shape, (self.batch_size, out_channels, self.nrows // 2, self.ncols // 2))
        self.assertEqual(output_mask.shape, (self.batch_size, out_channels, self.nrows // 2, self.ncols // 2))
        
    def test_decoder_layer_output_shape(self):
        in_channels = 512
        out_channels = 256
        kernel_size = 3
        
        layer = DecoderLayer(in_filters=(in_channels + out_channels), out_filters=out_channels, kernel_size=kernel_size)
        
        input_tensor = th.rand((4, in_channels, 32, 32))
        mask_tensor = th.ones_like(input_tensor)
        tensor2 = th.rand((4, out_channels, 64, 64))
        mask_tensor2 = th.ones_like(tensor2)
        
        output_tensor, output_mask = layer(input_tensor, mask_tensor, tensor2, mask_tensor2)
        
        self.assertEqual(output_tensor.shape, (4, out_channels, 64, 64))
        self.assertEqual(output_mask.shape, (4, out_channels, 64, 64))
        
    def test_unet_forward(self):
        input_shape = (32, 3, 256, 256)
        
        model = UNet(self.model_params)
        
        input_tensor = th.rand(input_shape)
        mask_tensor = th.ones_like(input_tensor)
        
        output_tensor = model(input_tensor, mask_tensor)
        
        self.assertEqual(output_tensor.shape, (input_shape))
        
    def test_unet_lite_forward(self):
        input_shape = (32, 3, 32, 32)
        
        model = UNetLite()
        
        input_tensor = th.rand(input_shape)
        mask_tensor = th.ones_like(input_tensor)
        
        output_tensor = model(input_tensor, mask_tensor)
        
        self.assertEqual(output_tensor.shape, (input_shape))