import torch as th
import torch.nn as nn
from PartialConv import PartialConv2d

class UNet(nn.Module):
    def __init__(self, model_params: dict = None):
        super(UNet, self).__init__()
        
        self.model_params = model_params
        if self.model_params is not None:
            self.import_params(self.model_params)
        
        self.layer_setup()
        self.print_shapes = True  # Set to True to print shapes during forward pass
        
    def import_params(self, params: dict):
        self.e_filters = list(params["e_filters"])
        self.e_kernels = list(params["e_kernels"])
        self.d_filters = list(params["d_filters"])
        self.d_kernels = list(params["d_kernels"])
        self.e_bn = list(params["e_bn"])
        self.d_bn = list(params["d_bn"])
        self.e_bn = [bool(b) for b in self.e_bn]
        self.d_bn = [bool(b) for b in self.d_bn]
        self.e_strides = list(params["e_strides"])
        self.d_strides = list(params["d_strides"])

    def _print_shape(self, name, tensor):
        if self.print_shapes:
            print(f"{name} shape: {tensor.shape}")
    
    def layer_setup(self):
        if self.model_params is None:
            self.e_filters = [3, 64, 128, 256, 512, 512, 512, 512, 512]
            self.d_filters = [512, 512, 512, 512, 256, 128, 64, 3, 3]
            self.e_kernels = [3, 3, 3, 3, 3, 3, 3, 3]
            self.d_kernels = [3, 3, 3, 3, 3, 3, 3, 3, 1]
            self.e_bn = [False, True, True, True, True, True, True, True]
            self.d_bn = [True, True, True, True, True, True, True, False]
            self.e_strides = [2, 2, 2, 2, 2, 2, 2, 2]
            self.d_strides = [2, 2, 2, 2, 2, 2, 2, 2]
        
        self.enc1 = EncoderLayer(in_filters=self.e_filters[0], out_filters=self.e_filters[1], kernel_size=self.e_kernels[0], stride=self.e_strides[0], bn=self.e_bn[0])
        self.enc2 = EncoderLayer(in_filters=self.e_filters[1], out_filters=self.e_filters[2], kernel_size=self.e_kernels[1], stride=self.e_strides[1], bn=self.e_bn[1])
        self.enc3 = EncoderLayer(in_filters=self.e_filters[2], out_filters=self.e_filters[3], kernel_size=self.e_kernels[2], stride=self.e_strides[2], bn=self.e_bn[2])
        self.enc4 = EncoderLayer(in_filters=self.e_filters[3], out_filters=self.e_filters[4], kernel_size=self.e_kernels[3], stride=self.e_strides[3], bn=self.e_bn[3])
        self.enc5 = EncoderLayer(in_filters=self.e_filters[4], out_filters=self.e_filters[5], kernel_size=self.e_kernels[4], stride=self.e_strides[4], bn=self.e_bn[4])
        self.enc6 = EncoderLayer(in_filters=self.e_filters[5], out_filters=self.e_filters[6], kernel_size=self.e_kernels[5], stride=self.e_strides[5], bn=self.e_bn[5])
        self.enc7 = EncoderLayer(in_filters=self.e_filters[6], out_filters=self.e_filters[7], kernel_size=self.e_kernels[6], stride=self.e_strides[6], bn=self.e_bn[6])
        self.enc8 = EncoderLayer(in_filters=self.e_filters[7], out_filters=self.e_filters[8], kernel_size=self.e_kernels[7], stride=self.e_strides[7], bn=self.e_bn[7])
        
        self.dec9 = DecoderLayer(in_filters=self.e_filters[8] + self.e_filters[7], out_filters=self.d_filters[0], kernel_size=self.d_kernels[0], bn=self.d_bn[0], leaky_relu=True)
        self.dec10 = DecoderLayer(in_filters=self.d_filters[0] + self.e_filters[6], out_filters=self.d_filters[1], kernel_size=self.d_kernels[1], bn=self.d_bn[1], leaky_relu=True)
        self.dec11 = DecoderLayer(in_filters=self.d_filters[1] + self.e_filters[5], out_filters=self.d_filters[2], kernel_size=self.d_kernels[2], bn=self.d_bn[2], leaky_relu=True)
        self.dec12 = DecoderLayer(in_filters=self.d_filters[2] + self.e_filters[4], out_filters=self.d_filters[3], kernel_size=self.d_kernels[3], bn=self.d_bn[3], leaky_relu=True)
        self.dec13 = DecoderLayer(in_filters=self.d_filters[3] + self.e_filters[3], out_filters=self.d_filters[4], kernel_size=self.d_kernels[4], bn=self.d_bn[4], leaky_relu=True)
        self.dec14 = DecoderLayer(in_filters=self.d_filters[4] + self.e_filters[2], out_filters=self.d_filters[5], kernel_size=self.d_kernels[5], bn=self.d_bn[5], leaky_relu=True)
        self.dec15 = DecoderLayer(in_filters=self.d_filters[5] + self.e_filters[1], out_filters=self.d_filters[6], kernel_size=self.d_kernels[6], bn=self.d_bn[6], leaky_relu=True)
        self.dec16 = DecoderLayer(in_filters=self.d_filters[6] + self.e_filters[0], out_filters=self.d_filters[7], kernel_size=self.d_kernels[7], bn=self.d_bn[7], leaky_relu=True)
        self.output = nn.Conv2d(self.d_filters[8], self.d_filters[8], kernel_size=self.d_kernels[8])
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        x1, mask1 = self.enc1(x, mask)
        x2, mask2 = self.enc2(x1, mask1)
        x3, mask3 = self.enc3(x2, mask2)
        x4, mask4 = self.enc4(x3, mask3)
        x5, mask5 = self.enc5(x4, mask4)
        x6, mask6 = self.enc6(x5, mask5)
        x7, mask7 = self.enc7(x6, mask6)
        x8, mask8 = self.enc8(x7, mask7)
        
        x9, mask9 = self.dec9(x8, mask8, x7, mask7)
        x10, mask10 = self.dec10(x9, mask9, x6, mask6)
        x11, mask11 = self.dec11(x10, mask10, x5, mask5)
        x12, mask12 = self.dec12(x11, mask11, x4, mask4)
        x13, mask13 = self.dec13(x12, mask12, x3, mask3)
        x14, mask14 = self.dec14(x13, mask13, x2, mask2)
        x15, mask15 = self.dec15(x14, mask14, x1, mask1)
        x16, mask16 = self.dec16(x15, mask15, x, mask)
        
        x = self.output(x16)
        x = self.sigmoid(x)
        
        return x, mask16
    
class UNetLite(nn.Module):
    def __init__(self, model_params: dict = None):
        super(UNetLite, self).__init__()
        
        self.model_params = model_params
        if self.model_params is not None:
            self.import_params(self.model_params)
        
        self.layer_setup()
        self.print_shapes = False  # Set to True to print shapes during forward pass
        
    def import_params(self, params: dict):
        self.e_filters = list(params["e_filters"])
        self.e_kernels = list(params["e_kernels"])
        self.d_filters = list(params["d_filters"])
        self.d_kernels = list(params["d_kernels"])
        self.e_bn = list(params["e_bn"])
        self.d_bn = list(params["d_bn"])
        self.e_bn = [bool(b) for b in self.e_bn]
        self.d_bn = [bool(b) for b in self.d_bn]
        self.e_strides = list(params["e_strides"])
        self.d_strides = list(params["d_strides"])

    def _print_shape(self, name, tensor):
        if self.print_shapes:
            print(f"{name} shape: {tensor.shape}")
        
    def layer_setup(self):
        if self.model_params is None:
            self.e_filters = [3, 32, 32, 64, 64, 128, 128, 256, 256]
            self.d_filters = [256, 128, 128, 64, 64, 32, 32, 3, 3]
            self.e_kernels = [3, 3, 3, 3, 3, 3, 3, 3]
            self.d_kernels = [3, 3, 3, 3, 3, 3, 3, 3, 1]
            self.e_bn = [False, True, False, True, False, True, False, True]
            self.d_bn = [False, True, True, True, True, True, True, False]
            self.e_strides = [1, 2, 1, 2, 1, 2, 1, 2]
            
        self.enc1 = EncoderLayer(in_filters=self.e_filters[0], out_filters=self.e_filters[1], kernel_size=self.e_kernels[0], stride=self.e_strides[0], bn=self.e_bn[0])
        self.enc2 = EncoderLayer(in_filters=self.e_filters[1], out_filters=self.e_filters[2], kernel_size=self.e_kernels[1], stride=self.e_strides[1], bn=self.e_bn[1])
        self.enc3 = EncoderLayer(in_filters=self.e_filters[2], out_filters=self.e_filters[3], kernel_size=self.e_kernels[2], stride=self.e_strides[2], bn=self.e_bn[2])
        self.enc4 = EncoderLayer(in_filters=self.e_filters[3], out_filters=self.e_filters[4], kernel_size=self.e_kernels[3], stride=self.e_strides[3], bn=self.e_bn[3])
        self.enc5 = EncoderLayer(in_filters=self.e_filters[4], out_filters=self.e_filters[5], kernel_size=self.e_kernels[4], stride=self.e_strides[4], bn=self.e_bn[4])
        self.enc6 = EncoderLayer(in_filters=self.e_filters[5], out_filters=self.e_filters[6], kernel_size=self.e_kernels[5], stride=self.e_strides[5], bn=self.e_bn[5])
        self.enc7 = EncoderLayer(in_filters=self.e_filters[6], out_filters=self.e_filters[7], kernel_size=self.e_kernels[6], stride=self.e_strides[6], bn=self.e_bn[6])
        self.enc8 = EncoderLayer(in_filters=self.e_filters[7], out_filters=self.e_filters[8], kernel_size=self.e_kernels[7], stride=self.e_strides[7], bn=self.e_bn[7])
        
        self.dec9 = DecoderLayer(in_filters=self.e_filters[8] + self.e_filters[7], out_filters=self.d_filters[0], kernel_size=self.d_kernels[0], bn=self.d_bn[0])
        self.dec10 = EncoderLayer(in_filters=self.d_filters[0], out_filters=self.d_filters[1], kernel_size=self.d_kernels[1], stride =1, bn=self.d_bn[1])
        self.dec11 = DecoderLayer(in_filters=self.d_filters[1] + self.e_filters[5], out_filters=self.d_filters[2], kernel_size=self.d_kernels[2], bn=self.d_bn[2])
        self.dec12 = EncoderLayer(in_filters=self.d_filters[2], out_filters=self.d_filters[3], kernel_size=self.d_kernels[3], stride=1, bn=self.d_bn[3])
        self.dec13 = DecoderLayer(in_filters=self.d_filters[3] + self.e_filters[3], out_filters=self.d_filters[4], kernel_size=self.d_kernels[4], bn=self.d_bn[4])
        self.dec14 = EncoderLayer(in_filters=self.d_filters[4], out_filters=self.d_filters[5], kernel_size=self.d_kernels[5], stride=1, bn=self.d_bn[5])
        self.dec15 = DecoderLayer(in_filters=self.d_filters[5] + self.e_filters[1], out_filters=self.d_filters[6], kernel_size=self.d_kernels[6], bn=self.d_bn[6])
        self.dec16 = EncoderLayer(in_filters=self.d_filters[6], out_filters=self.d_filters[7], kernel_size=self.d_kernels[7], stride=1, bn=self.d_bn[7])
        self.output = nn.Conv2d(self.d_filters[8], self.d_filters[8], kernel_size=self.d_kernels[8], padding="same")
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        self._print_shape("Input", x)
        x1, mask1 = self.enc1(x, mask)
        self._print_shape("After enc1", x1)
        x2, mask2 = self.enc2(x1, mask1)
        self._print_shape("After enc2", x2)
        x3, mask3 = self.enc3(x2, mask2)
        self._print_shape("After enc3", x3)
        x4, mask4 = self.enc4(x3, mask3)
        self._print_shape("After enc4", x4)
        x5, mask5 = self.enc5(x4, mask4)
        self._print_shape("After enc5", x5)
        x6, mask6 = self.enc6(x5, mask5)
        self._print_shape("After enc6", x6)
        x7, mask7 = self.enc7(x6, mask6)
        self._print_shape("After enc7", x7)
        x8, mask8 = self.enc8(x7, mask7)
        self._print_shape("After enc8", x8)
        
        x9, mask9 = self.dec9(x8, mask8, x7, mask7)
        self._print_shape("After dec9", x9)
        x10, mask10 = self.dec10(x9, mask9)
        self._print_shape("After dec10", x10)
        x11, mask11 = self.dec11(x10, mask10, x5, mask5)
        self._print_shape("After dec11", x11)
        x12, mask12 = self.dec12(x11, mask11)
        self._print_shape("After dec12", x12)
        x13, mask13 = self.dec13(x12, mask12, x3, mask3)
        self._print_shape("After dec13", x13)
        x14, mask14 = self.dec14(x13, mask13)
        self._print_shape("After dec14", x14)
        x15, mask15 = self.dec15(x14, mask14, x1, mask1)
        self._print_shape("After dec15", x15)
        x16, mask16 = self.dec16(x15, mask15)
        self._print_shape("After dec16", x16)
        
        x = self.output(x16)
        x = self.sigmoid(x)
        
        return x, mask16
    
class EncoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride=2, bn=True):
        super(EncoderLayer, self).__init__()
        
        # padding = (stride - 2) * in_filters // 4 + (kernel_size + 1 - stride) // 2
        padding = self.get_same_padding(kernel_size, stride=stride)
        
        self.pconv = PartialConv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        
        if bn:
            self.bn = nn.BatchNorm2d(out_filters)
        else:
            self.bn = None
        
        self.activation = nn.ReLU()
        
    def get_same_padding(self, kernel_size, stride=1, dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        pad_h = max((stride[0] - 1) + dilation[0] * (kernel_size[0] - 1), 0)
        pad_w = max((stride[1] - 1) + dilation[1] * (kernel_size[1] - 1), 0)

        # Return padding as a tuple: (padding_h, padding_w)
        return (pad_h // 2, pad_w // 2)

        
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x, mask
    
class DecoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, bn=True, leaky_relu=False):
        super(DecoderLayer, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = PartialConv2d(in_filters, out_filters, kernel_size=kernel_size, padding="same")
        
        if bn:
            self.bn = nn.BatchNorm2d(out_filters)
        else:
            self.bn = None
        
        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.ReLU()
        
    def forward(self, x, mask, e_x, e_mask):
        x = self.upsample(x)
        mask = self.upsample(mask)
        
        x = th.cat([x, e_x], dim=1)
        mask = th.cat([mask, e_mask], dim=1)
        
        x, mask = self.conv(x, mask)
        
        if self.bn is not None:
            x = self.bn(x)
        
        x = self.activation(x)
        
        return x, mask

class DummyModel(nn.Module):
    def __init__(self):
        """Model that substitutes the missing pixels with the mean of the valid pixels."""
        super(DummyModel, self).__init__()
    
    def forward(self, x: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """
        Args:
            x (th.Tensor): (B, C, H, W)
            mask (th.Tensor): (B, 1, H, W), where True means keep value, False means fill with channel-wise mean

        Returns:
            th.Tensor: tensor with masked-out values replaced by channel-wise mean of unmasked values
        """
        # Broadcast mask to shape of x
        mask_broadcast = mask.expand_as(x)  # shape (B, C, H, W)

        # Only consider valid (True) values for the mean
        masked_x = x.masked_fill(~mask_broadcast, 0.0)  # zero out masked-out (False) pixels
        num_valid = mask_broadcast.sum(dim=(0, 2, 3), keepdim=True).clamp(min=1)  # shape (1, C, 1, 1)
        channel_sum = masked_x.sum(dim=(0, 2, 3), keepdim=True)
        channel_mean = channel_sum / num_valid  # shape (1, C, 1, 1)

        # Fill masked-out (False) areas with mean
        out = th.where(mask_broadcast, x, channel_mean)

        return out, mask
class SimpleModel(nn.Module):
    def __init__(self):
        """Simple model using PartialConv2d for testing purposes."""
        super(SimpleModel, self).__init__()
        self.conv = PartialConv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x, mask):
        return self.conv(x, mask)