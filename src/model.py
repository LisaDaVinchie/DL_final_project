import torch as th
import torch.nn as nn
from PartialConv import PartialConv2d

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.e_filters = [3, 64, 128, 256, 512, 512, 512, 512, 512]
        self.e_kernels = [7, 5, 5, 3, 3, 3, 3, 3]
        self.d_filters = [512, 512, 512, 512, 256, 128, 64, 3, 3]
        self.d_kernels = [3, 3, 3, 3, 3, 3, 3, 3, 1]
        
        self.layer_setup()
        self.print_shapes = True  # Set to True to print shapes during forward pass

    def _print_shape(self, name, tensor):
        if self.print_shapes:
            print(f"{name} shape: {tensor.shape}")
    
    def layer_setup(self):
        self.enc1 = EncoderLayer(in_filters=self.e_filters[0], out_filters=self.e_filters[1], kernel_size=self.e_kernels[0], bn=False)
        self.enc2 = EncoderLayer(in_filters=self.e_filters[1], out_filters=self.e_filters[2], kernel_size=self.e_kernels[1])
        self.enc3 = EncoderLayer(in_filters=self.e_filters[2], out_filters=self.e_filters[3], kernel_size=self.e_kernels[2])
        self.enc4 = EncoderLayer(in_filters=self.e_filters[3], out_filters=self.e_filters[4], kernel_size=self.e_kernels[3])
        self.enc5 = EncoderLayer(in_filters=self.e_filters[4], out_filters=self.e_filters[5], kernel_size=self.e_kernels[4])
        self.enc6 = EncoderLayer(in_filters=self.e_filters[5], out_filters=self.e_filters[6], kernel_size=self.e_kernels[5])
        self.enc7 = EncoderLayer(in_filters=self.e_filters[6], out_filters=self.e_filters[7], kernel_size=self.e_kernels[6])
        self.enc8 = EncoderLayer(in_filters=self.e_filters[7], out_filters=self.e_filters[8], kernel_size=self.e_kernels[7])
        
        self.dec9 = DecoderLayer(in_filters=self.e_filters[8] + self.e_filters[7], out_filters=self.d_filters[0], kernel_size=self.d_kernels[0])
        self.dec10 = DecoderLayer(in_filters=self.d_filters[0] + self.e_filters[6], out_filters=self.d_filters[1], kernel_size=self.d_kernels[1])
        self.dec11 = DecoderLayer(in_filters=self.d_filters[1] + self.e_filters[5], out_filters=self.d_filters[2], kernel_size=self.d_kernels[2])
        self.dec12 = DecoderLayer(in_filters=self.d_filters[2] + self.e_filters[4], out_filters=self.d_filters[3], kernel_size=self.d_kernels[3])
        self.dec13 = DecoderLayer(in_filters=self.d_filters[3] + self.e_filters[3], out_filters=self.d_filters[4], kernel_size=self.d_kernels[4])
        self.dec14 = DecoderLayer(in_filters=self.d_filters[4] + self.e_filters[2], out_filters=self.d_filters[5], kernel_size=self.d_kernels[5])
        self.dec15 = DecoderLayer(in_filters=self.d_filters[5] + self.e_filters[1], out_filters=self.d_filters[6], kernel_size=self.d_kernels[6])
        self.dec16 = DecoderLayer(in_filters=self.d_filters[6] + self.e_filters[0], out_filters=self.d_filters[7], kernel_size=self.d_kernels[7], bn=False)
        self.output = nn.Conv2d(self.d_filters[8], self.d_filters[8], kernel_size=self.d_kernels[8], padding=0)
        
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
    

class EncoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride=2, bn=True):
        super(EncoderLayer, self).__init__()
        
        padding = (stride - 2) * in_filters // 4 + (kernel_size + 1 - stride) // 2
        
        self.pconv = PartialConv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        
        if bn:
            self.bn = nn.BatchNorm2d(out_filters)
        else:
            self.bn = None
        
        self.activation = nn.ReLU()
        
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x, mask
    
class DecoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, bn=True):
        super(DecoderLayer, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = PartialConv2d(in_filters, out_filters, kernel_size=kernel_size, padding="same")
        
        if bn:
            self.bn = nn.BatchNorm2d(out_filters)
        else:
            self.bn = None
            
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
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
        