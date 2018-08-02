
import torch
from torch import nn
from torch import optim
import torch.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """One block of convolutions with a residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        assert kernel_size%2==1, "kernel_size should be odd number"
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = kernel_size//2)
        
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        
        self.bn2_1 = nn.BatchNorm2d(out_channels)
        self.bn2_2 = nn.BatchNorm2d(out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.out_channels != self.in_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.res_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = self.bn1_1(x)
        
        out = self.conv1(x)
        out = self.bn1_2(out)
        out = nn.ReLU()(out)
        
        out = self.bn2_1(out)
        out = self.conv2(out)
        out = self.bn2_2(out)
        
        if self.out_channels != self.in_channels:
            residual = self.res_conv(residual)
            residual = self.res_bn(residual)
        
        out = nn.ReLU()(out+residual)
            
        return out

class DeconvBlock(nn.Module):
    """One block of transposed convolutions with a residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        assert kernel_size%2==1, "kernel_size should be odd number"
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding = kernel_size//2)
        
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        
        self.bn2_1 = nn.BatchNorm2d(out_channels)
        self.bn2_2 = nn.BatchNorm2d(out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.out_channels != self.in_channels:
            self.res_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)
            self.res_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = self.bn1_1(x)
        
        out = self.deconv1(x)
        out = self.bn1_2(out)
        out = nn.ReLU()(out)
        
        out = self.bn2_1(out)
        out = self.deconv2(out)
        out = self.bn2_2(out)
        
        if self.out_channels != self.in_channels:
            residual = self.res_conv(residual)
            residual = self.res_bn(residual)
        
        out = nn.ReLU()(out+residual)
           
        return out

class BlockSet(nn.Module):
    """Cascades a set of given blocks. The first block maps in_channels to 
    out_channels, and remaining blocks map out_channels to out_channels."""
    def __init__(self, block, in_channels, out_channels, block_count, kernel_size=3):
        super().__init__()
        
        block1 = block(in_channels, out_channels, kernel_size=kernel_size)
        blocks = [block(out_channels, out_channels, kernel_size=kernel_size) for _ in range(block_count-1)]
        
        self.blocks = nn.Sequential(block1, *blocks)
            
    def forward(self, input):
        out = self.blocks(input)
        
        return out

class BlockNet(nn.Module):
    """
    Cascades multiple BlockSets to form a complete network. One BlockSet is used for each element in
    the channel_sequence, and size scaling is done between blocks based on size_sequence. A decrease in
    size is done using fractional max pooling, and an increase in size is done by bilinear upsampling.
    
    A final 1x1 block is added after all the BlockSets.
    """
    def __init__(self, block, channel_sequence, size_sequence, block_count, kernel_size=3, use_block_for_last=False):
        super().__init__()
        
        assert len(channel_sequence)==len(size_sequence), "channel and size sequences should have same length"
        
        old_channels, old_size = channel_sequence[0], size_sequence[0]
        
        layers = []
        for channels, size in zip(channel_sequence[1:], size_sequence[1:]):
            layers.append(BlockSet(block, 
                                   in_channels=old_channels,
                                   out_channels=channels,
                                   block_count=block_count,
                                   kernel_size=kernel_size))
            if size<old_size:
                layers.append(nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=size))
            elif size>old_size:
                layers.append(nn.Upsample(size=(size,size), mode='bilinear', align_corners=True))
            
            old_channels, old_size = channels, size
        
        if use_block_for_last:
            layers.append(block(channels, channels, kernel_size=1))
        else:
            layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        
        self.layers = nn.Sequential(*layers)
                
            
            
    def forward(self, input):
        out = self.layers(input)
        return out

class Autoencoder(nn.Module):
    """Combines an encoder and decoder into an autoencoder
    Can supply `return_embedding` as true to get back the embedding as well"""
    def __init__(self, encoder, decoder, output_activation=nn.LeakyReLU()):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.out_act = output_activation
        
    def forward(self, input, return_embedding=False):
        embedding = self.encoder(input)
        output = self.out_act(self.decoder(embedding))
        
        if return_embedding:
            return output, embedding
        
        return output

class NormalizeModule(nn.Module):
    """Returns (input-mean)/std"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, input):
        return (input-self.mean)/self.std

class CoordConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, input):
        h = input.shape[2]
        w = input.shape[3]
        
#         print(input.shape)
        
        i = np.linspace(-1,1,h)
        j = np.linspace(-1,1,w)
        
        ii,jj = np.meshgrid(i,j)
        
       
        ii = torch.tensor(ii, dtype=input.dtype).to(input.device).repeat((input.shape[0],1,1,1))
        jj = torch.tensor(jj, dtype=input.dtype).to(input.device).repeat((input.shape[0],1,1,1))
        
#         print(ii.device,jj)
        
        inp = torch.cat([input, ii, jj], dim=1)
        
        out = self.conv(inp)
        
        return out

class CoordConvBlock(nn.Module):
    """One block of convolutions with a residual connection. Convolutions are 'CoordConvs'.
    They add coordinate meshgrids to input channels."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        assert kernel_size%2==1, "kernel_size should be odd number"
        
        self.cconv1 = CoordConv(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.cconv2 = CoordConv(out_channels, out_channels, kernel_size, padding = kernel_size//2)
        
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        
        self.bn2_1 = nn.BatchNorm2d(out_channels)
        self.bn2_2 = nn.BatchNorm2d(out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.out_channels != self.in_channels:
            self.res_cconv = CoordConv(in_channels, out_channels, kernel_size=1)
            self.res_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = self.bn1_1(x)
        
        out = self.cconv1(x)
        out = self.bn1_2(out)
        out = nn.ReLU()(out)
        
        out = self.bn2_1(out)
        out = self.cconv2(out)
        out = self.bn2_2(out)
        
        if self.out_channels != self.in_channels:
            residual = self.res_cconv(residual)
            residual = self.res_bn(residual)
        
        out = nn.ReLU()(out+residual)
            
        return out