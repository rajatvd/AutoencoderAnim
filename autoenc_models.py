
import torch
from torch import nn
from torch import optim
import torch.functional as F
import numpy as np
from modules import *

def conv_conv128(DEVICE=torch.device('cpu')):
    """Conv-Conv architecture with latent dim of 128. Approx
    3.3M params each in encoder and decoder. Leaky ReLU activation.
    
    Returns encoder, decoder and autoencoder.
    Returned decoder includes output activation.
    """
    encoder = BlockNet(ConvBlock, 
                   channel_sequence=[1,64,64,128,128], 
                   size_sequence=[64,32,16,8,1], 
                   block_count=5,
                   kernel_size=3,
                   use_block_for_last=True).to(DEVICE)

    encp = nn.utils.parameters_to_vector(encoder.parameters()).shape[0]
    print(f"Encoder has {encp} params")
    
    decoder = BlockNet(ConvBlock, 
                   channel_sequence=[128,128,128,64,1], 
                   size_sequence=[1,8,16,32,64], 
                   block_count=5,
                   kernel_size=3,
                   use_block_for_last=True).to(DEVICE)

    decp = nn.utils.parameters_to_vector(decoder.parameters()).shape[0]
    print(f"Decoder has {decp} params")
    
    autoencoder = nn.Sequential(encoder, decoder, nn.LeakyReLU()).to(DEVICE)
    
    return encoder, nn.Sequential(decoder, nn.LeakyReLU()), autoencoder

def coordconv_conv128(DEVICE=torch.device('cpu')):
    """CoordConv-CoordConv architecture with latent dim of 128. Approx
    3.3M params each in encoder and decoder. Leaky ReLU activation.
    
    Input is normalized to [-1,1] as (input-0.5)/0.5
    
    Returns encoder, decoder and autoencoder.
    Returned decoder includes output activation.
    """
    encoder = nn.Sequential(
        NormalizeModule(0.5,0.5),
        BlockNet(CoordConvBlock, 
                       channel_sequence=[1,64,64,128,128], 
                       size_sequence=[64,32,16,8,1], 
                       block_count=5,
                       kernel_size=3)
    ).to(DEVICE)

    encp = nn.utils.parameters_to_vector(encoder.parameters()).shape[0]
    print(f"Encoder has {encp} params")
    
    decoder = BlockNet(CoordConvBlock, 
                   channel_sequence=[128,128,128,64,1], 
                   size_sequence=[1,8,16,32,64], 
                   block_count=5,
                   kernel_size=3).to(DEVICE)

    decp = nn.utils.parameters_to_vector(decoder.parameters()).shape[0]
    print(f"Decoder has {decp} params")
    
    autoencoder = Autoencoder(encoder, decoder, nn.LeakyReLU()).to(DEVICE)
    
    return encoder, nn.Sequential(decoder, nn.LeakyReLU()), autoencoder

def coordconv_conv64(DEVICE=torch.device('cpu')):
    """CoordConv-CoordConv architecture with latent dim of 64. Approx
    1M params each in encoder and decoder. Leaky ReLU activation.
    
    Input is normalized to [-1,1] as (input-0.5)/0.5
    
    Returns encoder, decoder and autoencoder.
    Returned decoder includes output activation.
    """
    encoder = nn.Sequential(
        NormalizeModule(0.5,0.5),
        BlockNet(CoordConvBlock, 
                       channel_sequence=[1,32,32,64,64], 
                       size_sequence=[64,32,16,8,1], 
                       block_count=5,
                       kernel_size=3)
    ).to(DEVICE)

    encp = nn.utils.parameters_to_vector(encoder.parameters()).shape[0]
    print(f"Encoder has {encp} params")
    
    decoder = BlockNet(CoordConvBlock, 
                   channel_sequence=[64,64,64,32,1], 
                   size_sequence=[1,8,16,32,64], 
                   block_count=5,
                   kernel_size=3).to(DEVICE)
    
    decp = nn.utils.parameters_to_vector(decoder.parameters()).shape[0]
    print(f"Decoder has {decp} params")
    
    autoencoder = Autoencoder(encoder, decoder, nn.LeakyReLU()).to(DEVICE)
    
    return encoder, nn.Sequential(decoder, nn.LeakyReLU()), autoencoder

def coordconv_conv256(DEVICE=torch.device('cpu')):
    """CoordConv-CoordConv architecture with latent dim of 256. Approx
    8M params each in encoder and decoder. Leaky ReLU activation.
    
    Input is normalized to [-1,1] as (input-0.5)/0.5
    
    Returns encoder, decoder and autoencoder.
    Returned decoder includes output activation.
    """
    encoder = nn.Sequential(
        NormalizeModule(0.5,0.5),
        BlockNet(CoordConvBlock, 
                       channel_sequence=[1,64,64,128,256], 
                       size_sequence=[64,32,16,8,1], 
                       block_count=5,
                       kernel_size=3)
    ).to(DEVICE)

    encp = nn.utils.parameters_to_vector(encoder.parameters()).shape[0]
    print(f"Encoder has {encp} params")
    
    decoder = BlockNet(CoordConvBlock, 
                   channel_sequence=[256,256,128,64,1], 
                   size_sequence=[1,8,16,32,64], 
                   block_count=5,
                   kernel_size=3).to(DEVICE)

    decp = nn.utils.parameters_to_vector(decoder.parameters()).shape[0]
    print(f"Decoder has {decp} params")
    
    autoencoder = Autoencoder(encoder, decoder, nn.LeakyReLU()).to(DEVICE)
    
    return encoder, nn.Sequential(decoder, nn.LeakyReLU()), autoencoder