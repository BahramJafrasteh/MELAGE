from functools import partial

import numpy as np
import torch.nn as nn
import torch
import math
#from .model_utils import *
import math
import torch
from functools import partial
import torch.nn as nn
from einops import repeat, rearrange
#from einops import reduce, rearrange
#from einops.layers.torch import Rearrange
#from torch.optim import lr_scheduler
#import torch.nn.functional as F

class BlockLayer(nn.Module):
    def __init__(self, num_blcks, block_layer, planes_in, planes_out, kernel_size=3, first_layer=False,
                 input_size=None, time_emb_dim=None, norm_type='layer'):
        super(BlockLayer, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blcks):
            if i == 0:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=first_layer,
                                               input_size=input_size, time_emb_dim=time_emb_dim, norm_type=norm_type))
            else:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=False,
                                               input_size=input_size, time_emb_dim=time_emb_dim, norm_type=norm_type))
            planes_in = planes_out


    def forward(self, x, t=None):
        for i, block in enumerate(self.blocks):
            x = block(x, t)
        return x




class ResidualBlock(nn.Module):
    def __init__(self, planes_in, planes_out, time_emb_dim = None, kernel_size=3, first_layer=False, input_size=128, norm_type='layer'):
        super(ResidualBlock, self).__init__()
        if time_emb_dim is not None:
            if planes_in>planes_out:
                dim = planes_in*2
            else:
                dim = planes_in*2
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim)
            )

        self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                        kernel_size=kernel_size, dilation=1,
                                        activation=nn.ReLU, input_size=input_size, norm_type= norm_type)
        self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                        kernel_size=1,
                                        dilation=1, activation=nn.ReLU, input_size=input_size, norm_type=norm_type)
        if planes_in != planes_out:
            self.sample = nn.Conv3d(planes_in, planes_out, (1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1),
                                    bias=True)  #
        else:
            self.sample = None

    def forward(self, x, time_emb= None):
        identity = x.clone()
        scale_shift = None
        if time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)
        x = self.conv1(x, scale_shift= scale_shift)
        x = self.conv2(x, scale_shift=None)


        if self.sample is not None:
            identity = self.sample(identity)


        x += identity

        return x


class UnetEncoder(nn.Module):
    def __init__(self, in_channel, base_inc_channel=8, layer=BlockLayer, block=None,layer_blocks=None,
                 downsampling_stride=None,feature_dilation=1.5, layer_widths=None, kernel_size=3,
                 time_emb_dim=None, norm_type='layer'):
        super(UnetEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.downsampling_zarib = []
        in_channel_layer = in_channel
        input_size = 192
        self._layers_with = []
        #self._layers_with.append(base_inc_channel)
        for i, num_blcks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_channel_layer = layer_widths[i]
            else:
                out_channel_layer = base_inc_channel * int(feature_dilation ** (i+1))//2

            if i == 0:
                first_layer = True
            else:
                first_layer = False
            self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                     planes_in=in_channel_layer, planes_out=out_channel_layer,
                                     kernel_size=kernel_size,
                                     first_layer=first_layer, input_size=input_size,
                                     time_emb_dim=time_emb_dim, norm_type=norm_type))
            #self.attention_modules.append(Attention(out_channel_layer))
            if i != len(layer_blocks) - 1:

                #padding = kernel_size // 2  # constant size
                downsampling_conv = nn.Conv3d(out_channel_layer, out_channel_layer, (1, 1, 1), padding=1//2,
                              stride=(downsampling_stride,downsampling_stride,downsampling_stride),
                                 bias=True)
                #downsampling_conv = nn.MaxPool3d(kernel_size=2, stride=2)

                self.downsampling_convolutions.append(downsampling_conv)

                input_size = input_size // 2
            print("Encoder {}:".format(i), in_channel_layer, out_channel_layer)
            self._layers_with.append(out_channel_layer)
            in_channel_layer = out_channel_layer
        self.out_channel_layer = in_channel_layer//2
        self.last_downsampling_conv = nn.Conv3d(out_channel_layer, out_channel_layer, (1, 1, 1),
                                      padding=1 // 2 ,
                                      stride=(downsampling_stride, downsampling_stride, downsampling_stride),
                                      bias=True)
        self.output_size = input_size

    def forward(self, x, time=None):
        outputs = list()
        #outputs.insert(0, x)
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x, time)

            outputs.insert(0, x)

            x = downsampling(x)
        outputs.insert(0, x)
        x = self.layers[-1](x, time)
        x = self.last_downsampling_conv(x)
        outputs.insert(0, x) #bottle neck layer
        return outputs

class ConvolutionalBlock(nn.Module):
    def __init__(self, planes_in, planes_out, first_layer=False, kernel_size=3, dilation=1, activation=None,
                 input_size=None, norm_type='layer'):
        super(ConvolutionalBlock, self).__init__()
        if dilation == 1:
            padding = kernel_size // 2  # constant size
        else:
            # (In + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1
            if kernel_size == 3:
                if dilation == 2:
                    padding = 2
                elif dilation == 4:
                    padding = 4
                elif dilation == 3:
                    padding = 3
                else:
                    padding = None
            elif kernel_size == 1:
                padding = 0
        self.activation = None
        self.norm = None
        if first_layer:
            self.norm = nn.InstanceNorm3d(planes_in)
            self.activation = activation()
            self.conv = nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                  padding=padding, bias=True,
                                                  dilation=(dilation, dilation, dilation))
        else:
            if activation is not None:
                if norm_type.lower()=='layer':
                    self.norm = nn.LayerNorm([input_size, input_size, input_size])
                elif norm_type.lower()=='group':
                    valid_num_groups = np.array([16, 8, 4, 2])
                    valid_num_groups = valid_num_groups[valid_num_groups<planes_in]
                    num_groups = None
                    for num_groups in valid_num_groups:
                        if planes_in % num_groups != 0:
                            break
                    if num_groups is None:
                        raise exit('Num groups can not be determined')
                    self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=planes_in)
                elif norm_type.lower()=='batch':
                    self.norm = nn.BatchNorm3d(planes_in)
                elif norm_type.lower() == 'instance':
                    self.norm = nn.InstanceNorm3d(planes_in)
                else:
                    self.norm= None

                self.activation = activation()
                self.conv = nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                      padding=padding, bias=True,
                                                      dilation=(dilation, dilation, dilation))

            else:
                if norm_type.lower()=='layer':
                    if input_size<120:
                        self.norm = nn.LayerNorm([input_size, input_size, input_size])
                    else:
                        self.norm = nn.InstanceNorm3d(planes_in)
                elif norm_type.lower()=='group':
                    valid_num_groups = [16, 8, 4, 2]
                    valid_num_groups = valid_num_groups[valid_num_groups < planes_in]
                    num_groups = None
                    for num_groups in valid_num_groups:
                        if planes_in % num_groups != 0:
                            break
                    if num_groups is None:
                        raise exit('Num groups can not be determined')
                    self.norm = nn.GroupNorm(num_groups=planes_in, num_channels=planes_in)
                elif norm_type.lower() == 'batch':
                    self.norm = nn.BatchNorm3d(planes_in)
                elif norm_type.lower() == 'instance':
                    self.norm = nn.InstanceNorm3d(planes_in)
                else:
                    self.norm = None

                self.conv = nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                      padding=padding, bias=True,
                                                      dilation=(dilation, dilation, dilation))


    def forward(self, x, scale_shift=None):
        if self.norm is not None:
            x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        if self.activation is not None:
            x = self.activation(x)

        x = self.conv(x)

        return x
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[...,None]*emb[None,:]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if len(emb.shape)==3:
            emb = emb.view(emb.shape[0], emb.shape[1] * emb.shape[2])
        return emb



class UnetDecoder(nn.Module):
    def __init__(self, in_channel, base_inc_channel=64, layer=BlockLayer, block=None,layer_blocks=[1,1,1,1],
                 feature_dilation=2, upsampling_stride=2, layer_widths=None, kernel_size=3,
                 upsampling_mode="trilinear", align_corners=False, use_transposed_convolutions=False, last_cov_channels=256,
                 time_emb_dim=None, norm_type='layer'
                 ):
        super(UnetDecoder, self).__init__()
        self.layers = nn.ModuleList()

        self.upsampling_blocks = nn.ModuleList()

        self.attention_modules = nn.ModuleList()
        in_channel_layer = in_channel
        #input_size = 24
        input_size = 16


        for i, num_blcks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_channel_layer = layer_widths[i]
            else:
                out_channel_layer = base_inc_channel // (feature_dilation ** (i))

            if i == 0:
                first_layer = True
                self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                         planes_in=last_cov_channels, planes_out=out_channel_layer,
                                         kernel_size=kernel_size,
                                         first_layer=first_layer, input_size=input_size, time_emb_dim=time_emb_dim, norm_type=norm_type))
            else:
                first_layer = False

                self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                         planes_in=in_channel_layer+layer_widths[i-1], planes_out=out_channel_layer,
                                         kernel_size=kernel_size,
                                         first_layer=first_layer, input_size=input_size, time_emb_dim=time_emb_dim, norm_type=norm_type))

            #self.upsampling_blocks.append(nn.ConvTranspose3d(out_channel_layer, out_channel_layer, kernel_size=2,
            #                                                         stride=upsampling_stride, padding=0))
            self.upsampling_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))

            input_size = input_size *2
            last_cov_channels = in_channel_layer#last_cov_channels//2
            print("Decoder {}:".format(i), in_channel_layer, out_channel_layer)
            in_channel_layer = out_channel_layer
        self.out_channel_layer = in_channel_layer
    def forward(self, x, t):
        i = 0
        outputs = list()
        y = x[0]
        for up, lay in zip(self.upsampling_blocks, self.layers[:-1]):
            if i == 0:
                y = lay(y, t)
            else:
                y = lay(y,t)
            outputs.insert(0, y)
            y = torch.cat([y, x[i ]], 1)
            y = up(y)
            #y = att(y)
            #y = torch.cat([y, x[i + 1]],1)
            i += 1
        outputs.insert(0, y)
        y = self.layers[-1](y,t)
        y = up(y)
        outputs.insert(0, y)
        return y, outputs

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim//2, 1)

    def forward(self, x, mask=None):
        b, c, h, w, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        v = v / (h * w* z)
        atv = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        out = rearrange(atv, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=z)
        return self.to_out(out)




class CrossConv3d(nn.Conv3d):

    """
    https://github.com/JJGO/UniverSeg/blob/main/universeg/nn/cross_conv.py
    Compute pairwise convolution between all element of x and all elements of y.
    x, y are tensors of size B,_,C,H,W where _ could be different number of elements in x and y
    essentially, we do a meshgrid of the elements to get B,Sx,Sy,C,H,W tensors, and then
    pairwise conv.
    Args:
        x (tensor): B,Sx,Cx,H,W
        y (tensor): B,Sy,Cy,H,W
    Returns:
        tensor: B,Sx,Sy,Cout,H,W
    """
    """
    CrossConv2d is a convolutional layer that performs pairwise convolutions between elements of two input tensors.

    Parameters
    ----------
    in_channels : int or tuple of ints
        Number of channels in the input tensor(s).
        If the tensors have different number of channels, in_channels must be a tuple
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of ints
        Size of the convolutional kernel.
    stride : int or tuple of ints, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple of ints, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple of ints, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Padding mode. Default is "zeros".
    device : str, optional
        Device on which to allocate the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type assigned to the tensor. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor resulting from the pairwise convolution between the elements of x and y.

    Notes
    -----
    x and y are tensors of size (B, Sx, Cx, H, W) and (B, Sy, Cy, H, W), respectively,
    The function does the cartesian product of the elements of x and y to obtain a tensor
    of size (B, Sx, Sy, Cx + Cy, H, W), and then performs the same convolution for all 
    (B, Sx, Sy) in the batch dimension. Runtime and memory are O(Sx * Sy).

    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 32, 32)
    >>> y = torch.randn(2, 5, 6, 32, 32)
    >>> conv = CrossConv2d(in_channels=(4, 6), out_channels=7, kernel_size=3, padding=1)
    >>> output = conv(x, y)
    >>> output.shape  #(2, 3, 5, 7, 32, 32)
    """


    def __init__(
        self,
        in_channels,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation= 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise convolution between all elements of x and all elements of y.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of size (B, Sx, Cx, H, W).
        y : torch.Tensor
            Input tensor of size (B, Sy, Cy, H, W).

        Returns
        -------
        torch.Tensor
            Tensor resulting from the cross-convolution between the elements of x and y.
            Has size (B, Sx, Sy, Co, H, W), where Co is the number of output channels.
        """
        B, Sx, *_ = x.shape
        _, Sy, *_ = y.shape

        xs = repeat(x, "B Sx Cx H W Y -> B Sx Sy Cx H W Y", Sy=Sy)
        ys = repeat(y, "B Sy Cy H W Y-> B Sx Sy Cy H W Y", Sx=Sx)

        xy = torch.cat([xs, ys], dim=3,)

        batched_xy = rearrange(xy, "B Sx Sy C2 H W Y -> (B Sx Sy) C2 H W Y")
        batched_output = super().forward(batched_xy)

        output = rearrange(
            batched_output, "(B Sx Sy) Co H W Y-> B Sx Sy Co H W Y", B=B, Sx=Sx, Sy=Sy
        )
        return output

class UnetGen(nn.Module):
    def __init__(self, base_inc_channel=8,
                 feature_dilation=2, downsampling_stride=2,
                 encoder_class=UnetEncoder, layer_widths=None, block=None,
                 kernel_size=3, interpolation_mode ="trilinear",decoder_class=None,
                 use_transposed_convolutions=True, time_embed = False, norm_type='layer'):
        super(UnetGen, self).__init__()
        time_embed = self.time_embed
        use_transposed_convolutions = self.use_tr_conv
        inblock = 16
        base_inc_channel = inblock
        self.base_inc_channel = base_inc_channel

        sinu_pos_emb = SinusoidalPosEmb(inblock)
        fourier_dim = inblock
        #if self.spacing_embed:
        #    fourier_dim*=4

        # time embeddings

        time_dim = inblock * 4
        if time_embed:
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None

        #encoder_blocks = [1, 1, 1, 1, 1, 1]

        #decoder_blocks = [1,1,1,1, 1, 1]
        encoder_blocks = [1, 1, 1]

        decoder_blocks = [1, 1, 1]

        padding = kernel_size // 2  # constant size
        self.before_encoder = nn.Conv3d(1, inblock, kernel_size=(3, 3, 3),
                                        stride=(1, 1, 1), padding=3//2,
                                        bias=True)


        self.encoder = encoder_class(in_channel=inblock, base_inc_channel=base_inc_channel, layer_blocks=encoder_blocks,
                                     block=block,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size,
                                     time_emb_dim=time_dim, norm_type=norm_type)

        layer_widths = self.encoder._layers_with
        in_channel = layer_widths[-1]
        self.BottleNeck = BlockLayer(num_blcks=1, block_layer=block,
                                         planes_in=in_channel, planes_out=in_channel,
                                         kernel_size=kernel_size,
                                         first_layer=False, input_size=self.encoder.output_size, time_emb_dim=time_dim, norm_type=norm_type)

        self.BottleNeck_att = Attention(in_channel)

        layer_widths = layer_widths[::-1]#[1:]
        layer_widths[0]= layer_widths[0]//2

        in_channel = in_channel//2
        self.decoder = decoder_class(in_channel=in_channel, base_inc_channel=base_inc_channel*8, layer_blocks=decoder_blocks,
                                     block=block, last_cov_channels = self.encoder.out_channel_layer,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size, time_emb_dim=time_dim, norm_type=norm_type,
                                     )
        self.decoder_mask = decoder_class(in_channel=in_channel, base_inc_channel=base_inc_channel*8, layer_blocks=decoder_blocks,
                                     block=block, last_cov_channels = self.encoder.out_channel_layer,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size, time_emb_dim=time_dim, norm_type=norm_type,
                                     )

        kernel_size = 3

        self.last_convolution = BlockLayer(num_blcks=1, block_layer=block,
                                         planes_in=inblock*2, planes_out=inblock//2,
                                         kernel_size=kernel_size,
                                         first_layer=False, input_size=192, time_emb_dim=time_dim, norm_type=norm_type)

        self.last_convolution_rec = BlockLayer(num_blcks=1, block_layer=block,
                                         planes_in=inblock*2, planes_out=inblock//2,
                                         kernel_size=kernel_size,
                                         first_layer=False, input_size=192, time_emb_dim=time_dim, norm_type=norm_type)

        self.final_convolution = nn.Conv3d(inblock//2, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                                           stride=(1, 1, 1), bias=True, padding=kernel_size // 2)
        self.final_convolution_rec = nn.Conv3d(inblock//2, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                                           stride=(1, 1, 1), bias=True, padding=kernel_size // 2)
        self.activation = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()





    def forward(self, y, time, t=0, noise = None):


        y = self.before_encoder(y)

        if self.time_embed:
            if len(time.shape)==1:
                t = self.time_mlp(time)
            else:
                t = self.time_mlp(time)
        else:
            t = None

        x = self.encoder(y, t)
        x[0] = self.BottleNeck(x[0], t)
        x[0] = self.BottleNeck_att(x[0])

        mask,_ = self.decoder_mask(x,t)
        x, _ = self.decoder(x, t)

        ###############################Attention########################################
        dim_head = 16
        self.heads =4
        self.scale = dim_head ** -0.5

        b, c, h, w, z = x.shape
        #qkv = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(x, 'b (h c) x y z -> b h c (x y z)', h = self.heads)
        k = rearrange(mask, 'b (h c) x y z -> b h c (x y z)', h = self.heads)
        #q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale
        attention = torch.softmax(scaled_dot_prod, dim=-1)

        k = k / (h * w* z)
        atv = torch.einsum('... i j , ... j d -> ... i d', attention, k)
        x = rearrange(atv, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=z)
        ###############################Attention########################################
        mask = torch.cat([mask, y], 1)
        mask = self.last_convolution(mask)
        mask = self.final_convolution(mask)


        x = torch.cat([x, y], 1)
        #x = (x * mask)
        z = self.last_convolution_rec(x)
        z = self.final_convolution_rec(z)


        return [mask,z]

class MGA_NET(UnetGen):
    def __init__(self,time_embed=False, channels=1, *args, encoder_class=UnetEncoder, **kwargs):
        self.time_embed = time_embed
        self.use_tr_conv = False

        norm_type = "instance"
        super().__init__(*args, encoder_class=encoder_class, decoder_class=UnetDecoder,
                         block=ResidualBlock, norm_type=norm_type, **kwargs)

        self.channels = channels
        self.netName = 'MGA_NET'
    def name(self):
        return 'MGA_NET'
