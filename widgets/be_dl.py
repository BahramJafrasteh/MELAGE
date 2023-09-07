__AUTHOR__ = 'Bahram Jafrasteh'

from functools import partial
import torch.nn as nn
import torch
# import pytorch_lightning as pl
import numpy as np


class BlockLayer(nn.Module):
    def __init__(self, num_blcks, block_layer, planes_in, planes_out, dropout=None, kernel_size=3, first_layer=False,
                 input_size=None):
        super(BlockLayer, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blcks):
            if i == 0:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=first_layer,
                                               input_size=input_size))
            else:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=False,
                                               input_size=input_size))
            planes_in = planes_out
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x


class NormalBlock(nn.Module):
    def __init__(self, planes_in, planes_out, stride=1, kernel_size=3, first_layer=False):
        super(NormalBlock, self).__init__()

        # dilated = False
        if dilated:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=2,
                                            activation=nn.ReLU)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=3, dilation=2,
                                            activation=nn.ReLU)
            if not first_layer:
                self.conv3 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                                kernel_size=1, dilation=2,
                                                activation=nn.ReLU)
        else:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=1,
                                            activation=nn.ReLU)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=1,
                                            dilation=1, activation=nn.ReLU)
            if not first_layer:
                self.conv3 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                                kernel_size=1, dilation=1,
                                                activation=nn.ReLU)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        if hasattr(self, 'conv3'):
            x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, planes_in, planes_out, stride=1, kernel_size=3, first_layer=False, input_size=128):
        super(ResidualBlock, self).__init__()

        # dilated = False
        if dilated:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=2,
                                            activation=nn.ReLU, input_size=input_size)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=1, dilation=1,
                                            activation=nn.ReLU, input_size=input_size)
        else:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=1,
                                            activation=nn.ReLU, input_size=input_size)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=1,
                                            dilation=1, activation=nn.ReLU, input_size=input_size)
        if planes_in != planes_out:
            if dilated:
                self.sample = nn.Conv3d(planes_in, planes_out, (1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1),
                                        bias=False)  #
            else:
                self.sample = nn.Conv3d(planes_in, planes_out, (1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1),
                                        bias=False)  #


        else:
            self.sample = None
        # self.tanh = nn.Tanh()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        if self.sample is not None:
            identity = self.sample(identity)


        x += identity

        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channel, base_inc_channel=16, layer_blocks=None, layer=BlockLayer, block=None,
                 feature_dilation=2, downsampling_stride=2, dropout=None, layer_widths=None, kernel_size=3):
        super(EncoderLayer, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.downsampling_zarib = []
        in_channel_layer = in_channel
        input_size = 192#192
        for i, num_blcks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_channel_layer = layer_widths[i]
            else:
                out_channel_layer = base_inc_channel * (feature_dilation ** (i))
            if dropout and i == 0:
                layer_dropout = dropout
                # in_channel_layer = out_channel_layer
            else:
                layer_dropout = None
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                     planes_in=in_channel_layer, planes_out=out_channel_layer,
                                     dropout=layer_dropout, kernel_size=kernel_size,
                                     first_layer=first_layer, input_size=input_size))
            if i != len(layer_blocks) - 1:

                padding = kernel_size // 2  # constant size
                maxpool3d_layer = nn.Conv3d(out_channel_layer, out_channel_layer, (kernel_size, kernel_size, kernel_size), padding=padding,
                              stride=(2,2,2),
                                 bias=True)
                #maxpool3d_layer = nn.MaxPool3d(kernel_size=(4, 4, 4),
                #                               stride=(4, 4, 4), padding=0)
                self.downsampling_convolutions.append(maxpool3d_layer)

                input_size = input_size // 2
            #print("Encoder {}:".format(i), in_channel_layer, out_channel_layer)
            in_channel_layer = out_channel_layer
        self.out_channel_layer = in_channel_layer


class ConvolutionalBlock(nn.Module):
    def __init__(self, planes_in, planes_out, first_layer=False, kernel_size=3, dilation=1, activation=None,
                 input_size=None):
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
        if first_layer:
            self.conv = nn.Sequential(*[activation(),
                                        nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                  padding=padding, bias=False,
                                                  dilation=(dilation, dilation, dilation)),

                                        ])
        else:
            if activation is not None:
                self.conv = nn.Sequential(*[nn.LayerNorm([input_size, input_size, input_size]), activation(),
                                            nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                      padding=padding, bias=False,
                                                      dilation=(dilation, dilation, dilation)),

                                            ])
            else:
                self.conv = nn.Sequential(*[nn.LayerNorm([input_size, input_size, input_size]),
                                            nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                      padding=padding, bias=False,
                                                      dilation=(dilation, dilation, dilation)),

                                            ])

        # self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.conv(x)

        return x


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None, in_channel=1, base_inc_channel=16, encoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2,decoder_blocks=None,
                 encoder_class=EncoderLayer, layer_widths=None, block=None,
                 kernel_size=3, interpolation_mode ="trilinear",decoder_class=None,
                 dropout=0.5, use_transposed_convolutions=False):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.base_inc_channel = base_inc_channel

        if encoder_blocks is None:
            encoder_blocks = [1, 1, 1, 1, 1]
        if decoder_class is None:
            decoder_blocks = [1,1,1,1, 1]
        if block_used == 'normal':
            inblock = 16
        else:
            inblock = 16
        if dilated:
            # (In + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1
            self.before_encoder = nn.Conv3d(1, inblock, kernel_size=(7, 7, 7),
                                            stride=(1, 1, 1), padding=7 // 2,
                                            bias=True, dilation=1)
        else:
            self.before_encoder = nn.Conv3d(1, inblock, kernel_size=(7, 7, 7),
                                            stride=(1, 1, 1), padding=7 // 2,
                                            bias=True, dilation=1)
        self.encoder = encoder_class(in_channel=inblock, base_inc_channel=base_inc_channel, layer_blocks=encoder_blocks,
                                     block=block,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size, dropout=dropout)

        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, True,
                                                                decoder_blocks)
        base_width = 8
        self.decoder = decoder_class(in_channel=inblock*4, base_inc_channel=base_inc_channel*4, layer_blocks=decoder_blocks,
                                     block=block, last_cov_channels = self.encoder.out_channel_layer,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)

        kernel_size = 3
        self.final_convolution = nn.Conv3d(4, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                                           stride=(1, 1, 1), bias=False, padding=kernel_size // 2)



    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            #decoder_blocks = encoder_blocks
            decoder_class = DecoderLayer

        return decoder_class, decoder_blocks

    def forward(self, y, time=None):
        x = self.before_encoder(y)
        x = self.encoder(x)
        x=self.decoder(x)
        x = self.final_convolution(x)
        return x


class UNetEncoder(EncoderLayer):
    def forward(self, x, class_ind=None):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return outputs


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, padding=None, dilation=1, kernel_size=3):
    """3x3x3 convolution with padding"""
    if padding is None:
        padding = kernel_size // 2  # padding to keep the image size constant
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class DecoderLayer(nn.Module):
    def __init__(self, in_channel, base_inc_channel=64, layer_blocks=None, layer=BlockLayer, block=None,
                 feature_dilation=2, upsampling_stride=2, dropout=None, layer_widths=None, kernel_size=3,
                 upsampling_mode="trilinear", align_corners=False, use_transposed_convolutions=False, last_cov_channels=256
                 ):
        super(DecoderLayer, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        in_channel_layer = in_channel
        input_size = 12#12
        for i, num_blcks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_channel_layer = layer_widths[i]
            else:
                out_channel_layer = base_inc_channel // (feature_dilation ** (i))
            if dropout and i == 0:
                layer_dropout = dropout
                # in_channel_layer = out_channel_layer
            else:
                layer_dropout = None
            if i == 0:
                first_layer = True
                self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                         planes_in=last_cov_channels, planes_out=out_channel_layer,
                                         dropout=layer_dropout, kernel_size=kernel_size,
                                         first_layer=first_layer, input_size=input_size))
            else:
                first_layer = False

                self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                         planes_in=in_channel_layer+last_cov_channels, planes_out=out_channel_layer,
                                         dropout=layer_dropout, kernel_size=kernel_size,
                                         first_layer=first_layer, input_size=input_size))
            if i != len(layer_blocks) - 1:

                padding = kernel_size // 2  # constant size
                # maxpool3d_layer = nn.Conv3d(out_channel_layer, out_channel_layer, (kernel_size, kernel_size, kernel_size), padding=padding,
                #              stride=(4,4,4),
                #                 bias=True)
                if use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_channel_layer, out_channel_layer, kernel_size=kernel_size,
                                                                     stride=upsampling_stride, padding=1))
                else:
                    self.pre_upsampling_blocks.append(conv1x1x1(in_channel_layer, out_channel_layer, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_stride,
                                                          mode=upsampling_mode, align_corners=align_corners))

                input_size = input_size *2
                last_cov_channels = last_cov_channels//2
            #print("Decoder {}:".format(i), in_channel_layer, out_channel_layer)
            in_channel_layer = out_channel_layer
        self.out_channel_layer = in_channel_layer
    def forward(self, x):
        i = 0
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1]):
            if i == 0:
                y = lay(x[0])
            else:
                y = lay(y)
            #x = pre(x)
            y = up(y)
            y = torch.cat([y, x[i + 1]],1)
            i += 1
        y = self.layers[-1](y)
        return y

class UNetDecoder(DecoderLayer):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0]
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            x = pre(x)
            x = up(x)
            x = torch.cat((x, inputs[i + 1]), 1)
        x = self.layers[-1](x)
        return x



class MRI_bet(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, **kwargs):
        global dilated
        dilated = False
        dropout = 0
        global block_used

        block = ResidualBlock
        block_used = 'residual'
        super().__init__(*args, encoder_class=encoder_class, decoder_class=UNetDecoder, dropout=dropout,
                         block=block, **kwargs)
        self.netName = 'MRI_bet'
