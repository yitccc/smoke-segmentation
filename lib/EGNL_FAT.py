###################################################################################################################
#  EGNL-FAT:An Edge-Guided Non-Local Network with Frequency-Aware Transformer for Smoke Segmentation
#  Date: 2024/**/**
#
##################################################################################################################
import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from .DeiT import deit_small_patch16_224 as deit
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class cofuse(nn.Module):
    def __init__(self, td_channels, in_channels):
        super(cofuse, self).__init__()
        self.conv_1x1_x2 = nn.Conv2d(td_channels, in_channels, kernel_size=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_final = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x2_conv = self.conv_1x1_x2(x2)
        x2_global_pool = self.global_pooling(x2_conv)
        x1_global_pool = self.global_pooling(x1)
        x2_weighted = x2_global_pool * x1
        x1_weighted = x1_global_pool * x2_conv
        combined = torch.cat((x2_weighted, x1_weighted), dim=1)

        out = self.conv_1x1_final(combined)
        out = self.relu(out)

        return out



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


def laplacian_and_add(input_feature_map):
    laplacian_output = laplacian_convolution(input_feature_map).to(device)
    laplacian_output_relu = F.relu(laplacian_output)
    result = input_feature_map + laplacian_output_relu
    return result

def laplacian_convolution(input_feature_map):
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(1, input_feature_map.size(1), 1, 1).to(device)

    laplacian_conv = F.conv2d(input_feature_map, laplacian_kernel, stride=1, padding=1)
    return laplacian_conv


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CC_module(nn.Module):
    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(y)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class FAT_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(FAT_S, self).__init__()

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True)
            )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True)
            )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True)
            )

        self.fa = Conv(64, num_classes, 3, bn=False, relu=False)
        self.fb = Conv(64, num_classes, 3, bn=False, relu=False)
        self.fc = Conv(64, num_classes, 3, bn=False, relu=False)


        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        self.laplacian_and_add = laplacian_and_add

        self.CC_module = CC_module(64)

        self.conv_output0 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv_output1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv_output2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv_output3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        self.cls_conv = nn.Conv2d(64, 1, 1, stride=1)
        self.cofuse1 = cofuse(384,256)
        self.Cood1 = CoordAtt(256,256)
        self.cofuse2 = cofuse(128,128)
        self.Cood2 = CoordAtt(128,128)
        self.cofuse3 = cofuse(64,64)
        self.Cood3 = CoordAtt(64,64)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 12, 16)
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)

        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        cf0 = self.laplacian_and_add(imgs)
        cf0 = self.conv_output0(cf0)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        eb0 = F.interpolate(x_u_2, size=(192, 256), mode='nearest')
        eb0 = self.conv_output1(eb0)
        concatenated_features0 = torch.cat([eb0, cf0], dim=1)
        lm0 = self.conv_output2(concatenated_features0)
        lm0 = self.cls_conv(lm0)
        result0 = torch.mul(lm0, cf0)
        cf1 = self.laplacian_and_add(result0)
        cf1 = self.conv_output1(cf1)

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)
        eb1 = F.interpolate(x_u_1, size=(192, 256), mode='nearest')
        eb1 = self.conv_output2(eb1)
        concatenated_features1 = torch.cat([eb1, cf1], dim=1)
        lm1 = self.conv_output2(concatenated_features1)
        lm1 = self.cls_conv(lm1)
        result1 = torch.mul(lm1, cf1)
        cf2 = self.laplacian_and_add(result1)
        cf2 = self.conv_output1(cf2)

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)
        eb2 = F.interpolate(x_u, size=(192, 256), mode='nearest')
        eb2 = self.conv_output3(eb2)
        concatenated_features2 = torch.cat([eb2, cf2], dim=1)
        lm2 = self.conv_output2(concatenated_features2)
        lm2 = self.cls_conv(lm2)
        result2 = torch.mul(lm2, cf2)

        x_c = self.cofuse1(x_u, x_b)
        x_c = self.Cood1(x_c)

        x_c_1_1 = self.cofuse2(x_u_1, x_b_1)
        x_c_1_1 = self.Cood2(x_c_1_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.cofuse3(x_u_2, x_b_2)
        x_c_2_1 = self.Cood3(x_c_2_1)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1) # joint predict low supervise here

        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear')
        ma = self.CC_module(map_x, result2)
        ma = torch.add(map_x, ma)
        map_x = self.fa(ma)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear')
        mb = self.CC_module(map_1, result2)
        mb = torch.add(map_1, mb)
        map_1 = self.fb(mb)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear')
        mc = self.CC_module(map_2, result2)
        mc = torch.add(map_2, mc)
        map_2 = self.fc(mc)
        return map_x, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x