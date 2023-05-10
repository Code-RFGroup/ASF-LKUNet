from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from .model_parts import *


class ASF_LKUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=9):
        super(MCM_Hit_mor, self).__init__()


        num_channels = [32,64,128,256,512]
        up_num_channels = [num_channels[0] * 2,
                           num_channels[1] * 2,
                           num_channels[2] * 2,
                           num_channels[3] * 2]


        self.inc = Inconv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.endown1 = encoder_dw(in_channels = num_channels[0], out_channels = num_channels[1])    #B,64,128,128
        self.endown2 = encoder_dw(in_channels = num_channels[1], out_channels = num_channels[2])    #B,128,64,64
        self.endown3 = encoder_dw(in_channels = num_channels[2], out_channels = num_channels[3])    #B,192,32,32
        self.endown4 = encoder_dw(in_channels = num_channels[3], out_channels = num_channels[4])    #B,256,16,16

        self.merge1 = merge_c2f_d(in_channels = num_channels[1], out_channels = num_channels[0],type= 'up') #256 + 128，256：down
        self.merge2 = merge_c2f_t(down_channels = num_channels[0], up_channels = num_channels[2],
                                out_channels = num_channels[1])                                             #128 + 64，64：upsample
        self.merge3 = merge_c2f_t(down_channels = num_channels[1], up_channels = num_channels[3],
                                out_channels = num_channels[2],     )
        self.merge4 = merge_c2f_t(down_channels = num_channels[2], up_channels = num_channels[4],
                                out_channels = num_channels[3],     )
        self.merge5 = merge_c2f_d(in_channels = num_channels[3], out_channels = num_channels[4],type= 'down')
       
        self.up1 = de_up(in_channels = num_channels[1],up_channels=up_num_channels[0],out_channels = num_channels[0])            #B,256,16,16
        self.up2 = de_up(in_channels = num_channels[2],up_channels=up_num_channels[1],out_channels = num_channels[1])             #B,192,32,32
        self.up3 = de_up(in_channels = num_channels[3],up_channels=up_num_channels[2],out_channels = num_channels[2])             #B,128,64,64
        self.up4 = de_up(in_channels = num_channels[4],up_channels=up_num_channels[3],out_channels = num_channels[3])   #B,64,128,128

        self.outc = OutConv(num_channels[0], n_classes)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.endown1(x1)
        x3 = self.endown2(x2)
        x4 = self.endown3(x3)
        x5 = self.endown4(x4)

        merge1 = self.merge1(x1, x2)
        merge2 = self.merge2(x1, x2,x3)
        merge3 = self.merge3(x2, x3,x4)
        merge4 = self.merge4(x3, x4,x5)
        merge5 = self.merge5(x5, x4)

        up4 = self.up4(merge5, merge4)
        up3 = self.up3(up4, merge3)
        up2 = self.up2(up3, merge2)
        up1 = self.up1(up2, merge1)
        out = self.outc(up1)
        return out

