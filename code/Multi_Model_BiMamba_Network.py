import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba.bimamba import Mamba as InBiMamba


class MAMBA(nn.Module):
    def __init__(self, seed, d_model, d_state, dconv, expand, len, bidirectional=True, bid_type='InBiMamba'):
        super(MAMBA, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dconv = dconv
        self.expand = expand
        self.seed = seed
        self.flag = 1
        self.bid_type = bid_type
        if bidirectional:
            if bid_type == 'InBiMamba':
                self.mamba = InBiMamba(seed=self.seed, d_model=self.d_model, d_state=self.d_state, d_conv=self.dconv, expand=self.expand, bimamba_type="v2")
            else:
                self.flag = 0
                self.mamba = nn.Identity()
        else:
            self.flag = 0
            self.mamba = nn.Identity()
        self.len = len

    def forward(self, x):
        x_res = x
        x = torch.squeeze(x, dim=2).permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, self.d_model, 1, self.len)
        if self.flag:
            return x + x_res
        else:
            return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Fusion(nn.Module):
    def fusion_block(self, in_chan, out_chan, kernel, stride):
        return nn.Sequential(
            SeparableConv2d(in_channels=in_chan,
                            out_channels=out_chan,
                            kernel_size=kernel,
                            stride=stride),
            nn.LeakyReLU()
        )

    def __init__(self, in_chan, out_chan, C, *args, **kwargs):
        super(Fusion, self).__init__(*args, **kwargs)
        self.C = C
        # Global spatial convolution (processes all channels together)
        self.spatial1 = self.fusion_block(
            in_chan, in_chan,
            kernel=(self.C, 1),
            stride=1
        )
        # Hemispheric spatial convolution (processes hemispheres separately)
        self.spatial2 = self.fusion_block(
            in_chan, in_chan,
            kernel=(self.C//2, 1),
            stride=(self.C//2, 1)
        )
        # Final fusion layer with batch normalization
        self.fusion_layer = nn.Sequential(
            nn.BatchNorm2d(in_chan),
            self.fusion_block(in_chan, out_chan, (3, 1), 1)
        )

    def forward(self, x):
        y1 = self.spatial1(x)
        y2 = self.spatial2(x)
        out_ = torch.cat((y1, y2), dim=2)
        out = self.fusion_layer(out_)
        return out


class extractor(nn.Module):
    def extractor_block(self, in_chan, out_chan, kernel, step, pool, nc):
        mid_chan = max(out_chan//2, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=mid_chan,
                      kernel_size=kernel, stride=step, padding=(0, kernel[1]//2)),
            nn.BatchNorm2d(mid_chan),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
            nn.Conv2d(in_channels=mid_chan, out_channels=out_chan,
                      kernel_size=(nc, 1), stride=1) if nc != 32 else Fusion(mid_chan, out_chan, 32),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        )

    def __init__(self, num_channel, sampling_rate, num_T,  dropout_rate, args):
        super(extractor, self).__init__()
        self.args = args
        self.window_size = [0.5, 0.25, 0.125]
        self.pool = 16
        self.W = 4*128

        self.branch1 = self.extractor_block(
            1, num_T, (1, int(self.window_size[0] * sampling_rate)), 1, self.pool, nc=num_channel)
        self.branch2 = self.extractor_block(
            1, num_T, (1, int(self.window_size[1] * sampling_rate)), 1, self.pool, nc=num_channel)
        self.branch3 = self.extractor_block(
            1, num_T, (1, int(self.window_size[2] * sampling_rate)), 1, self.pool, nc=num_channel)

        self.SMamba = MAMBA(seed=2025, d_state=36, d_model=num_T, dconv=4, expand=1, len=3*self.W//self.pool,
                            bidirectional=True, bid_type="InBiMamba")
        # 2025 36 4 1
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool2d(kernel_size=(1, 6), stride=(1, 6))
        self.cnn = nn.Conv2d(num_T, num_T, kernel_size=(1, 16), padding=(0, 8))

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        out = torch.cat((y1, y2, y3), dim=-1)
        out = self.SMamba(out)
        out = self.dropout(out)
        out = self.pool(self.cnn(out)).squeeze(dim=2)
        return out


class EEG_encoder(nn.Module):
    def __init__(self, args):
        super(EEG_encoder, self).__init__()
        self.module = extractor(num_channel=32, sampling_rate=128, num_T=16,
                          dropout_rate=0.5, args=args)
        self.dropout = 0.5
        self.num_classes = 2
        self.classify_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(16 * 16, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.module(x)
        out1 = self.classify_block(out)
        return out, out1


class EOG_encoder(nn.Module):
    def __init__(self, args):
        super(EOG_encoder, self).__init__()
        self.module = extractor(num_channel=2, sampling_rate=128, num_T=8,
                          dropout_rate=0.5, args=args)
        self.dropout = 0.5
        self.num_classes = 2
        self.classify_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(16 * 8, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.module(x)
        out1 = self.classify_block(out)
        return out, out1


class EMG_encoder(nn.Module):
    def __init__(self, args):
        super(EMG_encoder, self).__init__()
        self.module = extractor(num_channel=2, sampling_rate=128, num_T=6,
                          dropout_rate=0.5, args=args)
        self.dropout = 0.5
        self.num_classes = 2
        self.classify_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(16 * 6, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.module(x)
        out1 = self.classify_block(out)
        return out, out1


class Multi_Modal_BiMamba_Network(nn.Module):
    def __init__(self, args, **kwargs):
        super(Multi_Modal_BiMamba_Network, self).__init__(**kwargs)
        self.args = args
        self.dropout = 0.5
        self.num_classes = 2
        self.attention_num_len = 16 + 18 + 6
        self.encoder1 = EEG_encoder(args)
        self.encoder2 = EOG_encoder(args)
        self.encoder3 = EMG_encoder(args)
        if args.Fusion_type == 'SMF':
            self.SF = InBiMamba(seed=2025, d_state=12, d_model=16, d_conv=2, expand=1, bimamba_type="v2")
        else:
            self.SF = nn.Identity()
        self.end_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(16*(16+8+6), self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X1 = X[:, :, 0:32, :]
        X2 = X[:, :, 32:34, :]
        X3 = X[:, :, 34:36, :]
        Y1, out_y1 = self.encoder1(X1)
        Y2, out_y2 = self.encoder2(X2)
        Y3, out_y3 = self.encoder3(X3)
        Y = self.SF(torch.cat((Y1, Y2, Y3), dim=1))
        out = self.end_block(Y)
        return out, out_y1, out_y2, out_y3

    def forward_eval(self, X):
        X1 = X[:, :, 0:32, :]
        X2 = X[:, :, 32:34, :]
        X3 = X[:, :, 34:36, :]
        Y1, out_y1 = self.encoder1(X1)
        Y2, out_y2 = self.encoder2(X2)
        Y3, out_y3 = self.encoder3(X3)
        Y = self.SF(torch.cat((Y1, Y2, Y3), dim=1))
        out = self.end_block(Y)
        return out, out_y1, out_y2, out_y3, Y1, Y2, Y3
