import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer, TransformerDecoder2
from einops import rearrange
from .Memory import *
from .swin import Build_Backbone
from loss.losses import cross_entropy

class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=8)
        self.token_decoder = token_decoder(in_chan = in_chan, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out


class Classifier(nn.Module):
    def __init__(self, in_chan=64, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class CDNet(nn.Module):
    def __init__(self,  backbone='Swin_tiny_p4w7', output_stride=32, pretrained = True, img_size = 512, img_chan=3, chan_num = 32, n_class =2,pretrained_path='./checkpoints'):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone, self.channels_blocks, self.do_upsample = Build_Backbone(
            backbone, pretrained, img_chan, pretrained_path=pretrained_path)

        self.CA_s32 = context_aggregator(in_chan=768)
        self.CA_s16 = context_aggregator(in_chan=512)
        self.CA_s8 = context_aggregator(in_chan=160)
        self.CA_s4 = context_aggregator(in_chan=80)

        self.conv_s16 = nn.Conv2d(1152, 512, kernel_size=3, padding=1)
        self.conv_s8 = nn.Conv2d(704, 160, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(256, 80, kernel_size=3, padding=1)
        # self.conv = nn.Conv2d(160, 128, kernel_size=3, padding=1) # base
        self.conv = nn.Conv2d(160, 64, kernel_size=3, padding=1) #+memory

        self.m_items = F.normalize(torch.rand((n_class, 64), dtype=torch.float), dim=1)

        # self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)
        #
        # self.CA_s32 = context_aggregator(in_chan=chan_num)
        # self.CA_s16 = context_aggregator(in_chan=chan_num)
        # self.CA_s8 = context_aggregator(in_chan=chan_num)
        # self.CA_s4 = context_aggregator(in_chan=chan_num)
        #
        # self.conv_s16 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        # self.conv_s8 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        # self.conv_s4 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        # self.conv = nn.Conv2d(160, 64, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.classifier = Classifier(in_chan=64, n_class = n_class)
        self.memory = Memory(memory_size=n_class, feature_dim = 64, key_dim = 64, temp_update = 0.1, temp_gather=0.1)

    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        out1_s4, out1_s8, out1_s16, out1_s32  = self.backbone(img1)
        out2_s4, out2_s8, out2_s16, out2_s32 = self.backbone(img2)

        x1_s32 = self.CA_s32(out1_s32)
        x2_s32 = self.CA_s32(out2_s32)

        x1_s32 = F.interpolate(x1_s32, size=out1_s16.shape[2:], mode='bicubic', align_corners=True)
        x2_s32 = F.interpolate(x2_s32, size=out2_s16.shape[2:], mode='bicubic', align_corners=True)

        out1_s16 = self.conv_s16(torch.cat([x1_s32, out1_s16], dim=1))
        out2_s16 = self.conv_s16(torch.cat([x2_s32, out2_s16], dim=1))

        # context aggregate (scale 16, scale 8, scale 4)
        x1_s16= self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)

        x1_s16 = F.interpolate(x1_s16, size=out1_s8.shape[2:], mode='bicubic', align_corners=True)
        x2_s16 = F.interpolate(x2_s16, size=out2_s8.shape[2:], mode='bicubic', align_corners=True)
        out1_s8 = self.conv_s8(torch.cat([x1_s16, out1_s8], dim=1))
        out2_s8 = self.conv_s8(torch.cat([x2_s16, out2_s8], dim=1))

        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)

        x1_s8 = F.interpolate(x1_s8, size=out1_s4.shape[2:], mode='bicubic', align_corners=True)
        x2_s8 = F.interpolate(x2_s8, size=out2_s4.shape[2:], mode='bicubic', align_corners=True)
        out1_s4 = self.conv_s4(torch.cat([x1_s8, out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([x2_s8, out2_s4], dim=1))

        x1 = self.CA_s4(out1_s4)
        x2 = self.CA_s4(out2_s4)
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bicubic', align_corners=True)

        x = torch.cat([x1, x2], dim=1)
        x =self.conv(x)

        if self.training:
            updated_x, self.m_items = self.memory(x, self.m_items,train=True)
            x = self.classifier(updated_x)
            x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
            return x, self.m_items

        else:
            updated_x = self.memory(x, self.m_items,train=False)
            x = self.classifier(updated_x)
            x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
            return x
        #
        # x = self.classifier(x)
        # x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
        # return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

