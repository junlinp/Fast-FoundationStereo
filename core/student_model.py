import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from timm import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DLAFriendlyFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model(
            'mobilenetv3_small_100', pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3))
        self.chans = [16, 16, 24, 48]
        self.d_out = [32, 64, 96, 128]
        self.proj_x4 = nn.Sequential(
            nn.Conv2d(self.chans[0], self.d_out[0], 1, bias=False),
            nn.BatchNorm2d(self.d_out[0]), nn.ReLU())
        self.proj_x8 = nn.Sequential(
            nn.Conv2d(self.chans[1], self.d_out[1], 1, bias=False),
            nn.BatchNorm2d(self.d_out[1]), nn.ReLU())
        self.proj_x16 = nn.Sequential(
            nn.Conv2d(self.chans[2], self.d_out[2], 1, bias=False),
            nn.BatchNorm2d(self.d_out[2]), nn.ReLU())
        self.proj_x32 = nn.Sequential(
            nn.Conv2d(self.chans[3], self.d_out[3], 1, bias=False),
            nn.BatchNorm2d(self.d_out[3]), nn.ReLU())

    def forward(self, x):
        features = self.backbone(x)
        return [self.proj_x4(features[0]), self.proj_x8(features[1]),
                self.proj_x16(features[2]), self.proj_x32(features[3])]


class DLAConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class DLAMotionEncoder(nn.Module):
    def __init__(self, hidden_dim=128, corr_channels=14):
        super().__init__()
        self.convc = nn.Sequential(
            nn.Conv2d(corr_channels, 128, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.convd = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.merge = nn.Sequential(
            nn.Conv2d(128 + 64, hidden_dim - 1, 1), nn.BatchNorm2d(hidden_dim - 1), nn.ReLU())

    def forward(self, disp, corr):
        return torch.cat([self.merge(torch.cat([self.convc(corr), self.convd(disp)], dim=1)), disp], dim=1)


class DLAUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, corr_channels=14):
        super().__init__()
        self.encoder = DLAMotionEncoder(hidden_dim, corr_channels)
        self.gru = DLAConvGRU(hidden_dim, hidden_dim * 2)
        self.disp_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1))
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

    def forward(self, net, inp, corr, disp):
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, motion_features)
        return net, 0.25 * self.mask(net), self.disp_head(net)


class DLAUpsample(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 9, 4, stride=2, padding=1))

    def forward(self, x):
        return F.softmax(self.conv(x), dim=1)


class DLASpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.BatchNorm2d(channels // 4), nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class DistilledStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dtype = torch.float32

        # Cross-attention for correlation (causal: left only attends to right at >= position)
        ca_dim = 128
        self.ca_q = nn.Linear(128, ca_dim)  # Q from inp (128ch)
        self.ca_k = nn.Linear(64, ca_dim)   # K from right features (64ch)
        self.ca_v = nn.Linear(64, ca_dim)   # V from right features (64ch)
        self.ca_out = nn.Conv2d(ca_dim, 14, 1)  # project to corr_channels

        self.feature = DLAFriendlyFeature()
        hidden_dim = args.hidden_dims[0]

        self.cnet = nn.Sequential(
            nn.Conv2d(64, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1), nn.BatchNorm2d(hidden_dim * 2), nn.ReLU())

        self.update_block = DLAUpdateBlock(hidden_dim, corr_channels=14)
        self.spx_gru = DLAUpsample(in_channels=32)
        self.stem_2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        self.sam = DLASpatialAttention(hidden_dim)

    def normalize_image(self, img):
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img / 255.0 - mean) / std

    def _cross_attention_corr(self, feat_left, feat_right):
        """Cross-attention: Q from left attends to K,V from right with causal mask.
        Pixel at position w only attends to pixels at w' >= w (rightward only).
        feat_left: (B, 128, H, W) from inp
        feat_right: (B, 64, H, W) from features_right[1]
        Returns: (B, 14, H, W) correlation features
        """
        B, _, H, W = feat_left.shape
        ca_dim = 128
        Q = self.ca_q(feat_left.permute(0, 2, 3, 1)).view(B * H, W, ca_dim)
        K = self.ca_k(feat_right.permute(0, 2, 3, 1)).view(B * H, W, ca_dim)
        V = self.ca_v(feat_right.permute(0, 2, 3, 1)).view(B * H, W, ca_dim)

        scale = float(ca_dim) ** -0.5
        attn = torch.matmul(Q, K.transpose(1, 2)) * scale

        # Causal mask: pixel w can only attend to w' >= w
        mask = torch.triu(torch.ones(W, W, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        corr = torch.matmul(attn, V).view(B, H, W, ca_dim).permute(0, 3, 1, 2)
        return self.ca_out(corr)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        spx_pred = self.spx_gru(mask_feat_4)[:, :, :disp.shape[2], :disp.shape[3]]
        return F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, image1, image2, iters=2, test_mode=False, init_disp=None):
        B, C, H, W = image1.shape
        image1, image2 = self.normalize_image(image1), self.normalize_image(image2)

        out = self.feature(torch.cat([image1, image2], dim=0))
        features_left, features_right = [o[:B] for o in out], [o[B:] for o in out]
        stem_2x = self.stem_2(image1)

        if init_disp is None:
            init_disp = torch.zeros(B, 1, H // 4, W // 4, device=features_left[1].device)

        cnet_out = self.cnet(features_left[1])
        hidden_dim = cnet_out.shape[1] // 2
        net, inp = cnet_out[:, :hidden_dim], cnet_out[:, hidden_dim:]

        disp = init_disp
        for _ in range(iters):
            disp = disp.detach()
            corr = self._cross_attention_corr(inp, features_right[1])
            net, mask, delta_disp = self.update_block(net, inp, corr, disp)
            disp = disp + delta_disp

        return self.upsample_disp(disp, mask, stem_2x)


DistilledStereoLite = DistilledStereo
