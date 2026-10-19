"""
CM1 — Simple Concatenation U-Net
Custom baseline 1: all modalities concatenated as 9-channel input → standard U-Net.
No cross-modal attention, no uncertainty. Establishes the performance floor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double conv block: Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool → ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ConcatUNet(nn.Module):
    """
    Simple U-Net that concatenates all modalities into a single tensor.
    Input:  list of tensors [sar, optical, dem], each (B, 3, H, W)
    Output: (outputs_list, aux_info) matching HAEFNet interface
    """
    def __init__(self, in_channels=9, num_classes=2, base_channels=64):
        super().__init__()
        b = base_channels

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, b)      # (B, 64,  H,    W)
        self.enc2 = DownBlock(b,     b * 2)        # (B, 128, H/2,  W/2)
        self.enc3 = DownBlock(b * 2, b * 4)        # (B, 256, H/4,  W/4)
        self.enc4 = DownBlock(b * 4, b * 8)        # (B, 512, H/8,  W/8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = DownBlock(b * 8, b * 16) # (B, 1024, H/16, W/16)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = UpBlock(b * 16 + b * 8,  b * 8)
        self.dec3 = UpBlock(b * 8  + b * 4,  b * 4)
        self.dec2 = UpBlock(b * 4  + b * 2,  b * 2)
        self.dec1 = UpBlock(b * 2  + b,      b)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_param_groups(self):
        """Match HAEFNet interface for optimizer."""
        params = list(self.parameters())
        return [params, [], []]

    def forward(self, images):
        """
        images: list of tensors [sar, optical, dem], each (B, 3, H, W)
        Returns: (outputs_list, aux_info) matching HAEFNet interface
        """
        # Concatenate all modalities
        x = torch.cat(images, dim=1)           # (B, 9, H, W)

        # Encoder
        s1 = self.enc1(x)                      # (B, 64,  H,    W)
        s2 = self.enc2(s1)                     # (B, 128, H/2,  W/2)
        s3 = self.enc3(s2)                     # (B, 256, H/4,  W/4)
        s4 = self.enc4(s3)                     # (B, 512, H/8,  W/8)
        bn = self.bottleneck(s4)               # (B,1024, H/16, W/16)

        # Decoder
        d4 = self.dec4(bn, s4)                 # (B, 512, H/8,  W/8)
        d3 = self.dec3(d4, s3)                 # (B, 256, H/4,  W/4)
        d2 = self.dec2(d3, s2)                 # (B, 128, H/2,  W/2)
        d1 = self.dec1(d2, s1)                 # (B, 64,  H,    W)

        # Segmentation head → log softmax (matches HAEFNet output format)
        logits    = self.head(d1)              # (B, 2, H, W)
        log_probs = F.log_softmax(logits, dim=1)

        # Resize to input size if needed
        h, w = images[0].shape[2:]
        if log_probs.shape[2:] != (h, w):
            log_probs = F.interpolate(log_probs, size=(h, w),
                                      mode='bilinear', align_corners=False)

        # Return matching HAEFNet interface
        outputs  = [log_probs]
        aux_info = {}
        return outputs, aux_info
