"""
CM2 — Attention U-Net with Late Fusion
Custom baseline 2: three separate U-Net encoders (one per modality),
fused at the bottleneck via an attention gate, then decoded together.
Naive multi-modal fusion — no cross-modal interaction during encoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class AttentionGate(nn.Module):
    """
    Attention gate for modality fusion at the bottleneck.
    Learns which spatial locations to attend to from each modality.
    g: gating signal (query), x: encoder feature (key/value)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class ModalEncoder(nn.Module):
    """Single-modality U-Net encoder. Returns skip connections + bottleneck."""
    def __init__(self, in_ch=3, base_channels=32):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = DownBlock(b,     b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8)
        self.btn  = DownBlock(b * 8, b * 16)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn = self.btn(s4)
        return [s1, s2, s3, s4, bn]


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """
    Three separate modal encoders fused at the bottleneck via attention gates.
    Shared decoder reconstructs the segmentation map.
    Input:  list [sar, optical, dem], each (B, 3, H, W)
    Output: (outputs_list, aux_info) matching HAEFNet interface
    """
    def __init__(self, sar_channels=3, opt_channels=3, dem_channels=3,
                 num_classes=2, base_channels=32):
        super().__init__()
        b  = base_channels
        b3 = b * 3   # fused channel dim (3 modalities concatenated)

        # ── Three modal encoders ──────────────────────────────────────────────
        self.sar_enc = ModalEncoder(sar_channels, b)
        self.opt_enc = ModalEncoder(opt_channels, b)
        self.dem_enc = ModalEncoder(dem_channels, b)

        # ── Attention gates at bottleneck ─────────────────────────────────────
        # Gating signal is averaged bottleneck, applied to each modality
        btn_ch = b * 16
        self.att_sar = AttentionGate(btn_ch, btn_ch, btn_ch // 2)
        self.att_opt = AttentionGate(btn_ch, btn_ch, btn_ch // 2)
        self.att_dem = AttentionGate(btn_ch, btn_ch, btn_ch // 2)

        # ── Bottleneck fusion: concat 3 modalities → project ─────────────────
        self.btn_fusion = nn.Sequential(
            nn.Conv2d(btn_ch * 3, btn_ch, 1, bias=False),
            nn.BatchNorm2d(btn_ch),
            nn.ReLU(inplace=True))

        # ── Decoder (uses fused skip connections) ────────────────────────────
        self.dec4 = UpBlock(btn_ch     + b * 8  * 3, b * 8)
        self.dec3 = UpBlock(b * 8      + b * 4  * 3, b * 4)
        self.dec2 = UpBlock(b * 4      + b * 2  * 3, b * 2)
        self.dec1 = UpBlock(b * 2      + b      * 3, b)

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(b, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_param_groups(self):
        params = list(self.parameters())
        return [params, [], []]

    def forward(self, images):
        """
        images: [sar, optical, dem], each (B, 3, H, W)
        Returns: (outputs_list, aux_info)
        """
        sar, opt, dem = images[0], images[1], images[2]

        # ── Encode each modality separately ───────────────────────────────────
        sar_feats = self.sar_enc(sar)   # [s1,s2,s3,s4,btn]
        opt_feats = self.opt_enc(opt)
        dem_feats = self.dem_enc(dem)

        # Bottleneck features
        sar_btn = sar_feats[4]
        opt_btn = opt_feats[4]
        dem_btn = dem_feats[4]

        # ── Attention-weighted fusion at bottleneck ───────────────────────────
        # Gating signal = average of all three bottlenecks
        gate = (sar_btn + opt_btn + dem_btn) / 3.0

        sar_att = self.att_sar(gate, sar_btn)
        opt_att = self.att_opt(gate, opt_btn)
        dem_att = self.att_dem(gate, dem_btn)

        # Concatenate attended features → project to single bottleneck
        fused_btn = self.btn_fusion(
            torch.cat([sar_att, opt_att, dem_att], dim=1))

        # ── Decode with fused skip connections ────────────────────────────────
        # Concatenate skip connections from all three encoders at each level
        skip4 = torch.cat([sar_feats[3], opt_feats[3], dem_feats[3]], dim=1)
        skip3 = torch.cat([sar_feats[2], opt_feats[2], dem_feats[2]], dim=1)
        skip2 = torch.cat([sar_feats[1], opt_feats[1], dem_feats[1]], dim=1)
        skip1 = torch.cat([sar_feats[0], opt_feats[0], dem_feats[0]], dim=1)

        d4 = self.dec4(fused_btn, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        logits    = self.head(d1)
        log_probs = F.log_softmax(logits, dim=1)

        # Resize to input resolution if needed
        h, w = images[0].shape[2:]
        if log_probs.shape[2:] != (h, w):
            log_probs = F.interpolate(log_probs, size=(h, w),
                                      mode='bilinear', align_corners=False)

        outputs  = [log_probs]
        aux_info = {}
        return outputs, aux_info
