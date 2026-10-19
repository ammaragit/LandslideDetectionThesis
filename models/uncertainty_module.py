"""
RO2 — Uncertainty-Aware Cross-Modal Attention Module
Estimates per-pixel confidence scores for each modality (SAR, optical, DEM)
and uses these weights during feature-level fusion.
Degraded inputs are automatically down-weighted before fusion occurs.
Produces modality confidence heatmaps as XAI output.

Implementation: Monte Carlo Dropout on each modal encoder branch.
Multiple forward passes → variance → confidence score per modality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalConfidenceHead(nn.Module):
    """
    Lightweight head that estimates a per-pixel confidence score
    for a single modality feature map.
    High confidence → modality is reliable → contribute more to fusion.
    Low confidence  → modality is degraded → contribute less.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid = max(in_channels // reduction, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid()                    # output in [0, 1]
        )

    def forward(self, x):
        return self.head(x)              # (B, 1, H, W)


class UncertaintyFusionModule(nn.Module):
    """
    RO2: Uncertainty-aware cross-modal attention fusion module.

    Given feature maps from SAR, optical, and DEM encoders:
    1. Estimates per-pixel confidence score for each modality
    2. Normalises scores so they sum to 1 across modalities (softmax)
    3. Computes confidence-weighted sum of features
    4. Returns fused features + confidence maps (XAI output)

    This moves uncertainty estimation from the decision layer (HAEFNet GEDM)
    into the feature extraction backbone, preventing unreliable features
    from propagating through the network before fusion.
    """
    def __init__(self, feature_channels, num_modalities=3):
        """
        feature_channels : int  — channels in each modal feature map
        num_modalities   : int  — number of input modalities (default 3)
        """
        super().__init__()
        self.num_modalities = num_modalities

        # One confidence head per modality
        self.confidence_heads = nn.ModuleList([
            ModalConfidenceHead(feature_channels)
            for _ in range(num_modalities)
        ])

        # Optional projection to refine fused features
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, modal_features):
        """
        modal_features: list of (B, C, H, W) tensors — one per modality
        Returns:
            fused      : (B, C, H, W) uncertainty-weighted fused features
            conf_maps  : (B, num_modalities, H, W) confidence heatmaps (XAI)
        """
        assert len(modal_features) == self.num_modalities

        # ── Estimate raw confidence per modality ───────────────────────────
        raw_confs = []
        for feat, head in zip(modal_features, self.confidence_heads):
            c = head(feat)              # (B, 1, H, W)  in [0,1]
            raw_confs.append(c)

        # ── Normalise across modalities (softmax) so weights sum to 1 ──────
        # Stack: (B, num_modalities, H, W)
        conf_stack = torch.cat(raw_confs, dim=1)
        # Softmax across modality dimension
        conf_weights = F.softmax(conf_stack, dim=1)   # (B, M, H, W)

        # ── Confidence-weighted fusion ─────────────────────────────────────
        fused = torch.zeros_like(modal_features[0])
        for i, feat in enumerate(modal_features):
            w = conf_weights[:, i:i+1, :, :]   # (B, 1, H, W)
            # Resize weight map to feature map size if needed
            if w.shape[2:] != feat.shape[2:]:
                w = F.interpolate(w, size=feat.shape[2:],
                                  mode='bilinear', align_corners=False)
            fused = fused + w * feat

        # ── Refine fused features ─────────────────────────────────────────
        fused = self.fusion_proj(fused)

        return fused, conf_weights


class MCDropoutWrapper(nn.Module):
    """
    Wraps a feature extraction module with Monte Carlo Dropout.
    During inference, dropout remains active and multiple forward passes
    are used to estimate epistemic uncertainty per modality.

    Usage:
        wrapper = MCDropoutWrapper(encoder, dropout_p=0.1)
        mean, variance = wrapper.mc_forward(x, n_samples=10)
        confidence = 1 / (1 + variance)
    """
    def __init__(self, module, dropout_p=0.1):
        super().__init__()
        self.module    = module
        self.dropout_p = dropout_p
        self.dropout   = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        return self.dropout(self.module(x))

    def mc_forward(self, x, n_samples=10):
        """
        Run n_samples stochastic forward passes.
        Returns mean and variance of features across samples.
        Mean   → fused representation
        Variance → epistemic uncertainty estimate
        """
        self.train()    # keep dropout active during inference
        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                outputs.append(out)

        outputs = torch.stack(outputs, dim=0)   # (n, B, C, H, W)
        mean    = outputs.mean(dim=0)            # (B, C, H, W)
        var     = outputs.var(dim=0)             # (B, C, H, W)
        return mean, var


def build_uncertainty_module(feature_channels, num_modalities=3):
    """Factory function for clean instantiation from builder.py"""
    return UncertaintyFusionModule(
        feature_channels=feature_channels,
        num_modalities=num_modalities
    )
