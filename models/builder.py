"""
Model builder — routes config to correct model class.
Supports: concat_unet (CM1), attention_unet (CM2),
          haefnet (M1/M2), haefnet_proposed (M3)
"""
import logging
from typing import Dict, Any, List

from .haef_net import HAEFNet
from .concat_unet import ConcatUNet
from .attention_unet import AttentionUNet
from .modules import set_num_parallel


def _collect_modalities(config: Dict[str, Any]) -> List[str]:
    modalities = [
        k[4:] for k, v in config.get("modalities", {}).items()
        if v and k.startswith("use_")
    ]
    if not modalities:
        raise ValueError(
            "No modalities enabled in config.modalities (use_* flags).")
    return modalities


# ── CM1 ───────────────────────────────────────────────────────────────────────
def build_concat_unet(config: Dict[str, Any]) -> ConcatUNet:
    model_cfg   = config.get("model", {})
    modalities  = _collect_modalities(config)
    num_modal   = len(modalities)
    in_channels = model_cfg.get("in_channels", num_modal * 3)

    logging.info("Building CM1 — ConcatUNet | in_channels=%d | classes=%d",
                 in_channels, model_cfg.get("num_classes", 2))

    return ConcatUNet(
        in_channels   = in_channels,
        num_classes   = model_cfg.get("num_classes", 2),
        base_channels = model_cfg.get("base_channels", 64),
    )


# ── CM2 ───────────────────────────────────────────────────────────────────────
def build_attention_unet(config: Dict[str, Any]) -> AttentionUNet:
    model_cfg = config.get("model", {})

    logging.info("Building CM2 — AttentionUNet (late fusion) | classes=%d",
                 model_cfg.get("num_classes", 2))

    return AttentionUNet(
        sar_channels  = model_cfg.get("sar_channels", 3),
        opt_channels  = model_cfg.get("opt_channels", 3),
        dem_channels  = model_cfg.get("dem_channels", 3),
        num_classes   = model_cfg.get("num_classes", 2),
        base_channels = model_cfg.get("base_channels", 32),
    )


# ── M1 / M2 (HAEFNet) ────────────────────────────────────────────────────────
def build_haefnet(config: Dict[str, Any]) -> HAEFNet:
    model_cfg      = config.get("model", {})
    modalities     = _collect_modalities(config)
    num_modalities = len(modalities)
    set_num_parallel(num_modalities)

    backbone             = model_cfg.get("backbone", "swin_tiny")
    num_classes          = model_cfg.get("num_classes", 2)
    n_heads              = model_cfg.get("n_heads", 8)
    dpr                  = model_cfg.get("drop_path_rate", 0.1)
    drop_rate            = model_cfg.get("drop_rate", 0.0)
    gem_prototype_dim    = model_cfg.get("gem_prototype_dim", 20)
    gem_geo_prior_weight = model_cfg.get("gem_geo_prior_weight", 0.1)
    use_evidential_fusion= model_cfg.get("use_evidential_fusion", True)
    use_aux_head         = model_cfg.get("use_aux_head", False)
    use_mrg              = model_cfg.get("use_mrg", False)
    prob_fusion          = model_cfg.get("prob_fusion", "product")
    use_dempster         = model_cfg.get("use_dempster", True)
    mrg_discount_on_mass = model_cfg.get("mrg_discount_on_mass", True)
    keep_pca_before_gem  = model_cfg.get("keep_pca_before_gem", True)
    aggregation_channels = model_cfg.get("aggregation_channels", 256)
    fusion_cfg           = model_cfg.get("fusion", {})
    fusion_params        = {
        "type":     fusion_cfg.get("type", "conditional"),
        "sparsity": fusion_cfg.get("sparsity", 0.5),
    }

    # Determine model variant for logging
    model_type = model_cfg.get("type", "haefnet")
    label      = "M3 — HAEFNet+RO1+RO2 (proposed)" \
                 if model_type == "haefnet_proposed" else "M1/M2 — HAEFNet"

    logging.info(
        "Building %s | backbone=%s | classes=%d | modalities=%s",
        label, backbone, num_classes, ",".join(modalities))

    model = HAEFNet(
        backbone             = backbone,
        num_classes          = num_classes,
        n_heads              = n_heads,
        dpr                  = dpr,
        drop_rate            = drop_rate,
        num_parallel         = num_modalities,
        fusion_params        = fusion_params,
        gem_prototype_dim    = gem_prototype_dim,
        gem_geo_prior_weight = gem_geo_prior_weight,
        use_evidential_fusion= use_evidential_fusion,
        use_aux_head         = use_aux_head,
        use_mrg              = use_mrg,
        prob_fusion          = prob_fusion,
        use_dempster         = use_dempster,
        mrg_discount_on_mass = mrg_discount_on_mass,
        keep_pca_before_gem  = keep_pca_before_gem,
        aggregation_channels = aggregation_channels,
    )

    # ── Attach RO2 uncertainty module for M3 ─────────────────────────────────
    if model_type == "haefnet_proposed" and \
       model_cfg.get("use_uncertainty_module", False):

        from .uncertainty_module import build_uncertainty_module

        # Feature channels at the aggregation level
        feat_ch = aggregation_channels

        model.uncertainty_module = build_uncertainty_module(
            feature_channels = feat_ch,
            num_modalities   = num_modalities,
        )
        model.use_uncertainty_module = True
        logging.info("RO2 uncertainty module attached — "
                     "feature_channels=%d, modalities=%d",
                     feat_ch, num_modalities)
    else:
        model.use_uncertainty_module = False

    return model


# ── Main entry point ──────────────────────────────────────────────────────────
def build_model(config: Dict[str, Any]):
    """
    Routes to correct model builder based on config['model']['type'].
    Called from train.py via build_haefnet() for backward compatibility.
    """
    model_type = config.get("model", {}).get("type", "haefnet")

    if model_type == "concat_unet":
        return build_concat_unet(config)
    elif model_type == "attention_unet":
        return build_attention_unet(config)
    elif model_type in ("haefnet", "haefnet_proposed"):
        return build_haefnet(config)
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: concat_unet, attention_unet, haefnet, "
            f"haefnet_proposed")
