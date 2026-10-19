# save as fix_weights3.py
import torch
from models.swin_transformer import SwinTransformer

# Load HuggingFace weights
raw = torch.load('pretrained/swin_tiny_patch4_window7_224.pth', map_location='cpu')
hf = raw['model'] if 'model' in raw else raw

# Load HAEFNet model to get target state dict
model = SwinTransformer()
model_sd = model.state_dict()

converted = {}

for hf_key, hf_val in hf.items():

    # ── patch_embed.proj ─────────────────────────────────────────────────────
    if hf_key == 'swin.embeddings.patch_embeddings.projection.weight':
        converted['patch_embed.proj.module.weight'] = hf_val
    elif hf_key == 'swin.embeddings.patch_embeddings.projection.bias':
        converted['patch_embed.proj.module.bias'] = hf_val

    # ── patch_embed.norm (LayerNorm repeated 3x for parallel streams) ────────
    elif hf_key == 'swin.embeddings.norm.weight':
        converted['patch_embed.norm.layers.0.weight'] = hf_val
        converted['patch_embed.norm.layers.1.weight'] = hf_val
        converted['patch_embed.norm.layers.2.weight'] = hf_val
    elif hf_key == 'swin.embeddings.norm.bias':
        converted['patch_embed.norm.layers.0.bias'] = hf_val
        converted['patch_embed.norm.layers.1.bias'] = hf_val
        converted['patch_embed.norm.layers.2.bias'] = hf_val

    # ── block norm1 (layernorm_before) ───────────────────────────────────────
    elif '.layernorm_before.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.layernorm_before.weight', '')
        new_key = f"{base}.norm1.module.weight"
        converted[new_key] = hf_val

    elif '.layernorm_before.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.layernorm_before.bias', '')
        new_key = f"{base}.norm1.module.bias"
        converted[new_key] = hf_val

    # ── block norm2 (layernorm_after) ────────────────────────────────────────
    elif '.layernorm_after.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.layernorm_after.weight', '')
        new_key = f"{base}.norm2.module.weight"
        converted[new_key] = hf_val

    elif '.layernorm_after.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.layernorm_after.bias', '')
        new_key = f"{base}.norm2.module.bias"
        converted[new_key] = hf_val

    # ── attention: relative_position_bias_table ───────────────────────────────
    elif '.attention.self.relative_position_bias_table' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.attention.self.relative_position_bias_table', '')
        new_key = f"{base}.attn.module.relative_position_bias_table"
        converted[new_key] = hf_val

    # ── attention: relative_position_index ───────────────────────────────────
    elif '.attention.self.relative_position_index' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.attention.self.relative_position_index', '')
        new_key = f"{base}.attn.module.relative_position_index"
        converted[new_key] = hf_val

    # ── attention: proj (output dense) ───────────────────────────────────────
    elif '.attention.output.dense.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.attention.output.dense.weight', '')
        new_key = f"{base}.attn.module.proj.weight"
        converted[new_key] = hf_val

    elif '.attention.output.dense.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.attention.output.dense.bias', '')
        new_key = f"{base}.attn.module.proj.bias"
        converted[new_key] = hf_val

    # ── mlp fc1 ──────────────────────────────────────────────────────────────
    elif '.intermediate.dense.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.intermediate.dense.weight', '')
        new_key = f"{base}.mlp.fc1.module.weight"   # ← removed .module. between mlp and fc1
        converted[new_key] = hf_val

    elif '.intermediate.dense.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.intermediate.dense.bias', '')
        new_key = f"{base}.mlp.fc1.module.bias"     # ← removed .module. between mlp and fc1
        converted[new_key] = hf_val

    # ── mlp fc2 ──────────────────────────────────────────────────────────────
    elif '.output.dense.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.output.dense.weight', '')
        new_key = f"{base}.mlp.fc2.module.weight"   # ← removed .module. between mlp and fc2
        converted[new_key] = hf_val

    elif '.output.dense.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.output.dense.bias', '')
        new_key = f"{base}.mlp.fc2.module.bias"     # ← removed .module. between mlp and fc2
        converted[new_key] = hf_val

    # ── downsample ────────────────────────────────────────────────────────────
    elif '.downsample.reduction.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.downsample.reduction.weight', '')
        new_key = f"{base}.downsample.module.reduction.weight"  # ← added .module.
        converted[new_key] = hf_val

    elif '.downsample.norm.weight' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.downsample.norm.weight', '')
        new_key = f"{base}.downsample.module.norm.weight"       # ← added .module.
        converted[new_key] = hf_val

    elif '.downsample.norm.bias' in hf_key:
        base = hf_key.replace('swin.encoder.', '') \
                     .replace('.downsample.norm.bias', '')
        new_key = f"{base}.downsample.module.norm.bias"         # ← added .module.
        converted[new_key] = hf_val

    # ── final norm ────────────────────────────────────────────────────────────
    elif hf_key == 'swin.layernorm.weight':
        converted['norm.weight'] = hf_val
    elif hf_key == 'swin.layernorm.bias':
        converted['norm.bias'] = hf_val

# ── QKV: merge separate Q, K, V into single qkv weight ───────────────────────
# HuggingFace has separate query/key/value; HAEFNet expects merged qkv
q_keys = {k: v for k, v in hf.items() if '.attention.self.query.' in k}
for q_key, q_val in q_keys.items():
    base = q_key.replace('swin.encoder.', '') \
                .replace('.attention.self.query.weight', '') \
                .replace('.attention.self.query.bias', '')
    suffix = 'weight' if '.weight' in q_key else 'bias'
    k_key  = q_key.replace('.query.', '.key.')
    v_key  = q_key.replace('.query.', '.value.')
    if k_key in hf and v_key in hf:
        k_val = hf[k_key]
        v_val = hf[v_key]
        qkv   = torch.cat([q_val, k_val, v_val], dim=0)
        new_key = f"{base}.attn.module.qkv.{suffix}"
        converted[new_key] = qkv

# ── Check how many match ──────────────────────────────────────────────────────
matched   = {k: v for k, v in converted.items() if k in model_sd}
unmatched = [k for k in model_sd if k not in converted]

print(f"Converted keys : {len(converted)}")
print(f"Matched        : {len(matched)} / {len(model_sd)}")
print(f"Still missing  : {len(unmatched)}")
if unmatched:
    print("Missing keys (first 10):")
    for k in unmatched[:10]:
        print(" ", k)

torch.save({'model': converted}, 'pretrained/swin_tiny_fixed.pth')
print("\nSaved to pretrained/swin_tiny_fixed.pth")