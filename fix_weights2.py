# save as fix_weights2.py
import torch

hf_weights = torch.load('pretrained/swin_tiny_patch4_window7_224.pth', map_location='cpu')

if 'model' in hf_weights:
    state_dict = hf_weights['model']
elif 'state_dict' in hf_weights:
    state_dict = hf_weights['state_dict']
else:
    state_dict = hf_weights

# Mapping from HuggingFace keys → HAEFNet Swin keys
def convert_key(k):
    k = k.replace('swin.embeddings.patch_embeddings.projection.', 'patch_embed.proj.')
    k = k.replace('swin.embeddings.norm.', 'patch_embed.norm.')
    k = k.replace('swin.encoder.layers.', 'layers.')
    k = k.replace('.blocks.', '.blocks.')
    k = k.replace('.layernorm_before.', '.norm1.')
    k = k.replace('.layernorm_after.', '.norm2.')
    k = k.replace('.attention.self.query.', '.attn.q.')
    k = k.replace('.attention.self.key.',   '.attn.k.')
    k = k.replace('.attention.self.value.', '.attn.v.')
    k = k.replace('.attention.output.dense.', '.attn.proj.')
    k = k.replace('.intermediate.dense.', '.mlp.fc1.')
    k = k.replace('.output.dense.', '.mlp.fc2.')
    k = k.replace('.downsample.reduction.', '.downsample.reduction.')
    k = k.replace('.downsample.norm.', '.downsample.norm.')
    k = k.replace('swin.layernorm.', 'norm.')
    k = k.replace('.attention.self.relative_position_bias_table',
                  '.attn.relative_position_bias_table')
    k = k.replace('.attention.self.relative_position_index',
                  '.attn.relative_position_index')
    return k

converted = {}
for k, v in state_dict.items():
    new_k = convert_key(k)
    converted[new_k] = v

print(f"Total keys: {len(converted)}")
print("First 5 converted keys:", list(converted.keys())[:5])

torch.save({'model': converted}, 'pretrained/swin_tiny_fixed.pth')
print("Saved to pretrained/swin_tiny_fixed.pth")