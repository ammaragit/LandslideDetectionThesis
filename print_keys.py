
# save as print_keys.py
import torch
from models.swin_transformer import SwinTransformer

# HAEFNet expected keys
model = SwinTransformer()
model_keys = list(model.state_dict().keys())
print("=== HAEFNet expects (first 15) ===")
for k in model_keys[:15]:
    print(" ", k)

# HuggingFace keys
weights = torch.load('pretrained/swin_tiny_patch4_window7_224.pth', map_location='cpu')
if 'model' in weights:
    hf_keys = list(weights['model'].keys())
else:
    hf_keys = list(weights.keys())

print("\n=== HuggingFace has (first 15) ===")
for k in hf_keys[:15]:
    print(" ", k)