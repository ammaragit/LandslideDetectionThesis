
# save as check_mlp.py
import torch

weights = torch.load('pretrained/swin_tiny_fixed.pth', map_location='cpu')
converted = weights['model']

# Show all mlp keys that were converted
mlp_keys = [k for k in converted.keys() if 'mlp' in k]
print("MLP keys in converted (first 10):")
for k in mlp_keys[:10]:
    print(" ", k)

# Show what HAEFNet actually expects for mlp
from models.swin_transformer import SwinTransformer
model = SwinTransformer()
model_mlp = [k for k in model.state_dict().keys() if 'mlp' in k]
print("\nMLP keys HAEFNet expects (first 10):")
for k in model_mlp[:10]:
    print(" ", k)