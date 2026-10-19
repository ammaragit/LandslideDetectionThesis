
# save as check_weights.py
import torch
from models.swin_transformer import SwinTransformer

weights = torch.load('pretrained/swin_tiny_fixed.pth', map_location='cpu')
state_dict = weights['model']

model = SwinTransformer()
matched = {k: v for k, v in state_dict.items() if k in model.state_dict()}
print(f"Matched: {len(matched)} / {len(model.state_dict())} tensors")
print("First 5 matched:", list(matched.keys())[:5])