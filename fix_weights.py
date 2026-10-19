
# save as fix_weights.py in HAEFNet-main folder
import torch

# Load HuggingFace format
hf_weights = torch.load(
    'pretrained/swin_tiny_patch4_window7_224.pth',
    map_location='cpu'
)

# HuggingFace stores weights inside 'model' key
if 'model' in hf_weights:
    state_dict = hf_weights['model']
elif 'state_dict' in hf_weights:
    state_dict = hf_weights['state_dict']
else:
    state_dict = hf_weights

print(f"Keys found: {len(state_dict)}")
print("First 5 keys:", list(state_dict.keys())[:5])

# Save in the format HAEFNet expects
torch.save({'model': state_dict}, 'pretrained/swin_tiny_fixed.pth')
print("Saved to pretrained/swin_tiny_fixed.pth")