import torch

# Load the checkpoint
checkpoint = torch.load('/home/krkavinda/ProjectX-RGBDL-C-ATTN/Lorcon-LO_models/checkpoint_kitti.pt', map_location='cpu', weights_only=False)
print("Checkpoint keys:", checkpoint.keys())

# Extract the model_state_dict
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

# Print the state_dict keys to identify feature extraction layers
print("Model state dict keys:")
for key in state_dict.keys():
    print(key)