import torch

# Load the .pt model
model = torch.load('best.pt')

# Save as .pth
torch.save(model, 'best.pth')