import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import foolbox as fb
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
img_per_class = 100

class ModelWrapper(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        labels = self.base_model(x_norm)
        logits = torch.zeros(x.size(0), 10, device=x.device)
        logits.scatter_(1, labels.view(-1, 1), 1.0)
        return logits

# Load model
base_model = torch.jit.load("./data/p2_data/cnn_model.pt", map_location=device)
base_model.eval()
model = ModelWrapper(base_model, mean, std).to(device)
model.eval()

# Prepare data: keep them in [0, 1] for Foolbox
transform = transforms.ToTensor() 
paths = [f"./data/p2_data/clean_images/{name}_{i}.png" for name in class_names for i in range(img_per_class)]
labels = [class_names.index(name) for name in class_names for _ in range(img_per_class)]

X = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths], dim=0).to(device)
y_true = torch.tensor(labels).to(device)

fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

# We use a decision-based attack. HopSkipJump is excellent for this.
# To balance queries and imperceptibility, we use 20 steps.
attack = fb.attacks.HopSkipJumpAttack(steps=15)

adv_images_dir = "./data/p2_data/adv_images"
os.makedirs(adv_images_dir, exist_ok=True)

print("Starting attack...")
batch_size = 50 # batching the attack
all_advs = []

start_time = time.time()
for i in tqdm(range(0, len(X), batch_size)):
    x_batch = X[i:i+batch_size]
    y_batch = y_true[i:i+batch_size]
    
    # Run attack
    _, advs, success = attack(fmodel, x_batch, criterion=fb.criteria.Misclassification(y_batch), epsilons=None)
    
    # save images
    for j in range(len(x_batch)):
        idx = i + j
        # advs is a list of tensors for different epsilons, but epsilons=None returns just the tensor
        adv_img = advs[j].cpu().detach().permute(1, 2, 0).numpy()
        adv_img = (adv_img * 255).astype(np.uint8)
        img = Image.fromarray(adv_img)
        
        orig_path = paths[idx]
        basename = os.path.basename(orig_path).replace(".png", "_adv.png")
        img.save(os.path.join(adv_images_dir, basename))
        
        all_advs.append(advs[j])

print(f"Attack completed in {time.time() - start_time:.2f} seconds.")

# Evaluation
X_adv = torch.stack(all_advs, dim=0).to(device)
y_pred_adv = model(X_adv).argmax(dim=-1)
asr = (y_pred_adv != y_true).float().mean().item()
print(f"Attack Success Rate: {asr * 100:.2f}%")

# L2 Distance
l2_dist = torch.norm((X_adv - X).view(len(X), -1), p=2, dim=-1).mean().item()
print(f"Average L2 Distortion: {l2_dist:.4f}")

