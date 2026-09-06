import json
import os

with open("p2_code.ipynb", "r") as f:
    nb = json.load(f)

attack_code = """# IMPLEMENT YOUR ATTACK HERE
!pip install foolbox
import foolbox as fb
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

class ModelWrapper(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1).to(device))
        
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        labels = self.base_model(x_norm)
        logits = torch.zeros(x.size(0), 10, device=x.device)
        logits.scatter_(1, labels.view(-1, 1), 1.0)
        return logits

model_wrapped = ModelWrapper(model, mean, std).to(device)
model_wrapped.eval()

fmodel = fb.PyTorchModel(model_wrapped, bounds=(0, 1), device=device)
attack = fb.attacks.HopSkipJumpAttack(steps=15)

print("Starting attack...")
batch_size = 50
all_advs = []

for i in tqdm(range(0, len(X), batch_size)):
    x_batch = X[i:i+batch_size]
    y_batch = y_true[i:i+batch_size]
    
    _, advs, success = attack(fmodel, x_batch, criterion=fb.criteria.Misclassification(y_batch), epsilons=None)
    all_advs.append(advs)

# Combine all adversarial images
X_adv = torch.cat(all_advs, dim=0)
"""

save_code = """# SAVE ADVERSARIAL IMAGES
adv_images_dir = "./p2_data/adv_images"
os.makedirs(adv_images_dir, exist_ok=True)

for i in range(len(X_adv)):
    adv_img = X_adv[i].cpu().detach().permute(1, 2, 0).numpy()
    adv_img = (adv_img * 255).astype(np.uint8)
    img = Image.fromarray(adv_img)
    
    orig_path = paths[i]
    basename = os.path.basename(orig_path).replace(".png", "_adv.png")
    img.save(os.path.join(adv_images_dir, basename))

print(f"Saved {len(X_adv)} adversarial images to {adv_images_dir}")
"""

# Find the cells to replace
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if len(cell['source']) > 0 and 'IMPLEMENT YOUR ATTACK HERE' in cell['source'][0]:
            cell['source'] = [line + '\n' for line in attack_code.split('\n')]
        elif len(cell['source']) > 0 and 'SAVE ADVERSARIAL IMAGES' in cell['source'][0]:
            cell['source'] = [line + '\n' for line in save_code.split('\n')]

with open("p2_code.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
