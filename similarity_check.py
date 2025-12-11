import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model + preprocess
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
model.eval()

def embed(path):
    img = Image.open(path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    return feat.cpu().numpy()[0]

def cosine_dissim(a, b):
    return 1 - np.dot(a, b)

old_folder = "result_old"
new_folder = "result_new"

# Collect valid files
def is_image_file(filename):
    return filename.lower().endswith((".png",".jpg",".jpeg"))

def is_mask_file(filename):
    return filename.endswith("mask001.png") or filename.endswith("mask000.png") \
        or filename.endswith("mask001.jpg") or filename.endswith("mask000.jpg")

old_files = set(f for f in os.listdir(old_folder) if is_image_file(f) and not is_mask_file(f))
new_files = set(f for f in os.listdir(new_folder) if is_image_file(f) and not is_mask_file(f))

# Intersection
common_files = sorted(list(old_files & new_files))

print(f"Found {len(common_files)} matching filenames (excluding mask000/mask001).")

results = []

for fname in common_files:
    old_path = os.path.join(old_folder, fname)
    new_path = os.path.join(new_folder, fname)

    emb_old = embed(old_path)
    emb_new = embed(new_path)

    dissim = cosine_dissim(emb_old, emb_new)
    results.append((dissim, fname, old_path, new_path))

# Sort highest dissimilarity first
results.sort(reverse=True, key=lambda x: x[0])

# Output
print("\nTop most dissimilar paired images:")
for i, (d, fname, old_p, new_p) in enumerate(results[:20], start=1):
    print(f"[{i}] {fname}  |  Dissimilarity = {d:.4f}")
    print(f" old: {old_p}")
    print(f" new: {new_p}\n")
