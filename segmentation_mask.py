from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # <-- Add this

# ===== CONFIG =====
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic").to(device)

input_dir = "./flat"
output_dir = "./test_mask2"
os.makedirs(output_dir, exist_ok=True)

def resize_image(img, max_size=(256, 256)):
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

# ===== HELPER FUNCTION =====
def save_masks(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Save original (resized)
    image_resized = resize_image(image.copy())
    image_resized.save(os.path.join(output_dir, f"{image_name}.png"))
    
    # Predict
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    pred = pred.cpu().numpy()
    
    # Create masks
    unique_classes = np.unique(pred)
    for cls_id in unique_classes:
        mask = (pred == cls_id).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_img = resize_image(mask_img)
        mask_img.save(os.path.join(output_dir, f"{image_name}_mask{cls_id:03d}.png"))

# ===== RUN OVER ALL IMAGES =====
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for filename in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, filename)
    save_masks(image_path)

print(f"Processing complete! All images and masks saved to: {output_dir}")
