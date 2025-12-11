import os
import numpy as np
from PIL import Image
import torch
import lpips
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score

# ==== Paths ====
gt_dir = "./flat"
lama_dir = "./result_old"
refine_dir = "./result_new"

# Temporary folders for resized images
fid_gt = "./fid_temp/gt"
fid_lama = "./fid_temp/lama"
fid_refine = "./fid_temp/refine"

# ==== Logging Setup ====
log_file = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log = open(log_file, "w")

# Create folders if not exist
os.makedirs(fid_gt, exist_ok=True)
os.makedirs(fid_lama, exist_ok=True)
os.makedirs(fid_refine, exist_ok=True)

def write_log(message):
    print(message)
    log.write(message + "\n")

# ==== Image Loading ====
def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

# ==== Metric Calculator ====
def calc_metrics(gt_path, pred_path):
    gt = load_image(gt_path)
    pred = load_image(pred_path)

    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    gt_torch = torch.tensor(gt).permute(2, 0, 1).float() / 255
    pred_torch = torch.tensor(pred).permute(2, 0, 1).float() / 255

    loss_fn = lpips.LPIPS(net='vgg').to('cuda')
    lpips_val = loss_fn(gt_torch.unsqueeze(0).cuda(), pred_torch.unsqueeze(0).cuda()).item()

    psnr_val = psnr(gt, pred, data_range=255)
    ssim_val = ssim(gt, pred, channel_axis=2)

    return psnr_val, ssim_val, lpips_val

# ==== Evaluation Function ====
def evaluate(model_folder, label):
    psnr_list, ssim_list, lpips_list = [], [], []
    write_log(f"\n--- Evaluating {label} ---")

    pred_files = [f for f in os.listdir(model_folder) if "_mask" in f]

    for pred_name in tqdm(pred_files, desc=f"Processing {label}"):
        pred_path = os.path.join(model_folder, pred_name)

        base_name = pred_name.split("_mask")[0] + ".png"
        gt_path = os.path.join(gt_dir, base_name)

        if os.path.exists(gt_path):
            p, s, l = calc_metrics(gt_path, pred_path)
            psnr_list.append(p)
            ssim_list.append(s)
            lpips_list.append(l)

            write_log(f"{pred_name} | PSNR: {p:.4f} | SSIM: {s:.4f} | LPIPS: {l:.4f}")

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


# ==== Run Evaluation ====
lama_psnr, lama_ssim, lama_lpips = evaluate(lama_dir, "Base Model")
refine_psnr, refine_ssim, refine_lpips = evaluate(refine_dir, "Refined Model")

write_log("\n=== Final Results ===")
write_log(f"Base Model -> PSNR: {lama_psnr:.4f}, SSIM: {lama_ssim:.4f}, LPIPS: {lama_lpips:.4f}")
write_log(f"Refined Model -> PSNR: {refine_psnr:.4f}, SSIM: {refine_ssim:.4f}, LPIPS: {refine_lpips:.4f}")

def resize_and_save(src_folder, dst_folder):
    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Save with same name
            img.save(os.path.join(dst_folder, img_name))
        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")

# Resize all folders
print("\nResizing images for FID compatibility...")
resize_and_save(gt_dir, fid_gt)
resize_and_save(lama_dir, fid_lama)
resize_and_save(refine_dir, fid_refine)

print("Resizing complete. Running FID on resized sets...\n")

# ==== FID Calculation ====
write_log("\n--- Calculating FID (Folder-wise) ---")
fid_lama = fid_score.calculate_fid_given_paths([fid_gt, fid_lama], batch_size=32, device='cuda', dims=2048)
fid_refine = fid_score.calculate_fid_given_paths([fid_gt, fid_refine], batch_size=32, device='cuda', dims=2048)

write_log(f"FID (Base Model): {fid_lama:.4f}")
write_log(f"FID (Refined Model): {fid_refine:.4f}")

log.close()

# ==== Plot Comparison Graphs ====
metrics = ["PSNR", "SSIM", "LPIPS", "FID"]
base_values = [lama_psnr, lama_ssim, lama_lpips, fid_lama]
refine_values = [refine_psnr, refine_ssim, refine_lpips, fid_refine]

plt.figure()
plt.plot(metrics, base_values, marker='o', label='Base Model')
plt.plot(metrics, refine_values, marker='o', label='Refined Model')
plt.title("Model Comparison (PSNR, SSIM, LPIPS, FID)")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("model_comparison_metrics.png")
plt.show()
