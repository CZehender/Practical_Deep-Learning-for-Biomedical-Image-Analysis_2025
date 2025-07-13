import os
import pandas as pd
from PIL import Image
import numpy as np

# === Paths ===
script_location = os.path.dirname(os.path.abspath(__file__))

original_path = os.path.join(script_location, "test_set")
preprocessed_path = os.path.join(script_location, "preprocessed_test_images")
mask_path = os.path.join(script_location, "segmented_val")

output_restored_images = os.path.join(script_location, "restored_test_images")
output_restored_masks = os.path.join(script_location, "restored_test_masks")

os.makedirs(output_restored_images, exist_ok=True)
os.makedirs(output_restored_masks, exist_ok=True)

# === Load filename mapping ===
df = pd.read_csv("filename_test_mapping.csv", delimiter=";")

def reverse_preprocessing(preprocessed_img, original_img_size, target_size=(643, 438), is_mask=False):
    orig_w, orig_h = original_img_size
    orig_ratio = orig_w / orig_h
    target_ratio = target_size[0] / target_size[1]

    # Determine the padded size and crop box
    if orig_ratio > target_ratio:
        padded_h = int(orig_w / target_ratio)
        pad = (padded_h - orig_h) // 2
        padded_size = (orig_w, padded_h)
        crop_box = (0, pad, orig_w, pad + orig_h)
    else:
        padded_w = int(orig_h * target_ratio)
        pad = (padded_w - orig_w) // 2
        padded_size = (padded_w, orig_h)
        crop_box = (pad, 0, pad + orig_w, orig_h)

    # Resize to padded size (use NEAREST for masks to avoid interpolation artifacts)
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    resized_back = preprocessed_img.resize(padded_size, interp)

    # Crop out the padding
    restored_img = resized_back.crop(crop_box)
    return restored_img

# === Process all images and masks ===
for _, row in df.iterrows():
    original_filename = row["original_filename"]
    new_filename = row["new_filename"]
    mask_filename = new_filename.replace("_0000", "")
    try:
        # Load original image to get size
        orig_img = Image.open(os.path.join(original_path, original_filename))
        original_size = orig_img.size

        # === Load and binarize the input mask ===
        prep_mask = Image.open(os.path.join(mask_path, mask_filename)).convert('L')
        prep_mask_np = np.array(prep_mask)
        prep_mask_np = np.where(prep_mask_np > 0, 1, 0).astype(np.uint8)  # binarize
        prep_mask_bin = Image.fromarray(prep_mask_np * 255)  # scale to 0 and 255 for saving

        # === Reverse preprocessing ===
        restored_mask = reverse_preprocessing(prep_mask_bin, original_size, is_mask=True)

        # === Binarize restored mask again to be safe ===
        restored_np = np.array(restored_mask)
        restored_np = np.where(restored_np > 0, 1, 0).astype(np.uint8)
        restored_bin_mask = Image.fromarray(restored_np * 255)

        # Save mask
        restored_bin_mask.save(os.path.join(output_restored_masks, original_filename))

        print(f"Processed: {original_filename}")
    except Exception as e:
        print(f"Error processing {original_filename}: {e}")
    

