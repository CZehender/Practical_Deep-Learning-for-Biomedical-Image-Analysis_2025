import os
import numpy as np
import json
from PIL import Image, ImageOps
import random
import csv  # <-- Add this import

# Set random seed for reproducibility
random.seed(42)
size=(643, 438)
test_set_size=10  # in prozent

# Paths
script_location = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_location, "deepeyenet")

# Input folders
img_train_dir = os.path.join(data_root, "test_set")

dataset_name="Dataset001_RetinalLesions"

# Output folders
imagesTr = os.path.join(data_root, "preprocessed_test_images")
os.makedirs(imagesTr, exist_ok=True)
csv_output_path = os.path.join(imagesTr, "filename_test_mapping.csv")

# --- Image helpers ---
def center_crop_and_resize(image, target_size=(643, 438), interpolation=Image.BILINEAR):
    target_ratio = target_size[0] / target_size[1]
    width, height = image.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_height = int(width / target_ratio)
        pad = (new_height - height) // 2
        padding = (0, pad, 0, new_height - height - pad)
    else:
        new_width = int(height * target_ratio)
        pad = (new_width - width) // 2
        padding = (pad, 0, new_width - width - pad, 0)

    padded = ImageOps.expand(image, padding, fill=0)
    resized = padded.resize(target_size, interpolation)
    return resized

def masked_gaussian_blur(image, mask, ksize=(45, 45), sigma=0):
    """
    Applies Gaussian blur to an image, ignoring masked-out pixels.
    image: single-channel image (e.g., green channel)
    mask: binary mask (1 = keep, 0 = ignore)
    """
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    # Blur the image and the mask
    blurred_image = cv2.GaussianBlur(image * mask, ksize, sigma)
    blurred_mask = cv2.GaussianBlur(mask, ksize, sigma)

    # Avoid division by zero
    blurred_mask[blurred_mask == 0] = 1.0

    # Normalize
    return (blurred_image / blurred_mask).astype(np.uint8)


def enhance_lesions(image: Image.Image, size, interpolation=Image.BICUBIC) -> Image.Image:
    
    # Convert PIL Image to RGB NumPy array
    img_rgb = np.array(image.convert("RGB"))

    # Extract channels
    red, green, blue = cv2.split(img_rgb)

    # Local green background estimation
    green_blur = cv2.GaussianBlur(green, (21, 21), 0)

    # Difference reversed: green excess (more green than background)
    green_excess = (green.astype(np.int16) - green_blur.astype(np.int16))

    # Threshold for significant green excess patches
    deficit_thresh = 11
    mask = (green_excess > deficit_thresh).astype(np.uint8) * 255

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (45, 45), 0)

    # Normalize mask to [0,1]
    mask_f = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_f] * 3, axis=-1)

    # Brightness boost amount
    brighten_amount = 250

    # Convert to float for manipulation
    img_float = img_rgb.astype(np.float32)

    # Brighten pixels in masked areas
    fused = img_float + mask_3ch * brighten_amount


    # Estimate local green background with Gaussian blur
    # Use a smaller blur kernel for local green background estimation, e.g. 15x15 instead of 25x25
    valid_background_mask = (mask == 0).astype(np.uint8)

    # Apply masked blur to exclude bright lesions when estimating background for red lesion detection
    green_blur_red_lesions = masked_gaussian_blur(green, valid_background_mask, ksize=(45, 45))

    # Proceed as usual
    green_deficit = (green_blur_red_lesions.astype(np.int16) - green.astype(np.int16))
    deficit_thresh = 6
    mask_red_lesions = (green_deficit > deficit_thresh).astype(np.uint8) * 255


    # Optional: Smooth the mask for soft edges
    mask_red_lesions = cv2.GaussianBlur(mask_red_lesions, (7, 7), 0)

    # Normalize mask to [0, 1]
    mask_f_red_lesions = mask_red_lesions.astype(np.float32) / 255.0
    mask_3ch_red_lesions = np.stack([mask_f_red_lesions] * 3, axis=-1)

    # Convert to float for manipulation
    fused_float = fused.astype(np.float32)

    # Parameters for darkening
    darken_amount = 100  # how much darker the patch gets overall
    #green_reduction = 100  # extra reduction specifically for green channel

    # Darken all channels where mask is active
    fused = fused_float - mask_3ch_red_lesions * darken_amount

    # Further reduce green channel in mask areas
    #fused[:, :, 1] -= mask_f_red_lesions * green_reduction

    # Clip to valid [0,255]
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    out_image = Image.fromarray(fused)
    out_image = center_crop_and_resize(out_image, size, interpolation)

    return out_image

def load_image(img_path):
    return Image.open(img_path).convert("RGB")

def save_image(img, path):
    img.save(path)

# --- Process Training Images ---
print("Processing training set...")

# Collect all filenames
train_filenames = [f for f in os.listdir(img_train_dir) if f.endswith(".jpg")]

# Extract unique id_strs
id_strs = set(fname.split("_")[0] for fname in train_filenames)
id_strs = sorted(id_strs)
test_ids = set(random.sample(id_strs, max(1, len(id_strs) // test_set_size)))  # 10% for test

# Mapping list for CSV
filename_mapping = []

counter = 0
for i, fname in enumerate(train_filenames, 1):
    img = load_image(os.path.join(img_train_dir, fname))
    img= enhance_lesions(img, size)

    new_filename = f"Deepeyenet_{counter}_0000.png"
    save_image(img, os.path.join(imagesTr, new_filename))

    # Save mapping
    filename_mapping.append((fname, new_filename))
    counter += 1

    if i % 50 == 0 or i == len(train_filenames):
        print(f"Processed {i}/{len(train_filenames)} images")

# --- Save mapping to CSV ---
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['original_filename', 'new_filename'])  # header
    writer.writerows(filename_mapping)

print(f"Filename mapping saved to {csv_output_path}")
