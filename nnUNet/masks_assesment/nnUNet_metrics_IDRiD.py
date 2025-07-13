import os
import numpy as np
np.bool = np.bool_  # Monkey-patch the deprecated alias

from PIL import Image, ImageOps
from surface_distance import metrics

labels = [
    "MA",
    "EX",
    "HE",
    "SE",
]

spacing = (1.0, 1.0)  # Adjust to your data's pixel spacing (x,y)
tolerance_mm = 1.0
size=(643, 438)         # This is the same size as the predicted masks
dataset_name="IDRiD"
# Containers for overall metrics
all_dsc = []
all_nsd = []
all_recall = []
all_overlap = []

# Containers for subclass metrics per class (index from 1 to num_classes)
all_recalls = {label: [] for label in labels}
all_overlaps = {label: [] for label in labels}

# Paths -- adjust these to your system!
script_location = os.path.dirname(os.path.abspath(__file__))

# Input folders
ref_dir = os.path.join(script_location, "nnUNet_raw", "Dataset001_RetinalLesions", "IDRiD", "labelsTs")         # reference masks 
pred_dir = os.path.join(script_location, "nnUNet_raw", "Dataset001_RetinalLesions", "IDRiD", "fold0")       # predicted masks 
subclass_dir = os.path.join(script_location, "nnUNet_raw", "Dataset001_RetinalLesions", "IDRiD", "subclasses")   # subclass folder root





def open_mask(mask_path):
    return (np.array(Image.open(mask_path).convert("L"))).astype(np.uint8)

def center_crop_and_resize(image, target_size=(643, 438), interpolation=Image.NEAREST):
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

def load_subclass_mask(mask_path):
    mask = open_mask(mask_path)  # This function must return a NumPy array
    # Handle edge cases
    if np.sum(mask >= 1) == 0:
        print("No segmentation mask for subclass after loading")
    binary_mask = (mask > 0).astype(np.uint8)
    if np.sum(binary_mask >= 1) == 0:
        print("No segmentation mask for subclass after binary conversion")
    resized_mask=np.array(center_crop_and_resize(Image.fromarray(binary_mask), size, interpolation=Image.NEAREST)).astype(np.uint8)
    if np.sum(resized_mask >= 1) == 0:
        print("No segmentation mask for subclass after resizing")
    return resized_mask



def load_collapsed_mask(mask_path):
    return (np.array(Image.open(mask_path).convert("L"))).astype(np.uint8)


def dice_score(pred, gt):
    intersection = np.sum((pred == 1) & (gt == 1))
    size_sum = np.sum(pred == 1) + np.sum(gt == 1)
    return 2 * intersection / size_sum if size_sum != 0 else 1.0

def nsd(pred, gt, spacing=(1.0, 1.0), tolerance_mm=1.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    surface_distances = metrics.compute_surface_distances(gt, pred, spacing)
    return metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)




def per_class_metrics(pred_mask, subclass_mask):
    """
    Computes recall and overlap between prediction and a subclass mask.
    Both inputs are binary numpy arrays.
    """
    total_pred = np.sum(pred_mask >= 1)
    total_gt = np.sum(subclass_mask >= 1)

    # Handle edge cases
    if total_gt == 0:
        recall = None
        print("No segmentation mask for subclass")
    else:
        recall = np.sum((pred_mask == 1) & (subclass_mask == 1)) / total_gt
        #print(f"recall: {recall}")
    if total_pred == 0:
        overlap = None
        print("Predicted mask empty")
    else:
        overlap = np.sum((pred_mask == 1) & (subclass_mask == 1)) / total_pred
        #print(f"overlap: {overlap}")
    return recall, overlap


def format_stat(val):
    return f"{val:.3f}" if val is not None else "N/A"


for fname in os.listdir(ref_dir):
    if not fname.endswith('.png'):
        continue

    # Extract the case ID (e.g., "421220") from "RetinalLesions_421220.png"
    case_id = fname.replace('.png', '')
    id_str = case_id.split('_')[-1]  # Gets "421220"

    ref_path = os.path.join(ref_dir, fname)
    pred_fname = f"{dataset_name}_{id_str}.png"  # Expected filename in pred_dir
    pred_path = os.path.join(pred_dir, pred_fname)

    if not os.path.exists(pred_path):
        print(f"Missing predicted mask for {case_id}")
        continue

    ref_mask = load_collapsed_mask(ref_path)
    pred_mask = load_collapsed_mask(pred_path)

    # Dice and NSD
    dsc = dice_score(pred_mask, ref_mask)
    nsd_val = nsd(pred_mask, ref_mask, spacing, tolerance_mm)
    recall_val, overlap_val = per_class_metrics(pred_mask, ref_mask)

    all_dsc.append(dsc)
    all_nsd.append(nsd_val)
    if recall_val is not None:
        all_recall.append(recall_val)
    if overlap_val is not None:
        all_overlap.append(overlap_val)

    # Loop over subclasses for per-class metrics
    for subclass in labels:
        mask_path = os.path.join(subclass_dir, f"{dataset_name}_{id_str}_{subclass}.tif")
        #print(f"Mask path: {mask_path}")
        if not os.path.exists(mask_path):
            #print(f"Mask path for {subclass} does not exist")
            continue

        #print(f"Mask for {subclass} exists")
        subclass_mask = load_subclass_mask(mask_path)
        recall, overlap = per_class_metrics(pred_mask, subclass_mask)
        if recall is not None:
            all_recalls[subclass].append(recall)
        if overlap is not None:
            all_overlaps[subclass].append(overlap)



# After all cases processed, compute mean and standard error for DSC and NSD
mean_dsc = np.mean(all_dsc)
se_dsc = np.std(all_dsc, ddof=1) / np.sqrt(len(all_dsc)) if len(all_dsc) > 1 else None

mean_nsd = np.mean(all_nsd)
se_nsd = np.std(all_nsd, ddof=1) / np.sqrt(len(all_nsd)) if len(all_nsd) > 1 else None

mean_overlap = np.mean(all_overlap)
se_overlap = np.std(all_overlap, ddof=1) / np.sqrt(len(all_overlap)) if len(all_overlap) > 1 else None

mean_recall = np.mean(all_recall)
se_recall = np.std(all_recall, ddof=1) / np.sqrt(len(all_recall)) if len(all_recall) > 1 else None

print(f"Overall Dice Score: Mean = {mean_dsc:.4f} ± {se_dsc:.4f} (SE)")
print(f"Overall NSD: Mean = {mean_nsd:.4f} ± {se_nsd:.4f} (SE)")
print(f"Overall Overlap: Mean = {mean_overlap:.4f} ± {se_overlap:.4f} (SE)")
print(f"Overall Recall: Mean = {mean_recall:.4f} ± {se_recall:.4f} (SE)")

# Prepare output text
output_lines = []
output_lines.append(f"Overall Dice Score: Mean = {mean_dsc:.4f} ± {se_dsc:.4f} (SE)")
output_lines.append(f"Overall NSD: Mean = {mean_nsd:.4f} ± {se_nsd:.4f}")
output_lines.append(f"Overall Overlap: Mean = {mean_overlap:.4f} ± {se_overlap:.4f} (SE)")
output_lines.append(f"Overall Recall: Mean = {mean_recall:.4f} ± {se_recall:.4f} (SE)")
output_lines.append("")


for label in labels:
    recalls_array = np.array(all_recalls[label])
    overlaps_array = np.array(all_overlaps[label])
    
    mean_recall = np.mean(recalls_array) if recalls_array.size > 0 else None
    se_recall = (np.std(recalls_array, ddof=1) / np.sqrt(recalls_array.size)) if recalls_array.size > 1 else None
    
    mean_overlap = np.mean(overlaps_array) if overlaps_array.size > 0 else None
    se_overlap = (np.std(overlaps_array, ddof=1) / np.sqrt(overlaps_array.size)) if overlaps_array.size > 1 else None

    line = f"{label}: Recall {format_stat(mean_recall)} ± {format_stat(se_recall)}, Overlap {format_stat(mean_overlap)} ± {format_stat(se_overlap)}"
    print(line)

    output_lines.append(line)

# Save results to file in predicted mask folder
output_path = os.path.join(pred_dir, "evaluation_summary.txt")
with open(output_path, "w") as f:
    for line in output_lines:
        f.write(line + "\n")


print(f"Summary saved to {output_path}")





