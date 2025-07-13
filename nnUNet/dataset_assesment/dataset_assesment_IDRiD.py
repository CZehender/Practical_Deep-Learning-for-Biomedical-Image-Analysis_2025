import os
import numpy as np
np.bool = np.bool_  # Monkey-patch the deprecated alias
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

labels = [
    "EX",
    "MA",
    "HE",
    "SE",
]

spacing = (1.0, 1.0)  # Adjust to your data's pixel spacing (x,y)
tolerance_mm = 1.0
size=(643, 438)         # This is the same size as the predicted masks
dataset_name="IDRiD"

# Containers for subclass metrics per class (index from 1 to num_classes)
# Initialize occurrence as count (int), and joint_occurrence as nested dict
occurrence = {label: 0 for label in labels}
joint_occurrence = {label: {other_label: 0 for other_label in labels} for label in labels}
overlap = overlap = {label: {other_label: [] for other_label in labels} for label in labels}
lesion_size = {label: [] for label in labels}

# Paths -- adjust these to your system!
script_location = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_location, "A. Segmentation", "A. Segmentation")

# Input folders
ref_dir = os.path.join(data_root, "1. Original Images", "All_resized")         # reference masks 
subclass_dir = os.path.join(data_root, "2. All Segmentation Groundtruths", "All")  # subclass folder root





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




def per_class_metrics(label_mask, subclass_mask):
    """
    Computes recall and overlap between prediction and a subclass mask.
    Both inputs are binary numpy arrays.
    """
    total_label = np.sum(label_mask >= 1)

    if total_label == 0:
        overlap = None
        print("Predicted mask empty")
    else:
        overlap = np.sum((label_mask == 1) & (subclass_mask == 1)) / total_label

    return overlap


def format_stat(val):
    return f"{val:.3f}" if val is not None else "N/A"

counter=0
for fname in os.listdir(ref_dir):
    if not fname.endswith('.png'):
        continue
    counter+=1
    # Extract the case ID (e.g., "421220") from "IDRiD_421220_0000.png"
    case_id = fname.replace('.png', '')
    id_str = case_id.split('_')[-2]  # Gets "421220"

    ref_path = os.path.join(ref_dir, fname)

    # Loop over subclasses for per-class metrics
    for subclass in labels:
        mask_path = os.path.join(subclass_dir, f"IDRiD_{id_str}_{subclass}.tif")
        #print(f"Mask path: {mask_path}")
        if not os.path.exists(mask_path):
            continue
        occurrence[subclass]+=1
        #print(f"occurrence for {subclass}: {occurrence[subclass]}")
        subclass_mask = load_subclass_mask(mask_path)
        size_percentage = (np.sum(subclass_mask >= 1))/(size[0]* size[1])*100
        lesion_size[subclass].append(size_percentage)

        for other_subclass in labels:
            other_mask_path = os.path.join(subclass_dir, f"IDRiD_{id_str}_{other_subclass}.tif")
            if not os.path.exists(other_mask_path):
                continue
            joint_occurrence[subclass][other_subclass]+=1
            other_subclass_mask = load_subclass_mask(other_mask_path)
            overlap_ = per_class_metrics(subclass_mask, other_subclass_mask)
            if overlap_ is not None:
                overlap[subclass][other_subclass].append(overlap_)


for subclass in labels:
    for other_subclass in labels:
        joint_occurrence[subclass][other_subclass]=joint_occurrence[subclass][other_subclass]/occurrence[subclass]*100


# Initialize storage for stats
table_data = []
size_means = []
size_stds = []
total_occurrence_percents = []

for subclass in labels:
    size_array = np.array(lesion_size[subclass])
    mean_size = np.mean(size_array) if size_array.size > 0 else 0
    std_size = np.std(size_array) if size_array.size > 0 else 0
    percentual_total_occurrence = occurrence[subclass] / counter * 100 if counter > 0 else 0

    size_means.append(mean_size)
    size_stds.append(std_size)
    total_occurrence_percents.append(percentual_total_occurrence)

    row = {}
    for other_subclass in labels:
        overlaps_array = np.array(overlap[subclass][other_subclass])  
        percentual_mean_overlap = np.mean(overlaps_array) * 100
        se_overlap = (np.std(overlaps_array, ddof=1) / np.sqrt(overlaps_array.size)) * 100
        percentual_joint_occurrence = joint_occurrence[subclass][other_subclass]
        cell_text = f"{percentual_mean_overlap:.1f}±{se_overlap:.1f}% | {percentual_joint_occurrence:.1f}%"
        row[other_subclass] = cell_text
    table_data.append(row)


# Create DataFrame
df_table = pd.DataFrame(table_data, index=labels, columns=labels)

# Save as CSV with semicolon delimiter
csv_path = os.path.join(data_root, "overlap_joint_table.csv")
df_table.to_csv(csv_path, sep=';', encoding='utf-8')

print(f"Saved overlap/joint table to: {csv_path}")


# Prepare data for boxplot: gather all lesion sizes per subclass
boxplot_data = [lesion_size[subclass] for subclass in labels]

# Box plot: Lesion size distributions
plt.figure(figsize=(10, 6))
plt.boxplot(boxplot_data, labels=labels, showfliers=False)
plt.ylabel("Lesion Size in % of the total image size")
plt.title("Size Distribution per Lesion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_root, "lesion_size_boxplot.png"))
plt.close()


# Bar plot: Percentual total occurrence
plt.figure(figsize=(10, 6))
plt.bar(labels, total_occurrence_percents, color='lightgreen')
plt.ylabel("Percentual Total Occurrence (%)")
plt.title("Lesion Class Occurrence in Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_root, "total_occurrence_barplot.png"))
plt.close()
