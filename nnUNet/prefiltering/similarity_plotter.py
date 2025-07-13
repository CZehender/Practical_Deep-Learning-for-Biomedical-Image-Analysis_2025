import os
import shutil
import torch
from PIL import Image
import clip
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths
deepeyenet_path = "../deepeyenet/train_set"
model_dataset_path = "../nnUNet_raw/Dataset001_RetinalLesions/imagesTr"
output_base_path = "../deepeyenet/filtered_retina_images"
min_background_size = 500
os.makedirs(output_base_path, exist_ok=True)

# Load CLIP
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Helper: load and preprocess an image
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        white_thresh = 240
        black_thresh = 15

        white_mask = np.all(img_np >= white_thresh, axis=-1).astype(np.uint8)
        black_mask = np.all(img_np <= black_thresh, axis=-1).astype(np.uint8)

        def remove_large_regions(mask, img_np, threshold=min_background_size):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= threshold:
                    img_np[labels == i] = [127, 127, 127]
            return img_np

        img_np = remove_large_regions(white_mask, img_np)
        img_np = remove_large_regions(black_mask, img_np)

        img_filtered = Image.fromarray(img_np)
        return preprocess(img_filtered).unsqueeze(0).to(device)

    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

# Step 1: Build image embeddings for model_dataset
model_embeddings = []
print("Embedding reference images from model_dataset...")
for filename in tqdm(os.listdir(model_dataset_path)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = load_image(os.path.join(model_dataset_path, filename))
    if img is None:
        continue
    with torch.no_grad():
        emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        model_embeddings.append(emb)

model_embeddings = torch.cat(model_embeddings, dim=0)

# Step 2: Compute intra-model_dataset similarity stats
print("Calculating intra-dataset similarity stats...")
with torch.no_grad():
    sim_matrix = model_embeddings @ model_embeddings.T
    sim_vals = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=bool)]
    mean_similarity = sim_vals.mean().item()
    std_similarity = sim_vals.std().item()

print(f"Mean similarity: {mean_similarity:.4f}")
print(f"Standard deviation: {std_similarity:.4f}")

# Step 3: Loop over standard deviation thresholds and count matching images
std_range = np.arange(0.0, 10.1, 0.5)
results = []

print("Embedding deepeyenet images...")
deepeyenet_embeddings = []
deepeyenet_filenames = []

for filename in tqdm(os.listdir(deepeyenet_path)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = load_image(os.path.join(deepeyenet_path, filename))
    if img is None:
        continue
    with torch.no_grad():
        emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        deepeyenet_embeddings.append(emb)
        deepeyenet_filenames.append(filename)

deepeyenet_embeddings = torch.cat(deepeyenet_embeddings, dim=0)

print("Evaluating across std thresholds...")

for stds in std_range:
    similarity_cutoff = mean_similarity - stds * std_similarity
    kept = 0
    min_similarity = float("inf")
    min_filename = None

    for i in range(deepeyenet_embeddings.shape[0]):
        sim = (deepeyenet_embeddings[i] @ model_embeddings.T).max().item()
        
        if sim >= similarity_cutoff:
            kept += 1
            if sim < min_similarity:
                min_similarity = sim
                min_filename = deepeyenet_filenames[i]

    results.append(kept)
    print(f"Std {stds:.1f}: {kept} images kept, least similar among kept: {min_filename} (sim={min_similarity:.4f})")

    # Save the least similar image among those kept
    if min_filename is not None:
        src_path = os.path.join(deepeyenet_path, min_filename)
        dst_filename = f"{stds:.1f}_least_similar_{min_filename}"
        dst_path = os.path.join(output_base_path, dst_filename)
        shutil.copy2(src_path, dst_path)


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(std_range, results, marker='o')
plt.xlabel("Number of Standard Deviations (stds)")
plt.ylabel("Number of Images Kept")
plt.title("Filtered Image Count vs. Similarity Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_base_path, "image_similarity_plot.png"))
