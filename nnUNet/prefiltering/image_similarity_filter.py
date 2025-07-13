import os
import shutil
import torch
from PIL import Image
import clip
from tqdm import tqdm
import numpy as np
import cv2
from itertools import combinations

# Paths
deepeyenet_path = "../deepeyenet/train_set"                                   #path to target dataset
model_dataset_path = "../nnUNet_raw/Dataset001_RetinalLesions/imagesTr"      #path to reference dataset
output_path = "../deepeyenet/filtered_retina_images"
number_of_stds=4
min_background_size = 500
os.makedirs(output_path, exist_ok=True)

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
model_filenames = []

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
        model_filenames.append(filename)

# Stack embeddings into a matrix
model_embeddings = torch.cat(model_embeddings, dim=0)

# Step 2: Compute intra-model_dataset similarity stats
print("Calculating intra-dataset similarity stats...")
with torch.no_grad():
    sim_matrix = model_embeddings @ model_embeddings.T
    # Remove self-similarity by masking diagonal
    sim_vals = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=bool)]
    mean_similarity = sim_vals.mean().item()
    std_similarity = sim_vals.std().item()
    similarity_cutoff = mean_similarity - number_of_stds * std_similarity

print(f"Mean similarity among model images: {mean_similarity:.4f}")
print(f"Standard deviation: {std_similarity:.4f}")
print(f"Using similarity threshold: max(sim >= {similarity_cutoff:.4f})")

# Step 3: Compare deepeyenet images to model_dataset embeddings
print("Filtering deepeyenet images...")
counter=0
for filename in tqdm(os.listdir(deepeyenet_path)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = load_image(os.path.join(deepeyenet_path, filename))
    if img is None:
        continue
    with torch.no_grad():
        emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        similarities = (emb @ model_embeddings.T).squeeze(0)
        max_sim = similarities.max().item()

    if max_sim >= similarity_cutoff:
        counter+=1
        shutil.copy(os.path.join(deepeyenet_path, filename),
                    os.path.join(output_path, filename))
print(f"New number of samples: {counter}")
