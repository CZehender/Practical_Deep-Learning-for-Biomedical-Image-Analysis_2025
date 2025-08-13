import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef,roc_auc_score
import torch
from flair.modeling.dictionary import definitions, abbreviations
import ast

# -------------------------------------------------------------------
CSV_PATH      = ("/mnt/FLAIR/local_data/dataframes/transferability/"
                 "classification/37_DeepDRiD_test.csv")
DATASET_ROOT  = "/home/ubuntu/"           # absolute prefix for every rel_path
WEIGHTS_PATH  = "/mnt/FLAIR/local_data/results/pretraining/resnet_v2_epoch30.pth"
IMAGE_SIZE    = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASS_NAMES)}

# -------------------------------------------------------------------
# 2. LOAD FLAIR MODEL
# -------------------------------------------------------------------
from flair.modeling.model import FLAIRModel


model = FLAIRModel(
    vision_type="resnet_v2",
    from_checkpoint=True,
    weights_path=WEIGHTS_PATH
).to(DEVICE)

#state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
#model.load_state_dict(state, strict=False)
model.eval()
print(f"✅ Loaded FLAIR weights from: {WEIGHTS_PATH}")

# -------------------------------------------------------------------
# 3. LOAD CSV & FIX PATHS
# -------------------------------------------------------------------
df = pd.read_csv(CSV_PATH,skiprows=1, header=None)
df.columns = ["idx", "rel_path", "misc", "raw_label"]

# prepend absolute dataset root to relative image paths
df["image_path"] = df["rel_path"].apply(lambda p: os.path.join(DATASET_ROOT, p))

# safely parse label text from the raw_label column using ast.literal_eval
#print(df["raw_label"].headf["label_text"] = df["raw_label"].apply(lambda x: ast.literal_eval(x)[0])
df["label_text"] = df["raw_label"].apply(lambda x: ast.literal_eval(x)[0])
df["label_id"]   = df["label_text"].map(LABEL2ID)

# (optional) drop rows whose file is missing
missing = df[~df["image_path"].apply(os.path.exists)]
if not missing.empty:
    print(f"⚠️  {len(missing)} images missing, they will be skipped.")
    df = df[df["image_path"].apply(os.path.exists)]

print(f"✅  Validation samples: {len(df)}")

# -------------------------------------------------------------------
# 4. VALIDATION LOOP
# -------------------------------------------------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        img_path   = row["image_path"]
        true_id    = row["label_id"]

        # Load image → np.ndarray (H, W, 3)
        img_np = np.array(Image.open(img_path))

        # Forward FLAIR
        probs, _ = model(img_np, CLASS_NAMES)   # probs → (1, 5)
        pred_id  = int(np.argmax(probs, axis=1)[0])

        if idx < 5:
           print(f"\n🔎 Sample {idx}")
#           print(f"  Image path: {img_path}")
#           print(f"  True label ID: {true_id}")
#           print(f"  Binary GT: 0")
#           print(f"  Predicted label ID: {pred_id}")
#           print(f"  Image shape: {img_np.shape}")
#           print(f"  Data type: {img_np.dtype}")
#           print(f"  Pixel value range: min={img_np.min()}, max={img_np.max()}")
#           print(f"  Avg pixel value (overall): {img_np.mean():.2f}")
            # Average per channel (RGB)
#           avg_per_channel = img_np.mean(axis=(0,1))
#           print(f"  Avg per channel (R,G,B): {avg_per_channel}")
           print(f"True label: {true_id} ({CLASS_NAMES[true_id]})")
           print(f"Predicted label: {pred_id} ({CLASS_NAMES[pred_id]})")

        y_true.append(true_id)
        y_pred.append(pred_id)

# -------------------------------------------------------------------
# 5. METRICS
# -------------------------------------------------------------------
f1_macro = f1_score(y_true, y_pred, average="macro")
mcc      = matthews_corrcoef(y_true, y_pred)

#f1 = f1_score(y_true, y_pred)
#auc = roc_auc_score(y_true, y_pred)

print("\n📊  DeepDRiD Validation Results")
print(f"Macro‑F1 : {f1_macro:.4f}")
#print(f"F1 : {f1:.4f}") 
print(f"MCC      : {mcc:.4f}") 
#print(f"AUC      : {auc:.4f}")

