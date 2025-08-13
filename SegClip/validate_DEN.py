"""
-------------------------------------------------
One-file validation of a trained FLAIRModel on the DEN evaluation CSV with masks.
* Adds the correct absolute prefix to every relative image and mask path
* Parses first label string from CSV categories column
* Maps to canonical labels with abbreviation expansion and manual overrides
* Runs model.forward(image, mask, CLASS_NAMES)
* Outputs Macro-F1 and MCC for single-label classification
-------------------------------------------------

CSV columns:
idx,image,mask,attributes,categories
-------------------------------------------------
"""

import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
import torch

from flair.modeling.dictionary import definitions, abbreviations
from new_flair import FLAIRModel

# ----------------------------
# CONFIGURATION
# ----------------------------

CSV_PATH     = "/mnt/Segbiclip/local_data/dataframes/pretraining/DEN_VAL.csv"
DATASET_ROOT = "/mnt/"
WEIGHTS_PATH = "/mnt/Segbiclip/local_data/results/pretraining/From30resnet_v2_epoch_seg60.pth"
IMAGE_SIZE   = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# HELPER: Label normalization and mapping
# ----------------------------

manual_map = {
    "bvo": "branch retinal vein occlusion",
    "branch vein occlusion (bvo)": "branch retinal vein occlusion",
    "ped": "pigment epithelial detachment",
    "icsc": "idiopathic central serous chorioretinopathy",
    "md": "macular degeneration",
    "macular drusen": "drusen",
    "cuticular drusen": "drusen",
    "pohs": "presumed ocular histoplasmosis syndrome",
    "pic": "punctate inner choroidopathy",
    "aofped": "adult-onset foveomacular pigment epithelial dystrophy",
    "drusen": "drusen",
    "myopic degeneration": "myopic degeneration",
    "pigment epithelial detachment (ped)": "pigment epithelial detachment",
    "presumed ocular histoplasmosis syndrome (pohs)": "presumed ocular histoplasmosis syndrome",
    "retinal telangiectasia": "retinal telangiectasia",
    "juxtafoveal telangiectasis": "retinal telangiectasia",
    "best disease": "best vitelliform macular dystrophy",
    "suprachoroidal hemorrhage": "suprachoroidal hemorrhage",
    "punctate inner choroidopathy (pic)": "punctate inner choroidopathy",
    "acute macular neuroretinopathy": "acute macular neuroretinopathy",
    "cone dystrophy": "cone dystrophy",
    "malignant melanoma": "choroidal melanoma",
    "posterior microphthalmos": "posterior microphthalmos",
    "goldmann-favre syndrome": "goldmann-favre syndrome",
    "choroidal rupture": "choroidal rupture",
    "retinal metastasis": "retinal metastasis",
    "von hippel-lindau": "von hippel-lindau disease",
    "papillophlebitis": "papillophlebitis",
    "idiopathic central serous choroidopathy (icsc)": "idiopathic central serous chorioretinopathy",
    "central areolar choroidal sclerosis": "central areolar choroidal dystrophy",
    "angioid streaks": "angioid streaks",
    "morning glory syndrome": "morning glory disc anomaly",
    "toxoplasmosis": "ocular toxoplasmosis",
    "scleritis": "scleritis",
    "basal laminar drusen": "drusen",
    "birdshot choroidopathy": "birdshot chorioretinopathy",
    "familial exudative vitreoretinopathy (fevr)": "familial exudative vitreoretinopathy",

    # Non-diagnostic labels
    "14-year-old with optic pit.": None,
    "59-year-old white male": None,
}

abbr_to_full = {abbr.lower(): full for full, abbr in abbreviations.items()}

def replace_abbreviations(text):
    words = text.lower().split()
    return ' '.join([abbr_to_full.get(w, w) for w in words])

def is_diagnostic_label(text):
    if not isinstance(text, str):
        return False
    blacklist = ["year-old", "male", "female", "white", "black", "patient"]
    return not any(term in text.lower() for term in blacklist)

def map_label_to_canonical(free_text_label):
    if not isinstance(free_text_label, str) or not is_diagnostic_label(free_text_label):
        return None

    text = replace_abbreviations(free_text_label.strip().lower())

    if text in manual_map:
        return manual_map[text]

    for key, phrases in definitions.items():
        for phrase in phrases + [key]:
            if phrase.lower() in text:
                return key

    return None

# ----------------------------
# LOAD MODEL
# ----------------------------

model = FLAIRModel(
    vision_type="resnet_v2",
    from_checkpoint=True,
    weights_path=WEIGHTS_PATH
).to(DEVICE)
model.eval()
print(f"✅ Loaded FLAIR weights from: {WEIGHTS_PATH}")

# ----------------------------
# LOAD CSV & PREPARE LABELS
# ----------------------------

df = pd.read_csv(CSV_PATH)
df.columns = ["idx", "image", "mask", "attributes", "categories"]

# Absolute paths
df["image_path"] = df["image"].apply(lambda p: os.path.join(DATASET_ROOT, p))
df["mask_path"] = df["mask"].apply(lambda p: os.path.join(DATASET_ROOT, p))

# Parse label from list
df["label_text"] = df["categories"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else None)
df["canonical_label"] = df["label_text"].apply(map_label_to_canonical)

# Build class list and label IDs
CLASS_NAMES = sorted(df["canonical_label"].dropna().unique())
LABEL2ID = {label: i for i, label in enumerate(CLASS_NAMES)}
df["label_id"] = df["canonical_label"].map(LABEL2ID)

# Log unmatched
print("❌ Top unmatched label texts:")
print(df[df["canonical_label"].isna()]["label_text"].value_counts().head(30))

# Drop invalid labels
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

# Drop missing files
exists = df["image_path"].apply(os.path.exists) & df["mask_path"].apply(os.path.exists)
if (~exists).any():
    print(f"⚠️  Skipping {len(df) - exists.sum()} samples with missing image or mask")
    df = df[exists]

print(f"✅ Validation samples: {len(df)}")
print(f"✅ Classes: {CLASS_NAMES}")

# ----------------------------
# VALIDATION LOOP
# ----------------------------

y_true, y_pred = [], []

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

#        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
#            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
#        if mask.size != (IMAGE_SIZE, IMAGE_SIZE):
#            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_np = np.array(img)
        mask_np = np.array(mask)
        true_id = row["label_id"]

        probs, _ = model(img_np, mask_np, CLASS_NAMES)
        pred_id = int(np.argmax(probs, axis=1)[0])

        if idx < 10:
            print(f"\n🔎 Sample {idx}")
            print(f"  Image path: {row['image_path']}")
            print(f"  Mask path : {row['mask_path']}")
            print(f"  True ID   : {true_id} ({CLASS_NAMES[true_id]})")
            print(f"  Predicted : {pred_id} ({CLASS_NAMES[pred_id]})")

        y_true.append(true_id)
        y_pred.append(pred_id)

# ----------------------------
# METRICS
# ----------------------------

f1_macro = f1_score(y_true, y_pred, average="macro")
mcc = matthews_corrcoef(y_true, y_pred)

print("\n📊 DEN Segmentation Validation Results")
print(f"Macro-F1 : {f1_macro:.4f}")
print(f"MCC      : {mcc:.4f}")
