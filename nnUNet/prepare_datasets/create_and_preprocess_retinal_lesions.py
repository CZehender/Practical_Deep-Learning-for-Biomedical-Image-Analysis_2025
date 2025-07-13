import os
import numpy as np
import json
from PIL import Image, ImageOps
import random
import cv2


# Set random seed for reproducibility
random.seed(42)
size=(643, 438)
test_set_size=10            #in prozent
# Paths
script_location = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_location, "retinal-lesions-v20191227", "retinal-lesions-v20191227")

# Input folders
img_train_dir = os.path.join(data_root, "images_896x896")
gt_train_root = os.path.join(data_root, "lesion_segs_896x896")

dataset_name="Dataset001_RetinalLesions"

# Output folders
output_base = f"nnUNet_raw/{dataset_name}"
imagesTr = os.path.join(output_base, "imagesTr")
imagesTs = os.path.join(output_base, "imagesTs")
labelsTr = os.path.join(output_base, "labelsTr")
labelsTs = os.path.join(output_base, "labelsTs")
os.makedirs(imagesTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(labelsTs, exist_ok=True)

# Priority-ordered class labels (lower wins, so from low priority to high priority)
# Microaneurysms > Hard Exudates > Soft Exudates > Haemorrhages
labels = [
    "cotton_wool_spots",
    "microaneurysm",
    "hard_exudate",
    "retinal_hemorrhage",
    "preretinal_hemorrhage",
    "vitreous_hemorrhage",
    "neovascularization",
    "fibrous_proliferation",
]

def center_crop_and_resize(image, target_size=(643, 438), interpolation=Image.BICUBIC):
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



def save_json(obj, filepath, sort_keys=True, indent=4):
    """
    Save a Python object as a JSON file.

    Parameters:
    - obj: The Python object to serialize (usually a dict).
    - filepath: Full path to the output JSON file.
    - sort_keys: Whether to sort keys alphabetically in output.
    - indent: Indentation level for pretty-printing (default: 4).
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          #citation: Union[List[str], str] = None,
                          #regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None,
                          reference: str = None,
                          release: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None,
                          license: str = 'Whoever converted this dataset was lazy and didn\'t look it up!',
                          converted_by: str = "Please enter your name, especially when sharing datasets with others in a common infrastructure!",
                          **kwargs):
    
    #has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    #if has_regions:
    #    assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
    #                                            f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
        'licence': license,
        'converted_by': converted_by
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    #if citation is not None:
    #    dataset_json['citation'] = release
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    #if regions_class_order is not None:
    #    dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, os.path.join(output_folder, 'dataset.json'), sort_keys=False)


# --- Helper Functions ---
def load_image(img_path):
    return Image.open(img_path).convert("RGB")

def save_image(img, path):
    img.save(path)

def load_mask(mask_path):
    return (np.array(Image.open(mask_path).convert("L"))).astype(np.uint8)

def combine_masks(base_path, id_str):
    combined = None
    for subfolder in order:
        #print(f"base_path: {base_path}")
        #print(f"subfolder: {subfolder}, label_value: {label_value}")
        mask_path = os.path.join(base_path, subfolder, f"IDRiD_{id_str}_{get_abbreviation(subfolder)}.tif")
        #print(f"mask_path: {mask_path}")
        if not os.path.exists(mask_path):
            continue
        mask = load_mask(mask_path)
        binary_mask = mask > 0
        if combined is None:
            combined = np.zeros_like(mask, dtype=np.uint8)
        combined[binary_mask] = 1
    return Image.fromarray(combined)







# --- Process Training Images ---
print("Processing training set...")

# Collect all filenames
train_filenames = [f for f in os.listdir(img_train_dir) if f.endswith(".jpg")]

# Extract unique id_strs
id_strs = set(fname.split("_")[0] for fname in train_filenames)
id_strs = sorted(id_strs)
test_ids = set(random.sample(id_strs, max(1, len(id_strs) // test_set_size)))  # 10% for test

counter=0
for i, fname in enumerate(train_filenames, 1):
    img = load_image(os.path.join(img_train_dir, fname))
    id_str = fname.split("_")[0]
    side_str = fname.split("_")[1].split(".")[0].lower()
    side = "0" if side_str == "left" else "1"

    if id_str in test_ids:
        img_dest = imagesTs
        mask_dest = labelsTs
    else:
        img_dest = imagesTr
        mask_dest = labelsTr
        counter+=1


    # Save mask
    mask = combine_masks(gt_train_root, id_str, side_str)
    mask = center_crop_and_resize(mask, size, interpolation=Image.NEAREST)
    #mask= center_crop_and_resize(mask, size)
    img= enhance_lesions(img, size)
    save_image(img, os.path.join(img_dest, f"RetinalLesions_{id_str}{side}_0000.png"))
    save_image(mask, os.path.join(mask_dest, f"RetinalLesions_{id_str}{side}.png"))

    if i % 50 == 0 or i == len(train_filenames):
        print(f"Processed {i}/{len(train_filenames)} training images")


generate_dataset_json(os.path.join(script_location,output_base), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'lesion': 1},
                          counter, '.png', dataset_name=dataset_name)