import os
import numpy as np
import json
from PIL import Image, ImageOps
import random

# Set random seed for reproducibility
random.seed(42)
size=(643, 438)
test_set_size=10            #in prozent
# Paths
# Paths
script_location = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_location, "A. Segmentation", "A. Segmentation")

# Input folders
img_train_dir = os.path.join(data_root, "1. Original Images", "a. Training Set")
img_test_dir = os.path.join(data_root, "1. Original Images", "b. Testing Set")
gt_train_root = os.path.join(data_root, "2. All Segmentation Groundtruths", "a. Training Set")
gt_test_root = os.path.join(data_root, "2. All Segmentation Groundtruths", "b. Testing Set")

dataset_name="Dataset003_IDRiD"

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

order = [
    "2. Haemorrhages",
    "4. Soft Exudates",
    "3. Hard Exudates",
    "1. Microaneurysms",
]

def get_abbreviation(label: str):
    if label=="2. Haemorrhages":
        return "HE"
    elif label=="4. Soft Exudates":
        return "SE"
    elif label=="3. Hard Exudates":
        return "EX"
    elif label=="1. Microaneurysms":
        return "MA"

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
train_filenames = [f for f in os.listdir(img_train_dir) if f.endswith(".jpg")]
for i, fname in enumerate(train_filenames, 1):
    id_str = fname.split("_")[1].split(".")[0]

    # Save image
    img = load_image(os.path.join(img_train_dir, fname))
    img= center_crop_and_resize(img, size)
    save_image(img, os.path.join(imagesTr, f"IDRiD_{id_str}_0000.png"))

    # Save mask
    mask = combine_masks(gt_train_root, id_str)
    mask = center_crop_and_resize(mask, size, interpolation=Image.NEAREST)
    save_image(mask, os.path.join(labelsTr, f"IDRiD_{id_str}.png"))

    if i % 50 == 0 or i == len(train_filenames):
        print(f"Processed {i}/{len(train_filenames)} training images")

# --- Process Testing Images ---
print("Processing testing set...")
test_filenames = [f for f in os.listdir(img_test_dir) if f.endswith(".jpg")]
for i, fname in enumerate(test_filenames, 1):
    id_str = fname.split("_")[1].split(".")[0]

    img = load_image(os.path.join(img_test_dir, fname))
    img= center_crop_and_resize(img, size)
    save_image(img, os.path.join(imagesTs, f"IDRiD_{id_str}.png"))

    # Save mask
    mask = combine_masks(gt_test_root, id_str)
    mask = center_crop_and_resize(mask, size, interpolation=Image.NEAREST)
    save_image(mask, os.path.join(labelsTs, f"IDRiD_{id_str}.png"))

    if i % 50 == 0 or i == len(test_filenames):
        print(f"Processed {i}/{len(test_filenames)} testing images")

generate_dataset_json(os.path.join(script_location,output_base), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'lesions': 1},
                          len(train_filenames), '.png', dataset_name=dataset_name)