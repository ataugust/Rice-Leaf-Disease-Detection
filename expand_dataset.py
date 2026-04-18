import os
import random
from PIL import Image, ImageEnhance
import torchvision.transforms as T
from tqdm import tqdm

# Configuration
TARGET_PER_CLASS = 2000
DATASET_DIR = r"D:\RESNET MODEL\Rice_Leaf_AUG"

# Define offline augmentations
augmentations = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
])

def expand_class_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)
    
    if current_count >= TARGET_PER_CLASS:
        print(f"Skipping {os.path.basename(folder_path)}: Already has {current_count} images.")
        return

    deficit = TARGET_PER_CLASS - current_count
    print(f"Generating {deficit} images for {os.path.basename(folder_path)}...")

    # Load all existing images in memory for faster processing
    loaded_images = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            with Image.open(img_path) as img:
                loaded_images.append((img.copy(), img_name))
        except Exception as e:
            continue

    # Generate new images
    for i in tqdm(range(deficit), desc=os.path.basename(folder_path), leave=False):
        # Randomly select an existing image to augment
        base_img, base_name = random.choice(loaded_images)
        
        # Apply augmentation
        aug_img = augmentations(base_img)
        
        # Save with a new, unique filename
        new_filename = f"aug_{i}_{base_name}"
        aug_img.save(os.path.join(folder_path, new_filename))

def main():
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for cls in classes:
        folder_path = os.path.join(DATASET_DIR, cls)
        expand_class_folder(folder_path)
    print("Dataset expansion complete. You now have 12,000 images.")

if __name__ == "__main__":
    main()