import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import tkinter as tk
from tkinter import filedialog

# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "resnet_rice_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. LOAD MODEL & CLASSES
# ==========================================
print(f"Using device: {DEVICE}")
print("Loading model weights...")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['classes']
    num_classes = len(class_names)
except FileNotFoundError:
    print(f"Error: Could not find {CHECKPOINT_PATH}. Make sure it is in the same folder.")
    sys.exit()

# Rebuild the ResNet-50 architecture
model = models.resnet50(weights=None) 
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, num_classes)

# Inject your trained weights
model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval() # Set model to evaluation/testing mode

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================
test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. RUN INFERENCE
# ==========================================
def predict_image(image_path):
    try:
        # Open image and force convert to RGB
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Apply transforms and add a batch dimension
    input_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get the highest probability
    top_prob, top_class_idx = torch.max(probabilities, 0)
    
    # Print results
    print("\n" + "="*50)
    print(f" TESTING IMAGE: {image_path.split('/')[-1]}")
    print("="*50)
    print(f"Predicted Disease : {class_names[top_class_idx]}")
    print(f"Confidence Score  : {top_prob.item() * 100:.2f}%\n")
    
    print("Other Probabilities:")
    for i, prob in enumerate(probabilities):
        if i != top_class_idx:
            print(f"- {class_names[i]}: {prob.item() * 100:.2f}%")
    print("="*50 + "\n")


# ==========================================
# 4. FILE SELECTOR & EXECUTION
# ==========================================
if __name__ == '__main__':
    # Initialize hidden tkinter window
    root = tk.Tk()
    root.withdraw() # Hides the small blank tkinter box

    print("Opening file selector...")
    
    # Open the native Windows file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("No file selected. Exiting...")
        sys.exit()
        
    predict_image(file_path)