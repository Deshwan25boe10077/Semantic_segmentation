> .:
import os
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Model
# -----------------------------
def get_model():
    model = models.segmentation.fcn_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model


# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Test on Single Image
# -----------------------------
def predict_image(image_path):
    transform = T.ToTensor()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    pred_mask = pred.squeeze().cpu().numpy()

    # Show result
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

    plt.show()


# Example usage
if name == "main":
    test_image_path = "test_image.jpg"  # Change this
    predict_image(test_image_path)
