> .:
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# -----------------------------
# Dataset
# -----------------------------
class DesertDataset(Dataset):
    def init(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if not f.startswith(".")]

        self.image_transform = T.ToTensor()
        self.mask_transform = T.ToTensor()

    def len(self):
        return len(self.images)

    def getitem(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0).float()  # Binary mask

        return image, mask


# -----------------------------
# Model
# -----------------------------
def get_model():
    model = models.segmentation.fcn_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model


# -----------------------------
# IoU Metric
# -----------------------------
def iou_score(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    if union == 0:
        return torch.tensor(1.0)

    return intersection / union


# -----------------------------
# Training Function
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_dir = os.getcwd()

    train_image_path = os.path.join(
        base_dir, "Offroad_Segmentation_Training_Dataset", "train", "Color_Images"
    )
    train_mask_path = os.path.join(
        base_dir, "Offroad_Segmentation_Training_Dataset", "train", "Segmentation"
    )

    val_image_path = os.path.join(
        base_dir, "Offroad_Segmentation_Training_Dataset", "val", "Color_Images"
    )
    val_mask_path = os.path.join(
        base_dir, "Offroad_Segmentation_Training_Dataset", "val", "Segmentation"
    )

    train_dataset = DesertDataset(train_image_path, train_mask_path)
    val_dataset = DesertDataset(val_image_path, val_mask_path)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]
                val_iou += iou_score(outputs, masks).item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {train_loss/len(train_loader):.4f} "
            f"Val IoU: {val_iou/len(val_loader):.4f}"
        )

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if name == "main":
    train()
