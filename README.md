# Semantic_segmentation
# 🏜️ Offroad Semantic Segmentation

A deep learning pipeline for **binary semantic segmentation** of offroad/desert terrain images. The project uses a pretrained **FCN-ResNet50** backbone fine-tuned to classify each pixel as either traversable terrain or background — useful for autonomous navigation in unstructured outdoor environments.

---

## 📁 Project Structure

```
offroad-segmentation/
│
├── train.py                          # Model training script
├── test.py                           # Model evaluation/inference script
├── semantic_segmentation.ipynb       # Development notebook (exploratory)
│
└── Offroad_Segmentation_Training_Dataset/
    ├── train/
    │   ├── images/                   # Training RGB images (960x540)
    │   └── masks/                    # Corresponding binary segmentation masks
    └── val/
        ├── images/                   # Validation RGB images
        └── masks/                    # Corresponding binary segmentation masks
```

---

## 🧠 Model Architecture

The model is built on **FCN-ResNet50** from `torchvision.models.segmentation`, pretrained on COCO. The final classifier head is replaced with a single-channel `Conv2d` output layer to support **binary segmentation** (1 class: traversable terrain vs. background).

```python
model = models.segmentation.fcn_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
```

The model outputs raw **logits** (one channel per pixel). A sigmoid activation followed by a 0.5 threshold is applied during evaluation to produce the final binary mask.

---

## 📦 Dataset

- **Source:** `Offroad_Segmentation_Training_Dataset/`
- **Train set:** 2,859 image-mask pairs
- **Validation set:** 317 image-mask pairs
- **Image resolution:** 960 × 540 pixels (RGB)
- **Masks:** Grayscale images, binarized to 0 (background) or 1 (terrain) during loading

The dataset follows a simple folder convention: each image in `images/` has a corresponding mask of the **same filename** in `masks/`.

---

## 🗂️ Dataset Class — `DesertDataset`

Defined in both scripts, `DesertDataset` is a PyTorch `Dataset` subclass that:

1. Scans the image directory for all non-hidden files
2. Loads each image as RGB using `PIL`
3. Loads the corresponding mask as grayscale (`"L"` mode)
4. Converts both to tensors via `torchvision.transforms.ToTensor()`
5. Binarizes the mask: any non-zero pixel becomes `1.0`, all others become `0.0`

```python
class DesertDataset(Dataset):
    def __getitem__(self, idx):
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        image = self.image_transform(image)       # → FloatTensor [3, H, W]
        mask  = self.mask_transform(mask)         # → FloatTensor [1, H, W]
        mask  = (mask > 0).float()                # Binarize
        return image, mask
```

---

## 📏 Evaluation Metric — IoU (Intersection over Union)

The primary evaluation metric is **IoU (Jaccard Index)**, computed per batch and averaged over the validation loader.

```python
def iou_score(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)          # Logits → probabilities
    preds = (preds > threshold).float()  # Probabilities → binary mask
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    return intersection / union           # Returns 1.0 if both are empty
```

- **Input:** raw model logits and ground-truth binary masks
- **Threshold:** 0.5 (configurable)
- **Edge case:** returns `1.0` when both prediction and ground truth are fully empty (no positive pixels)

---

## 🚀 Usage

### Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision pillow segmentation-models-pytorch
```

### Training

```bash
python train.py
```

`train.py` handles:
- Building the `DesertDataset` for the training split
- Creating a `DataLoader` with `batch_size=4` and `shuffle=True`
- Initializing the FCN-ResNet50 model with the modified single-channel head
- Running the training loop with loss computation (Binary Cross-Entropy with logits)
- Saving model checkpoints

### Testing / Evaluation

```bash
python test.py
```

`test.py` handles:
- Loading the validation split via `DesertDataset`
- Running the model in `eval()` mode with `torch.no_grad()`
- Computing and printing the average **IoU score** across the validation set

```
Validation IoU: 0.XXXX
```

---

## ⚙️ Configuration

The key parameters to configure at the top of `train.py` and `test.py`:

| Parameter | Description | Default |
|---|---|---|
| `train_image_path` | Path to training images folder | `Offroad_Segmentation_Training_Dataset/train/images` |
| `train_mask_path` | Path to training masks folder | `Offroad_Segmentation_Training_Dataset/train/masks` |
| `val_image_path` | Path to validation images folder | `Offroad_Segmentation_Training_Dataset/val/images` |
| `val_mask_path` | Path to validation masks folder | `Offroad_Segmentation_Training_Dataset/val/masks` |
| `batch_size` | Batch size for DataLoader | `4` |
| `device` | Compute device (`"cuda"` or `"cpu"`) | Auto-detected |
| `threshold` | IoU binarization threshold | `0.5` |

---

## 🔬 Technical Notes

**Why FCN-ResNet50?**
FCN (Fully Convolutional Network) with a ResNet50 backbone is a well-established baseline for semantic segmentation. Pretraining on COCO provides strong low-level feature representations that transfer well to terrain segmentation tasks.

**Why binary segmentation?**
The dataset contains a single class of interest (traversable terrain). Collapsing to a single output channel with BCE loss is more efficient and numerically stable than softmax over two channels.

**Output format:**
The model's `forward()` returns a dict — the segmentation logits are accessed via `outputs['out']`, which has shape `[B, 1, H, W]`.

**No resizing:**
Images are fed at their native 960×540 resolution. If GPU memory is a constraint, consider adding a `T.Resize()` transform in the dataset.

---

## 📊 Results

| Split | Size | IoU |
|---|---|---|
| Training | 2,859 | 0.86 |
| Validation | 317 | 0.73 |


---

## 🛠️ Dependencies

- `torch` + `torchvision` — model, transforms, data loading
- `Pillow` — image I/O
- `numpy` — numerical operations
- `matplotlib` — visualization (notebook)
- `segmentation-models-pytorch` — imported in notebook (optional alternative architectures)
- `opencv-python` (`cv2`) — imported in notebook (optional preprocessing)
