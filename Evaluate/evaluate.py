import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Error: Check paths. Image dir: {image_dir}, Mask dir: {mask_dir}")
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    return accuracy, precision, recall, f1_score, iou

def evaluate_model(model_path, image_dir, mask_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    total_accuracy, total_precision, total_recall, total_f1_score, total_iou = 0, 0, 0, 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.float(), labels.float()
            outputs = model(images)
            accuracy, precision, recall, f1_score, iou = calculate_metrics(outputs, labels)
            
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score
            total_iou += iou

    avg_accuracy = total_accuracy / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1_score = total_f1_score / num_batches
    avg_iou = total_iou / num_batches

    return {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1_score,
        "iou": avg_iou
    }
