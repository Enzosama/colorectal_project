import os
import torch
import cv2
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Định nghĩa dataset class
class ColorectalSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for class_name in class_dirs:
            image_dir = os.path.join(root_dir, class_name, 'image')
            label_dir = os.path.join(root_dir, class_name, 'label')
            
            if os.path.exists(image_dir) and os.path.exists(label_dir):
                images = sorted(os.listdir(image_dir))
                labels = sorted(os.listdir(label_dir))
                
                for img_name in images:
                    if img_name in labels:
                        self.image_paths.append(os.path.join(image_dir, img_name))
                        self.label_paths.append(os.path.join(label_dir, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256))
        
        label = (label > 0).astype(np.float32)  # Chuyển label thành nhị phân
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return image, label

# Tạo dataset và chia thành tập train, val, test
root_dir = "/Volumes/Home/Desktop/ML/env/Model_ML/EBHI-SEG"
dataset = ColorectalSegmentationDataset(root_dir)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Định nghĩa mô hình đơn giản
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

# Khởi tạo mô hình, hàm mất mát và optimizer
model = SimpleUNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=40):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "colorectal_segmentation_model.pth")
    print("Model saved successfully!")
train_model(model, train_loader, criterion, optimizer)