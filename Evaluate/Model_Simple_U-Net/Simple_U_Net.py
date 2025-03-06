import os
import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Định nghĩa lại kiến trúc Simple U-Net
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

# Load mô hình đã huấn luyện
model_path = "/Volumes/Home/Desktop/ML/env/Model_ML/Model/Train_6/colorectal_segmentation_model (6).pth"
model = SimpleUNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Định nghĩa dataset class để load ảnh test
class PolypDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image, img_path

# Load ảnh từ folder test
image_folder = "/Volumes/Home/Desktop/ML/env/Model_ML/Kvasir-SEG/images"
dataset = PolypDataset(image_folder)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Thư mục lưu ảnh có Polyp
output_folder = "/Volumes/Home/Desktop/ML/env/Model_ML/Check_model"
os.makedirs(output_folder, exist_ok=True)

# Chạy dự đoán
for image, img_path in data_loader:
    with torch.no_grad():
        output = model(image.float())
    
    mask = output.squeeze().numpy()
    mask = (mask > 0.4).astype(np.uint8) * 255  # Ngưỡng xác định vùng có polyp
    
    # Nếu có vùng Polyp thì lưu ảnh
    if mask.sum() > 0:
        img_name = os.path.basename(img_path[0])
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, mask)
        print(f"Saved: {save_path}")

# Đánh giá mô hình
json_path = "/Volumes/Home/Desktop/ML/env/Model_ML/Kvasir-SEG/kavsir_bboxes.json"
with open(json_path, 'r') as f:
    ground_truths = json.load(f)

def calculate_accuracy(output_folder, ground_truths):
    total_images = 0
    correct_predictions = 0
    
    for img_name in os.listdir(output_folder):
        img_id = os.path.splitext(img_name)[0]
        if img_id in ground_truths:
            total_images += 1
            mask_path = os.path.join(output_folder, img_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            bbox = ground_truths[img_id]['bbox']
            height, width = ground_truths[img_id]['height'], ground_truths[img_id]['width']
            mask = cv2.resize(mask, (width, height))
            
            predicted_polyp = (mask > 0).astype(np.uint8)
            
            print(f"Evaluating {img_id}: {len(bbox)} ground truth bboxes")
            
            for box in bbox:
                x_min, y_min, x_max, y_max = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                gt_mask = np.zeros((height, width), dtype=np.uint8)
                gt_mask[y_min:y_max, x_min:x_max] = 1
                
                intersection = np.logical_and(predicted_polyp, gt_mask).sum()
                union = np.logical_or(predicted_polyp, gt_mask).sum()
                iou = intersection / union if union > 0 else 0
                
                print(f"IoU for {img_id}: {iou:.4f}")
                
                if iou > 0.5:
                    correct_predictions += 1
                    break
    
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"Model Accuracy: {accuracy:.2%}")

calculate_accuracy(output_folder, ground_truths)
