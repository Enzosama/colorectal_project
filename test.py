import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os

def load_images_and_labels(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".png")])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(".png")])
    
    images = []
    labels = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256)) / 255.0  # Chuẩn hóa ảnh
        
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (256, 256))
        label = (label > 0).astype(np.float32)  # Chuyển thành nhãn nhị phân
        
        images.append(image)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=-1)
    
    return images, labels

def evaluate(model_path, image_dir, mask_dir):
    # Load mô hình đã được train
    model = keras.models.load_model(model_path)
    
    # Đọc và tiền xử lý dữ liệu
    test_images, test_labels = load_images_and_labels(image_dir, mask_dir)
    
    loss, accuracy = model.evaluate(test_images, test_labels, batch_size=8, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

model_path = "/Volumes/Home/Desktop/ML/env/Model_ML/Model/Train_4/colorectal_segmentation_model (3).h5"
image_dir = "/Volumes/Home/Desktop/ML/env/Model_ML/EBHI-SEG/Polyp/image"
mask_dir = "/Volumes/Home/Desktop/ML/env/Model_ML/EBHI-SEG/Polyp/label"

evaluate(model_path, image_dir, mask_dir)
