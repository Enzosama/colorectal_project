import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Đường dẫn đến mô hình đã huấn luyện
model_path = "/Volumes/Home/Desktop/ML/env/Model_ML/Model/Train_4/colorectal_segmentation_model (3).h5"
model = load_model(model_path)

# Đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Định nghĩa hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256)) / 255.0
    return image

# Đường dẫn dữ liệu đầu vào và đầu ra
input_folder = "/Volumes/Home/Desktop/ML/env/Model_ML/Kvasir-SEG/images"
output_folder = "/Volumes/Home/Desktop/ML/env/Model_ML/Check_model"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dự đoán và lưu ảnh có Polyp
threshold = 0.5  # Ngưỡng để xác định có Polyp hay không

for img_name in tqdm(os.listdir(input_folder), desc="Processing images"):
    img_path = os.path.join(input_folder, img_name)
    
    # Tiền xử lý ảnh
    img = preprocess_image(img_path)
    img_input = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Dự đoán
    pred_mask = model.predict(img_input)[0]  # Dự đoán mask
    pred_mask = (pred_mask > threshold).astype(np.uint8)  # Chuyển sang nhị phân

    # Nếu có Polyp (ít nhất một điểm được phát hiện)
    if np.any(pred_mask):
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, cv2.imread(img_path))  # Lưu ảnh gốc
        print(f"Detected Polyp: {img_name} -> Saved in Check_model")

print("Processing completed!")
