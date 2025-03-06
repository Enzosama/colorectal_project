from Evaluate import evaluate_model

# Đường dẫn mô hình và dữ liệu
model_path = "/Volumes/Home/Desktop/ML/env/Model_ML/Model/Train_6/colorectal_segmentation_model (6).pth"
image_dir = "/Volumes/Home/Desktop/ML/env/Model_ML/EBHI-SEG/Polyp/image"
mask_dir = "/Volumes/Home/Desktop/ML/env/Model_ML/EBHI-SEG/Polyp/label"

# Chạy đánh giá
metrics = evaluate_model(model_path, image_dir, mask_dir)

# In kết quả
print("Evaluation Results:")
print(f"Average Accuracy: {metrics['accuracy']:.4f}")
print(f"Average Precision: {metrics['precision']:.4f}")
print(f"Average Recall: {metrics['recall']:.4f}")
print(f"Average F1 Score: {metrics['f1_score']:.4f}")
print(f"Average IoU: {metrics['iou']:.4f}")

