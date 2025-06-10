import os
import cv2
import numpy as np
from PIL import Image
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
input_dir = "D:/DATN/FaceAuthProject/dataset/faces-63cntt-clc"
output_dir = "D:/DATN/FaceAuthProject/data_processing/processed_data"
cascade_path = "D:/DATN/FaceAuthProject/data_processing/haarcascade_frontalface_alt2.xml"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    logging.error("Không thể tải file haarcascade_frontalface_alt2.xml. Kiểm tra đường dẫn!")
    raise FileNotFoundError("Không tìm thấy file haarcascade_frontalface_alt2.xml")

def preprocess_image(img_path, output_path, target_size=(224, 224)):
    """Phát hiện khuôn mặt, cắt, resize, chuẩn hóa, lưu dưới dạng .jpg"""
    try:
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Không đọc được ảnh: {img_path}")
            return False

        # Chuyển sang grayscale để phát hiện khuôn mặt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            logging.warning(f"Không phát hiện khuôn mặt trong ảnh: {img_path}")
            return False

        # Lấy khuôn mặt đầu tiên 
        (x, y, w, h) = faces[0]
        # Cắt vùng khuôn mặt
        face_img = img[y:y+h, x:x+w]
        # Resize về 224x224
        face_img = cv2.resize(face_img, target_size)
        # Chuyển từ BGR sang RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Chuẩn hóa giá trị pixel (0-1)
        face_img = face_img / 255.0

        # Chuyển sang PIL để lưu dưới dạng .jpg
        img_pil = Image.fromarray((face_img * 255).astype(np.uint8))
        img_pil.save(output_path, "JPEG", quality=95)
        logging.info(f"Đã xử lý và lưu ảnh: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
        return False

def create_processed_dataset(input_dir, output_dir):
    """Xử lý toàn bộ dataset và lưu vào thư mục mới"""
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Tạo thư mục đầu ra: {output_dir}")

    # Duyệt qua các thư mục MSSV
    for mssv in os.listdir(input_dir):
        mssv_input_dir = os.path.join(input_dir, mssv)
        mssv_output_dir = os.path.join(output_dir, mssv)

        if os.path.isdir(mssv_input_dir):
            # Tạo thư mục MSSV trong thư mục đầu ra
            if not os.path.exists(mssv_output_dir):
                os.makedirs(mssv_output_dir)
                logging.info(f"Tạo thư mục MSSV: {mssv_output_dir}")

            # Xử lý từng ảnh trong thư mục MSSV
            for filename in os.listdir(mssv_input_dir):
                # Chỉ xử lý các định dạng ảnh
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_input_path = os.path.join(mssv_input_dir, filename)
                    # Tạo tên file đầu ra (chuyển thành .jpg)
                    img_output_path = os.path.join(mssv_output_dir, f"{os.path.splitext(filename)[0]}.jpg")
                    preprocess_image(img_input_path, img_output_path)

if __name__ == "__main__":
    logging.info("Bắt đầu tiền xử lý dataset...")
    create_processed_dataset(input_dir, output_dir)
    logging.info("Hoàn tất tiền xử lý dataset!")