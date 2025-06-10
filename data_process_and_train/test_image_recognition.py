import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = os.path.join(current_dir, 'test_images') 
MODEL_PATH = os.path.join(current_dir, 'face_auth_model_facenet.h5')
EMBEDDINGS_JSON = os.path.join(current_dir, 'embeddings.json')
STUDENTS_JSON = os.path.join(current_dir, 'students.json')
CASCADE_PATH = os.path.join(current_dir, 'haarcascade_frontalface_alt2.xml')

# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    logging.error("Không thể tải file haarcascade_frontalface_alt2.xml. Kiểm tra đường dẫn!")
    raise FileNotFoundError("Không tìm thấy file haarcascade_frontalface_alt2.xml")

# Hàm tiền xử lý ảnh (tương tự preprocess_image.py)
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Không đọc được ảnh: {img_path}")
            return None

        # Chuyển sang grayscale để phát hiện khuôn mặt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            logging.warning(f"Không phát hiện khuôn mặt trong ảnh: {img_path}")
            return None

        # Lấy khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, target_size)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255.0  # Chuẩn hóa [0, 1]

        # Thêm batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
        return None

# Hàm tạo embedding từ ảnh
def get_embedding(model, image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None
    embedding = model.predict(img)
    return embedding[0]  # Trả về embedding 128 chiều

# Hàm tính khoảng cách Euclidean
def euclidean_distance(emb1, emb2):
    return np.sqrt(np.sum((emb1 - emb2) ** 2))

# Hàm tính độ tương đồng cosine
def cosine_similarity_score(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

# Hàm nhận diện danh tính
def identify_person(model, image_path, reference_embeddings, students_info, threshold=0.7):
    test_embedding = get_embedding(model, image_path)
    if test_embedding is None:
        return None, None, None, None

    best_match_mssv = None
    best_score = -1  # Cosine similarity: cao hơn là tốt hơn
    best_distance = float('inf')  # Euclidean: nhỏ hơn là tốt hơn

    # So sánh với các embedding tham chiếu
    for ref_data in reference_embeddings['array']:
        mssv = ref_data['mssv']
        ref_embedding = np.array(ref_data['embedding'])
        score = cosine_similarity_score(test_embedding, ref_embedding)
        distance = euclidean_distance(test_embedding, ref_embedding)

        if score > best_score:
            best_score = score
            best_distance = distance
            best_match_mssv = mssv

    # Kiểm tra ngưỡng
    if best_score >= threshold:
        name = students_info.get(best_match_mssv, "Unknown")
        return best_match_mssv, name, best_score, best_distance
    else:
        return "Unknown", "Unknown", best_score, best_distance

# Tải mô hình FaceNet
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss}
    )
    logging.info(f"Đã tải mô hình từ: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình: {str(e)}")
    raise

# Tải dữ liệu tham chiếu
try:
    with open(EMBEDDINGS_JSON, 'r') as f:
        reference_embeddings = json.load(f)
    with open(STUDENTS_JSON, 'r') as f:
        students_info = json.load(f)
    logging.info("Đã tải embeddings.json và students.json")
except Exception as e:
    logging.error(f"Lỗi khi tải dữ liệu tham chiếu: {str(e)}")
    raise

# Kiểm tra ảnh trong thư mục test_images
results = []
for img_file in os.listdir(TEST_IMAGE_DIR):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(TEST_IMAGE_DIR, img_file)
        
        # Nhận diện danh tính
        mssv, name, cosine_score, euclidean_dist = identify_person(
            model, image_path, reference_embeddings, students_info, threshold=0.7
        )
        
        if mssv is None:
            logging.warning(f"Bỏ qua ảnh không hợp lệ: {img_file}")
            continue
        
        # Lưu kết quả
        results.append({
            'image': img_file,
            'mssv': mssv,
            'name': name,
            'cosine_score': cosine_score,
            'euclidean_distance': euclidean_dist
        })
        
        # Hiển thị ảnh và kết quả
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        plt.title(f"Image: {img_file}\nMSSV: {mssv}\nName: {name}\nCosine Score: {cosine_score:.4f}\nEuclidean Distance: {euclidean_dist:.4f}")
        plt.axis('off')
        plt.show()

# In kết quả tổng hợp
print("\nKết quả nhận diện:")
for result in results:
    print(f"Ảnh: {result['image']}")
    print(f"MSSV: {result['mssv']}")
    print(f"Tên: {result['name']}")
    print(f"Độ tương đồng cosine: {result['cosine_score']:.4f}")
    print(f"Khoảng cách Euclidean: {result['euclidean_distance']:.4f}")
    print("-" * 50)

# Lưu kết quả vào file CSV
import pandas as pd
df = pd.DataFrame(results)
# df.to_csv(os.path.join(current_dir, 'recognition_results.csv'), index=False)
# logging.info("Đã lưu kết quả vào recognition_results.csv")