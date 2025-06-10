import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
import pickle
import tensorflow_addons as tfa

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'face_auth_model_facenet.h5')
data_dir = "D:/DATN/FaceAuthProject/data_processing/processed_data"
output_embeddings_file = os.path.join(current_dir, 'embeddings.pkl')

# Load mô hình FaceNet
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Addons>TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss})
    logging.info("Đã tải mô hình FaceNet thành công!")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình FaceNet: {str(e)}")
    raise

# Tạo embeddings cho dataset
def create_embeddings(data_dir, model):
    embeddings = []
    mssv_list = []
    for mssv in os.listdir(data_dir):
        mssv_dir = os.path.join(data_dir, mssv)
        if os.path.isdir(mssv_dir):
            mssv_list.append(mssv)
            for filename in os.listdir(mssv_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(mssv_dir, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.warning(f"Không đọc được ảnh: {img_path}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    embedding = model.predict(img_array)[0]
                    embeddings.append((embedding, mssv))
    if not embeddings:
        raise ValueError("Không tạo được embeddings từ dataset!")
    return embeddings, mssv_list

# Tạo và lưu embeddings
def save_embeddings():
    logging.info("Tạo embeddings cho dataset...")
    embeddings, mssv_list = create_embeddings(data_dir, model)
    try:
        with open(output_embeddings_file, 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'mssv_list': mssv_list}, f)
        logging.info(f"Đã lưu embeddings vào {output_embeddings_file}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    logging.info("Bắt đầu tạo và lưu embeddings...")
    try:
        save_embeddings()
    except Exception as e:
        logging.error(f"Lỗi trong quá trình tạo embeddings: {str(e)}")
    logging.info("Hoàn tất tạo và lưu embeddings.")