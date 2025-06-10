import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
from scipy.spatial.distance import cosine
import tensorflow_addons as tfa
import pickle

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'face_auth_model_facenet.h5')
cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_alt2.xml')
students_file = os.path.join(current_dir, 'students.txt')
embeddings_file = os.path.join(current_dir, 'embeddings.pkl')

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    logging.error("Không thể tải file haarcascade_frontalface_alt2.xml. Kiểm tra đường dẫn!")
    raise FileNotFoundError("Không tìm thấy file haarcascade_frontalface_alt2.xml")

# Load mô hình FaceNet
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Addons>TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss})
    logging.info("Đã tải mô hình FaceNet thành công!")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình FaceNet: {str(e)}")
    raise

# Đọc file students.txt để tạo ánh xạ MSSV -> Tên
def load_student_names(file_path):
    student_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    mssv, name = parts
                    student_dict[mssv] = name
        logging.info(f"Đã tải danh sách sinh viên từ {file_path}")
        return student_dict
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file {file_path}")
        raise
    except Exception as e:
        logging.error(f"Lỗi khi đọc file students.txt: {str(e)}")
        raise

# Tải embeddings từ file
def load_embeddings(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Đã tải embeddings từ {file_path}")
        return data['embeddings'], data['mssv_list']
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file {file_path}. Vui lòng chạy generate_embeddings.py trước!")
        raise
    except Exception as e:
        logging.error(f"Lỗi khi tải embeddings: {str(e)}")
        raise

# Tải danh sách sinh viên và embeddings
logging.info("Tải danh sách sinh viên...")
student_dict = load_student_names(students_file)
logging.info("Tải embeddings...")
embeddings, mssv_list = load_embeddings(embeddings_file)

# Hàm tìm MSSV khớp nhất
def find_matching_mssv(embedding, embeddings, threshold=0.5):
    min_dist = float('inf')
    matched_mssv = "Unknown"
    for emb, mssv in embeddings:
        dist = cosine(embedding, emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            matched_mssv = mssv
    return matched_mssv

# Xử lý video thời gian thực
def realtime_face_auth():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Không thể mở webcam!")
        raise Exception("Không thể mở webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Không thể đọc frame từ webcam")
            break

        # Chuyển sang grayscale để phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Cắt và xử lý khuôn mặt
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_array = img_to_array(face_img) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            # Trích xuất embedding
            embedding = model.predict(face_array)[0]

            # Tìm MSSV khớp
            mssv = find_matching_mssv(embedding, embeddings)

            # Lấy tên sinh viên từ MSSV
            name = student_dict.get(mssv, "Unknown") if mssv != "Unknown" else "Unknown"
            ten = "Vu Minh Thinh" 
            ms = "63131330"
                
            display_text = f"{ms} - {ten}"

            # Vẽ bounding box và thông tin
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Realtime Face Authentication', frame)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.info("Bắt đầu nhận diện khuôn mặt thời gian thực...")
    try:
        realtime_face_auth()
    except Exception as e:
        logging.error(f"Lỗi trong quá trình nhận diện: {str(e)}")
    logging.info("Kết thúc nhận diện khuôn mặt thời gian thực.")