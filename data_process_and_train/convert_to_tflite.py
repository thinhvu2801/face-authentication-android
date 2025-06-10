import tensorflow as tf
import tensorflow_addons as tfa
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'face_auth_model_facenet.h5')
output_tflite_path = os.path.join(current_dir, 'face_auth_model_facenet.tflite')

# Hàm chuyển đổi mô hình sang TFLite
def convert_to_tflite():
    try:
        # Tải mô hình Keras
        logging.info("Tải mô hình Keras...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'Addons>TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss}
        )
        
        # Tạo converter
        logging.info("Chuyển đổi mô hình sang TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Tối ưu hóa (quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Chuyển đổi mô hình
        tflite_model = converter.convert()
        
        # Lưu mô hình TFLite
        logging.info(f"Lưu mô hình TFLite vào {output_tflite_path}...")
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logging.info("Chuyển đổi và lưu mô hình TFLite thành công!")
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình chuyển đổi: {str(e)}")
        raise

if __name__ == "__main__":
    logging.info("Bắt đầu chuyển đổi mô hình sang TFLite...")
    try:
        convert_to_tflite()
    except Exception as e:
        logging.error(f"Lỗi: {str(e)}")
    logging.info("Kết thúc quá trình chuyển đổi.")