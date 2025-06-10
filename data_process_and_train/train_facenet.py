import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
import numpy as np
import os
import pickle
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'processed_data')
model_path = os.path.join(current_dir, 'face_auth_model_facenet.h5')
history_path = os.path.join(current_dir, 'history.pkl')

# Hàm tải dữ liệu
def load_data(data_dir):
    images = []
    labels = []
    mssv_list = []
    for label, mssv in enumerate(os.listdir(data_dir)):
        mssv_dir = os.path.join(data_dir, mssv)
        if os.path.isdir(mssv_dir):
            mssv_list.append(mssv)
            for filename in os.listdir(mssv_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(mssv_dir, filename)
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)
    if len(images) == 0:
        raise ValueError("Không tìm thấy ảnh trong dataset!")
    if len(images) != len(labels):
        raise ValueError("Số lượng ảnh và nhãn không khớp nhau!")
    return np.array(images), np.array(labels), mssv_list

# Tải dữ liệu
logging.info("Tải dữ liệu...")
images, labels, mssv_list = load_data(data_dir)
logging.info(f"Đã tải {len(images)} ảnh và {len(labels)} nhãn.")

# Xây dựng mô hình FaceNet
def create_facenet_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    return model

model = create_facenet_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=tfa.losses.TripletSemiHardLoss())

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,              
    width_shift_range=0.05,       
    height_shift_range=0.05,      
    horizontal_flip=True,         
    brightness_range=[1.0, 1.2],  
    zoom_range=[0.98, 1.02],      
    fill_mode='constant',         
    cval=0.9                
)

# # Early Stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True
# )

# Chia dữ liệu huấn luyện và kiểm tra
train_size = int(0.8 * len(images))
train_images, val_images = images[:train_size], images[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# Huấn luyện
logging.info("Bắt đầu huấn luyện...")
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=50,
    steps_per_epoch=len(train_images) // 32,
    validation_data=(val_images, val_labels),
    # callbacks=[early_stopping]
)

# Lưu mô hình
model.save(model_path)
logging.info(f"Mô hình đã được lưu vào: {model_path}")

# Lưu history
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
logging.info(f"Lịch sử huấn luyện đã được lưu vào: {history_path}")