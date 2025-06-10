import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'processed_data')
student_mssv = '63131330'
student_dir = os.path.join(data_dir, student_mssv)

# Hàm tải một ảnh mẫu của sinh viên cụ thể
def load_sample_image(student_dir):
    if not os.path.exists(student_dir):
        raise ValueError(f"Không tìm thấy thư mục của sinh viên {student_mssv} trong {data_dir}!")
    for filename in os.listdir(student_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(student_dir, filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            logging.info(f"Giá trị pixel ảnh gốc (min, max): {img_array.min()}, {img_array.max()}")
            return np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    raise ValueError(f"Không tìm thấy ảnh .jpg trong thư mục của sinh viên {student_mssv}!")

# Tải một ảnh mẫu của sinh viên 63131330
sample_image = load_sample_image(student_dir)
print(f"Kích thước ảnh mẫu của sinh viên {student_mssv}: {sample_image.shape}")

# Tạo bản sao để tăng cường, giữ nguyên ảnh gốc để hiển thị
sample_image_for_augmentation = np.copy(sample_image)
# Tăng độ sáng 400% cho bản sao dùng để tăng cường (theo code trước đó)
sample_image_for_augmentation = np.clip(sample_image_for_augmentation * 4.0, 0, 1)
logging.info(f"Giá trị pixel ảnh dùng để tăng cường sau khi tăng độ sáng 400% (min, max): {sample_image_for_augmentation.min()}, {sample_image_for_augmentation.max()}")

# Hàm hiển thị ảnh
def display_images(original_image, augmented_images, titles):
    plt.figure(figsize=(15, 5))
    # Ảnh gốc nguyên bản (giữ nguyên, không tăng độ sáng)
    plt.subplot(1, len(augmented_images) + 1, 1)
    plt.imshow(original_image[0])
    plt.title(f'Ảnh gốc')
    plt.axis('off')
    # Các ảnh tăng cường
    for i, (augmented_image, title) in enumerate(zip(augmented_images, titles)):
        plt.subplot(1, len(augmented_images) + 1, i + 2)
        plt.imshow(augmented_image)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Kiểm tra từng biến đổi và kết hợp
augmented_images = []
titles = []

# 1. Chỉ áp dụng brightness_range (chỉ tăng độ sáng)
logging.info("Kiểm tra với brightness_range (chỉ tăng độ sáng)...")
datagen_brightness = ImageDataGenerator(
    brightness_range=[1.0, 1.2],  # Tăng độ sáng tối đa lên 120%
    fill_mode='nearest'
)
augmented_generator = datagen_brightness.flow(sample_image_for_augmentation, batch_size=1)
# augmented_image_brightness = next(augmented_generator)[0]
# augmented_image_brightness = np.clip(augmented_image_brightness, 0, 1)
# logging.info(f"Giá trị pixel với brightness_range (min, max): {augmented_image_brightness.min()}, {augmented_image_brightness.max()}")
# augmented_images.append(augmented_image_brightness)
# titles.append('Chỉ brightness_range')

# 2. Tắt các biến đổi hình học để kiểm tra brightness_range
logging.info("Kiểm tra brightness_range mà không có biến đổi hình học...")
datagen_no_geometry = ImageDataGenerator(
    brightness_range=[1.0, 1.2],
    fill_mode='constant',
    cval=0.9  # Điền vùng trống bằng màu rất sáng
)
augmented_generator = datagen_no_geometry.flow(sample_image_for_augmentation, batch_size=1)
augmented_image_no_geometry = next(augmented_generator)[0]
augmented_image_no_geometry = np.clip(augmented_image_no_geometry, 0, 1)
logging.info(f"Giá trị pixel với brightness_range (không hình học, min, max): {augmented_image_no_geometry.min()}, {augmented_image_no_geometry.max()}")
augmented_images.append(augmented_image_no_geometry)
titles.append('Brightness_range')

# 3. Kết hợp tất cả biến đổi với phạm vi nhỏ
logging.info("Kết hợp tất cả biến đổi với phạm vi nhỏ...")
datagen_combined = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    brightness_range=[1.0, 1.2],
    zoom_range=[0.98, 1.02],
    fill_mode='constant',
    cval=0.9  # Điền vùng trống bằng màu rất sáng
)
augmented_generator = datagen_combined.flow(sample_image_for_augmentation, batch_size=1)

# Hiển thị 2 ảnh kết hợp
for i in range(2):
    augmented_image_combined = next(augmented_generator)[0]
    augmented_image_combined = np.clip(augmented_image_combined, 0, 1)
    logging.info(f"Giá trị pixel kết hợp {i+1} (min, max): {augmented_image_combined.min()}, {augmented_image_combined.max()}")
    augmented_images.append(augmented_image_combined)
    titles.append(f'Xoay ảnh #{i+1}')

# Hiển thị tất cả (giữ nguyên ảnh gốc)
display_images(sample_image, augmented_images, titles)