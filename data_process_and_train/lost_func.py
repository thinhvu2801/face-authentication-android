import pickle
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(current_dir, 'history.pkl')
output_loss_plot = os.path.join(current_dir, 'loss_plot.png')

# Hàm tải history
def load_history(file_path):
    try:
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
        logging.info(f"Đã tải lịch sử huấn luyện từ {file_path}")
        return history
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file {file_path}. Vui lòng chạy lại train_facenet.py!")
        raise
    except Exception as e:
        logging.error(f"Lỗi khi tải history: {str(e)}")
        raise

# Hàm vẽ biểu đồ loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (TripletSemiHardLoss)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(output_loss_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Đã lưu biểu đồ loss vào {output_loss_plot}")

# Hàm phân tích bổ sung (giả định có dữ liệu khoảng cách embedding)
def plot_embedding_distance(history, output_path):
    # Giả định: Tính khoảng cách trung bình của embedding trên tập kiểm tra
    # (Cần dữ liệu bổ sung từ quá trình huấn luyện, hiện tại chỉ là placeholder)
    logging.warning("Biểu đồ khoảng cách embedding chưa được triển khai do thiếu dữ liệu!")
    # Nếu có dữ liệu, thêm code để vẽ (yêu cầu bạn cung cấp dữ liệu hoặc logic tính khoảng cách)

# Main
if __name__ == "__main__":
    logging.info("Bắt đầu phân tích và vẽ biểu đồ...")
    try:
        # Tải history
        history = load_history(history_path)
        
        # Vẽ biểu đồ loss
        plot_loss(history)
        
        # Biểu đồ bổ sung (nếu cần)
        plot_embedding_distance(history, os.path.join(current_dir, 'embedding_distance_plot.png'))
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình phân tích: {str(e)}")
    logging.info("Hoàn tất phân tích và vẽ biểu đồ!")