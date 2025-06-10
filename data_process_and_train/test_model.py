import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import seaborn as sns

# Load mô hình FaceNet
model = tf.keras.models.load_model('face_auth_model_facenet.h5', custom_objects={'Addons>TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss})
print("Đã tải mô hình thành công!")

# Load embeddings
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings_data = data['embeddings']  # list of (embedding, mssv)
mssv_list = data['mssv_list']         # danh sách mssv (nhãn)

# Tách embedding và nhãn
embeddings = np.array([item[0] for item in embeddings_data])
labels = np.array([item[1] for item in embeddings_data])

# Tạo tập train/test (leave-one-out)
y_true = []
y_pred = []

for i in range(len(embeddings)):
    test_emb = embeddings[i]
    true_label = labels[i]

    # Reference là toàn bộ trừ mẫu test hiện tại
    reference_embs = np.delete(embeddings, i, axis=0)
    reference_labels = np.delete(labels, i, axis=0)

    min_dist = float('inf')
    predicted_label = None
    for ref_emb, ref_label in zip(reference_embs, reference_labels):
        dist = cosine(test_emb, ref_emb)
        if dist < min_dist:
            min_dist = dist
            predicted_label = ref_label

    y_true.append(true_label)
    y_pred.append(predicted_label)

# Tính độ chính xác và F1-score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
print("=== Đánh giá mô hình FaceNet ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Báo cáo phân loại
print("\nBáo cáo phân loại:")
print(classification_report(y_true, y_pred))

# ---------------- Vẽ biểu đồ ----------------
unique_labels = np.unique(y_true)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Biểu đồ F1 theo lớp
precision, recall, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=unique_labels)
plt.figure(figsize=(14, 6))
plt.bar(unique_labels, f1_per_class, color='skyblue')
plt.xlabel('MSSV')
plt.ylabel('F1-score')
plt.title('F1-score theo MSSV')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('f1_per_class.png', dpi=300)
plt.show()