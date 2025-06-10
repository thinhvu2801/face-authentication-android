import pickle
import json
import os

# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_file = os.path.join(current_dir, 'embeddings.pkl')
students_file = os.path.join(current_dir, 'students.txt')
output_embeddings_json = os.path.join(current_dir, 'embeddings.json')
output_students_json = os.path.join(current_dir, 'students.json')

# Chuyển embeddings.pkl sang JSON
with open(embeddings_file, 'rb') as f:
    data = pickle.load(f)
embeddings = data['embeddings']  # Danh sách các tuple (embedding, mssv)
mssv_list = data['mssv_list']

# Chuyển embedding sang danh sách nếu cần
embeddings_json = []
for emb_tuple, mssv in zip(embeddings, mssv_list):
    emb = emb_tuple[0]  # Lấy embedding từ tuple
    # Nếu embedding là numpy array, chuyển sang list; nếu đã là list, giữ nguyên
    emb_list = emb.tolist() if hasattr(emb, 'tolist') else emb
    embeddings_json.append({'mssv': mssv, 'embedding': emb_list})

# Bọc danh sách trong key "array"
with open(output_embeddings_json, 'w') as f:
    json.dump({'array': embeddings_json}, f)

# Chuyển students.txt sang JSON
students = {}
with open(students_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            mssv, name = parts
            students[mssv] = name
with open(output_students_json, 'w') as f:
    json.dump(students, f)

print("Đã tạo embeddings.json và students.json")