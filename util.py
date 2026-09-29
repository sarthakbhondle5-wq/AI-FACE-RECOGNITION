import os
import json
import numpy as np
import cv2
import torch

def ensure_dir(path):
    """
    Tao thu muc neu chua ton tai.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fname):
    """
    Doc file cau hinh json.
    """
    with open(fname, 'r') as handle:
        return json.load(handle)

def write_json(content, fname):
    """
    Ghi cau hinh ra file json.
    """
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4)

def calculate_l2_distance(emb1, emb2):
    """
    Tinh khoang cach L2 (Euclidean) giua hai vector embedding.
    """
    if isinstance(emb1, torch.Tensor):
        return torch.dist(emb1, emb2).item()
    return np.linalg.norm(emb1 - emb2)

def calculate_cosine_similarity(emb1, emb2):
    """
    Tinh do tuong dong cosine giua hai vector embedding.
    """
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
        emb2 = emb2.detach().cpu().numpy()
    
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    return dot_product / (norm_a * norm_b + 1e-8)

def generate_synthetic_faces(data_dir, num_classes=10, images_per_class=15, img_size=(250, 250)):
    """
    Tao anh gia lap trong cac thu muc de test khi khong co dataset that.
    Bat chuoc dung cau truc cua CASIA-WebFace.
    """
    target_dir = os.path.join(data_dir, "CASIA-WebFace")
    
    # kiem tra xem da co du class chua
    if os.path.exists(target_dir):
        existing_subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        if len(existing_subdirs) >= num_classes:
            print(f"[Dataset] Da co san dataset tai '{target_dir}' voi {len(existing_subdirs)} classes.")
            return target_dir

    print(f"[Dataset] Dang tao dataset gia lap tai '{target_dir}' ({num_classes} classes, {images_per_class} anh/class)...")
    os.makedirs(target_dir, exist_ok=True)

    for i in range(num_classes):
        class_name = f"id_{i:04d}"
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # random mau da va dac trung khuon mat co dinh cho moi class
        skin_color = (
            np.random.randint(160, 240),
            np.random.randint(180, 245),
            np.random.randint(220, 255)
        )
        eye_y = np.random.randint(85, 110)
        eye_dist = np.random.randint(35, 50)
        mouth_y = np.random.randint(155, 185)
        mouth_w = np.random.randint(35, 60)
        smile = np.random.choice([True, False])

        for j in range(images_per_class):
            # tao background ngau nhien
            bg_color = (np.random.randint(40, 120), np.random.randint(40, 120), np.random.randint(40, 120))
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            img[:, :] = bg_color

            # them bien doi nho ve pose cho tung anh
            dx = np.random.randint(-8, 9)
            dy = np.random.randint(-8, 9)
            r_var = np.random.randint(-6, 7)
            
            center = (img_size[1] // 2 + dx, img_size[0] // 2 + dy)
            radius = 80 + r_var
            
            # ve mat
            cv2.circle(img, center, radius, skin_color, -1)
            cv2.circle(img, center, radius, (0, 0, 0), 2)

            # ve mat (nen trang + mat den)
            left_eye_center = (center[0] - eye_dist, eye_y + dy)
            right_eye_center = (center[0] + eye_dist, eye_y + dy)
            cv2.circle(img, left_eye_center, 14, (255, 255, 255), -1)
            cv2.circle(img, right_eye_center, 14, (255, 255, 255), -1)
            cv2.circle(img, left_eye_center, 14, (0, 0, 0), 1)
            cv2.circle(img, right_eye_center, 14, (0, 0, 0), 1)

            # huong nhin ngau nhien
            gaze_x = np.random.randint(-4, 5)
            gaze_y = np.random.randint(-4, 5)
            cv2.circle(img, (left_eye_center[0] + gaze_x, left_eye_center[1] + gaze_y), 5, (0, 0, 0), -1)
            cv2.circle(img, (right_eye_center[0] + gaze_x, right_eye_center[1] + gaze_y), 5, (0, 0, 0), -1)

            # ve mieng
            mouth_center_y = mouth_y + dy
            if smile:
                cv2.ellipse(img, (center[0], mouth_center_y), (mouth_w, 20), 0, 0, 180, (0, 0, 220), 3)
            else:
                cv2.line(img, (center[0] - mouth_w, mouth_center_y), (center[0] + mouth_w, mouth_center_y), (0, 0, 220), 3)

            # noise nhe
            noise = np.random.normal(0, 4, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # luu anh
            img_path = os.path.join(class_dir, f"img_{j:03d}.jpg")
            cv2.imwrite(img_path, img)

    print(f"[Dataset] Da sinh xong {num_classes * images_per_class} anh gia lap.")
    return target_dir
