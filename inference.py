import os
import argparse
import pickle
import shutil
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from model.model import FaceNetModel
from parse_config import ConfigParser
from utils.util import ensure_dir, calculate_l2_distance, calculate_cosine_similarity

class FaceDetector:
    """
    Wrapper cho detect khuon mat. Ho tro YOLOv8 va OpenCV Haar Cascades.
    """
    def __init__(self, mode="opencv"):
        self.mode = mode.lower()
        self.model_path = "yolov8n.pt"
        
        # kiem tra thu vien ultralytics de dung YOLOv8
        if self.mode == "yolov8":
            try:
                from ultralytics import YOLO
                import urllib.request
                
                self.model_path = "yolov8n-face.pt"
                if not os.path.exists(self.model_path):
                    print(f"[Detector] Dang tai weights YOLOv8-Face: '{self.model_path}'...")
                    url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
                    try:
                        urllib.request.urlretrieve(url, self.model_path)
                        print("[Detector] Tai xong.")
                    except Exception as e:
                        print(f"[Detector] Tai loi ({e}). Dung 'yolov8n.pt' mac dinh.")
                        self.model_path = "yolov8n.pt"
                        
                self.yolo_model = YOLO(self.model_path)
                print(f"[Detector] Da load YOLOv8 tu '{self.model_path}'.")
            except ImportError:
                print("[Detector] Khong tim thay 'ultralytics'. Chuyen ve dung OpenCV Haar Cascade.")
                self.mode = "opencv"
                
        if self.mode == "opencv":
            frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
            self.profile_cascade = cv2.CascadeClassifier(profile_path)
            if self.frontal_cascade.empty():
                # fallback neu khong tim thay alt2
                frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
            print("[Detector] Da load OpenCV Frontal & Profile Face Cascade.")

    def detect(self, img_bgr):
        # detect mat trong anh BGR, tra ve box: [xmin, ymin, xmax, ymax]
        boxes = []
        h_img, w_img = img_bgr.shape[:2]
        img_area = h_img * w_img
        
        if self.mode == "yolov8":
            results = self.yolo_model(img_bgr, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf > 0.4:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        # neu dung yolov8n.pt thi lay 25% phia tren cua body de lam face box
                        if "yolov8n-face.pt" not in self.model_path:
                            if cls == 0:  # class person
                                w = xyxy[2] - xyxy[0]
                                h = xyxy[3] - xyxy[1]
                                face_xyxy = [
                                    int(xyxy[0] + 0.1 * w),
                                    int(xyxy[1]),
                                    int(xyxy[2] - 0.1 * w),
                                    int(xyxy[1] + 0.25 * h)
                                ]
                                boxes.append(face_xyxy)
                        else:
                            # neu dung yolov8n-face.pt thi lay thang xyxy
                            boxes.append(xyxy.tolist())
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # 1. Thu frontal face cascade
            faces = self.frontal_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=5, 
                minSize=(40, 40)
            )
            for (x, y, w, h) in faces:
                box_area = w * h
                # Neu anh dau vao la anh nho da crop san (< 350x350)
                # ma phat hien ra box qua be (< 15% dien tich anh) thi bo qua
                if w_img < 350 and h_img < 350 and (box_area < 0.15 * img_area):
                    continue
                boxes.append([x, y, x + w, y + h])
                
            # 2. Neu khong tim thay, thu profile face cascade
            if len(boxes) == 0:
                p_faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=5,
                    minSize=(40, 40)
                )
                for (x, y, w, h) in p_faces:
                    box_area = w * h
                    if w_img < 350 and h_img < 350 and (box_area < 0.15 * img_area):
                        continue
                    boxes.append([x, y, x + w, y + h])
        return boxes


class VectorDatabase:
    """
    Database vector luu tru face embedding.
    Dung de luu/load cac user qua file pickle va so khop vector.
    """
    def __init__(self, db_path="saved/models/face_vector_db.pkl"):
        self.db_path = db_path
        self.database = {}
        self.load()

    def register(self, name, embedding):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy().flatten()
        self.database[name] = embedding.flatten()

    def search(self, query_emb, threshold=0.6, metric="l2"):
        # tim kiem identity gan nhat, tra ve (name, similarity, distance)
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.detach().cpu().numpy().flatten()
        query_emb = query_emb.flatten()
        
        if not self.database:
            return "Unknown", 0.0, 999.0

        best_name = "Unknown"
        best_dist = 999.0
        best_sim = -1.0

        for name, db_emb in self.database.items():
            dist = calculate_l2_distance(query_emb, db_emb)
            sim = calculate_cosine_similarity(query_emb, db_emb)
            
            if metric == "l2":
                if dist < best_dist:
                    best_dist = dist
                    best_sim = sim
                    best_name = name
            else:
                if sim > best_sim:
                    best_dist = dist
                    best_sim = sim
                    best_name = name

        # kiem tra nguong dat quyet dinh (decision threshold)
        if metric == "l2" and best_dist < threshold:
            return best_name, best_sim, best_dist
        elif metric == "cosine" and best_sim > threshold:
            return best_name, best_sim, best_dist
            
        return "Unknown", best_sim, best_dist

    def save(self):
        ensure_dir(os.path.dirname(self.db_path))
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        print(f"[VectorDB] Da luu database vao '{self.db_path}' ({len(self.database)} users).")

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"[VectorDB] Da load database tu '{self.db_path}' voi {len(self.database)} identities.")
        else:
            print("[VectorDB] Khong tim thay file database. Khoi tao db trong.")


def build_database_from_dataset(model, db, dataset_path, img_size=160, device="cpu"):
    # Quet thu muc data, dang ky anh dau tien cua moi identity vao database
    print(f"[VectorDB] Dang build face database tu thu muc '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"[VectorDB] Loi: khong tim thay duong dan '{dataset_path}'.")
        return

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    identities = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for identity in identities:
        identity_dir = os.path.join(dataset_path, identity)
        img_names = [f for f in os.listdir(identity_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_names:
            continue
            
        # lay anh dau tien lam anh dai dien
        reg_img_path = os.path.join(identity_dir, img_names[0])
        img = Image.open(reg_img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model(img_tensor).cpu().numpy().flatten()
            
        db.register(identity, emb)
        
    db.save()


def run_pipeline(config, checkpoint_path, query_img_path, detector_mode="opencv", metric="l2", threshold=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Dang chay tren device: {device}")

    # 1. Load model tu checkpoint
    try:
        model = FaceNetModel.load_from_checkpoint(checkpoint_path)
        model.eval()
    except Exception as e:
        print(f"[Inference] Warning: Khong load duoc checkpoint ({e}). Khoi tao ngau nhien.")
        model_config = {
            "embedding_dim": config.get("embedding_dim", 128),
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": config.get("weight_decay", 0.0005),
            "num_classes": config.get("num_classes", 10),
            "loss": config.get("loss", {}),
        }
        model = FaceNetModel(model_config)
        model.eval()
    model.to(device)

    # 2. Setup Vector Database
    db_file = os.path.join(config.get("save_dir", "saved/models"), "face_vector_db.pkl")
    db = VectorDatabase(db_path=db_file)
    
    dataset_path = os.path.join(config.get("data_loader", {}).get("data_dir", "data"), "CASIA-WebFace")
    
    # tu dong rebuild database neu checkpoint moi hon file database (da duoc train lai)
    need_rebuild = False
    if not db.database:
        need_rebuild = True
    elif checkpoint_path and os.path.exists(checkpoint_path) and os.path.exists(db_file):
        checkpoint_time = os.path.getmtime(checkpoint_path)
        db_time = os.path.getmtime(db_file)
        if checkpoint_time > db_time:
            print("[Inference] Phat hien checkpoint moi hon Vector Database. Tien hanh build lai database...")
            need_rebuild = True
            db.database = {}  # xoa dbs cu truoc khi build lai
            
    if need_rebuild:
        build_database_from_dataset(model, db, dataset_path, img_size=160, device=device)

    # 3. Khoi tao Face Detector
    detector = FaceDetector(mode=detector_mode)

    # 4. Doc anh query
    print(f"[Inference] Doc anh query: '{query_img_path}'")
    if not os.path.exists(query_img_path):
        # tu dong tao anh query gia lap de test neu khong tim thay
        print(f"[Inference] Khong thay anh query. Dang tao anh gia lap tai '{query_img_path}'...")
        ensure_dir(os.path.dirname(query_img_path))
        
        if os.path.exists(dataset_path):
            first_class = sorted(os.listdir(dataset_path))[0]
            first_class_dir = os.path.join(dataset_path, first_class)
            first_class_imgs = sorted(os.listdir(first_class_dir))
            # lay anh thu 2 de so khop thu nghiem
            if len(first_class_imgs) > 1:
                source_img = os.path.join(first_class_dir, first_class_imgs[1])
                shutil.copy(source_img, query_img_path)
                print(f"[Inference] Da copy anh class '{first_class}' de lam query.")
        
        if not os.path.exists(query_img_path):
            # fallback ve anh ve tay
            img = np.ones((400, 400, 3), dtype=np.uint8) * 100
            cv2.circle(img, (200, 200), 100, (190, 220, 240), -1)
            cv2.circle(img, (200, 200), 100, (0, 0, 0), 2)
            cv2.circle(img, (160, 180), 15, (255, 255, 255), -1)
            cv2.circle(img, (240, 180), 15, (255, 255, 255), -1)
            cv2.circle(img, (160, 180), 5, (0, 0, 0), -1)
            cv2.circle(img, (240, 180), 5, (0, 0, 0), -1)
            cv2.ellipse(img, (200, 250), (40, 15), 0, 0, 180, (0, 0, 220), 3)
            cv2.imwrite(query_img_path, img)

    img_bgr = cv2.imread(query_img_path)
    if img_bgr is None:
        raise ValueError(f"Khong load duoc anh '{query_img_path}'")
        
    img_display = img_bgr.copy()

    # 5. Detect mat
    boxes = detector.detect(img_bgr)
    print(f"[Inference] Tim thay {len(boxes)} mat.")
    if len(boxes) == 0:
        print("[Inference] Khong detect duoc mat. Chuyen ve che do treat ca anh lam mat.")
        boxes = [[0, 0, img_bgr.shape[1], img_bgr.shape[0]]]

    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 6. Crop mat va so khop
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_bgr.shape[1], xmax)
        ymax = min(img_bgr.shape[0], ymax)
        
        face_crop = img_bgr[ymin:ymax, xmin:xmax]
        if face_crop.size == 0:
            continue
            
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # trich xuat embedding
        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy().flatten()

        # tim kiem trong database
        name, similarity, distance = db.search(embedding, threshold=threshold, metric=metric)
        print(f" -> Tim thay khop: {name} (Khoang cach L2: {distance:.3f}, Cos Sim: {similarity:.3f})")

        # ve box va label len anh
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img_display, (xmin, ymin), (xmax, ymax), color, 3)
        
        label = f"{name} ({distance:.2f})" if metric == "l2" else f"{name} ({similarity:.2f})"
        cv2.putText(
            img_display, 
            label, 
            (xmin, max(15, ymin - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )

    # luu anh ket qua
    out_dir = os.path.join(config.get("save_dir", "saved/models"), "predictions")
    ensure_dir(out_dir)
    save_path = os.path.join(out_dir, "inference_result.png")
    cv2.imwrite(save_path, img_display)
    
    print("\n" + "="*50)
    print("KET QUA INFERENCE:")
    print(f" - Anh ket qua annotated da luu tai: '{save_path}'")
    print("="*50 + "\n")
    
    # Tu dong mo anh ket qua tren Windows
    try:
        os.startfile(os.path.abspath(save_path))
        print("[Inference] Da tu dong mo anh ket qua.")
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition Inference')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json)')
    parser.add_argument('-cp', '--checkpoint', default=None, type=str,
                        help='trained model checkpoint (.ckpt)')
    parser.add_argument('-q', '--query', default='data/query_face.jpg', type=str,
                        help='query image path')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='override device')
    parser.add_argument('--detector', default=None, type=str, choices=['opencv', 'yolov8'],
                        help='face detector type (opencv or yolov8)')
    parser.add_argument('--metric', default=None, type=str, choices=['l2', 'cosine'],
                        help='matching distance metric (l2 or cosine)')
    parser.add_argument('--threshold', default=None, type=float,
                        help='decision threshold')
                        
    # argument tam thoi de tuong thich ConfigParser
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-e', '--epochs', default=None, type=int)
    parser.add_argument('-b', '--batch_size', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--run_id', default=None, type=str)

    args = parser.parse_args()
    config = ConfigParser.from_args(parser, args)

    checkpoint = args.checkpoint or args.resume
    if checkpoint is None:
        model_name = config.get("name", "FaceRecognition_Project")
        parent_dir = os.path.join(config.get("save_dir", "saved/models"), model_name, "latest")
        if os.path.exists(parent_dir):
            ckpts = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.ckpt')]
            if ckpts:
                checkpoint = max(ckpts, key=os.path.getmtime)
                
    # doc cau hinh tu config hoac cli
    detector_mode = args.detector if args.detector is not None else config.get("detector", "opencv")
    metric_mode = args.metric if args.metric is not None else config.get("metric", "l2")
    threshold_val = args.threshold if args.threshold is not None else config.get("threshold", 0.6)

    run_pipeline(
        config, 
        checkpoint, 
        args.query, 
        detector_mode=detector_mode, 
        metric=metric_mode, 
        threshold=threshold_val
    )
