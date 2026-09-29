import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2

def plot_embeddings(embeddings, labels, class_names, save_path):
    """
    Ve bieu do phan bo embedding su dung PCA 2D.
    Nham truc quan hoa muc do phan tach cua cac lop trong khong gian embedding.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Ap dung PCA don gian de giam xuong 2 chieu
    # Tru di trung binh
    x = embeddings - np.mean(embeddings, axis=0)
    # Tinh ma tran hiep bien
    cov = np.cov(x.T)
    # Tinh tri rieng va vector rieng
    evals, evecs = np.linalg.eigh(cov)
    # Sap xep giam dan
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # Project du lieu vao 2 thanh phan chinh dau tien
    pca_proj = np.dot(x, evecs[:, :2])

    # Ve bieu do
    plt.figure(figsize=(10, 8), dpi=100)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if label < len(class_names) else f"ID_{label}"
        plt.scatter(
            pca_proj[mask, 0], 
            pca_proj[mask, 1], 
            label=name, 
            color=colors(idx), 
            alpha=0.8, 
            edgecolors='k', 
            s=60
        )

    plt.title("Face Embedding Space (PCA Projection)", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Identities")
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Da luu bieu do embedding tai '{save_path}'")


def draw_match_result(query_img, db_img, distance, similarity, is_match, query_name, db_name, save_path):
    """
    Tao anh so sanh song song giua anh Query va anh trong Database.
    Hien thi thong tin khoang cach L2, do tuong dong Cosine va ket qua so khop.
    """
    # Chuyen doi sang numpy array tuong thich cv2 (BGR)
    if isinstance(query_img, Image.Image):
        query_img = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2BGR)
    if isinstance(db_img, Image.Image):
        db_img = cv2.cvtColor(np.array(db_img), cv2.COLOR_RGB2BGR)

    # Resize anh ve cung kich thuoc
    h, w = 200, 200
    query_img = cv2.resize(query_img, (w, h))
    db_img = cv2.resize(db_img, (w, h))

    # Tao canvas anh (chua them thong tin text)
    canvas_h = h + 100
    canvas_w = w * 2 + 60
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240  # light grey background

    # Copy anh len canvas
    canvas[20:20+h, 20:20+w] = query_img
    canvas[20:20+h, w+40:w+40+w] = db_img

    # Ve vien quanh anh
    cv2.rectangle(canvas, (19, 19), (20+w, 20+h), (0, 0, 0), 2)
    cv2.rectangle(canvas, (w+39, 19), (w+40+w, 20+h), (0, 0, 0), 2)

    # Ve text
    cv2.putText(canvas, "Query Image", (20, h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, f"Label: {query_name}", (20, h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.putText(canvas, "Matched DB", (w+40, h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, f"Label: {db_name}", (w+40, h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Banner ket qua so khop
    banner_color = (40, 180, 40) if is_match else (40, 40, 220)  # Green for Match, Red for No Match
    banner_text = "MATCHED" if is_match else "NO MATCH"
    
    cv2.rectangle(canvas, (canvas_w//2 - 70, h+50), (canvas_w//2 + 70, h+85), banner_color, -1)
    cv2.putText(canvas, banner_text, (canvas_w//2 - 50, h+73), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Ve cac metrics
    metrics_text = f"L2 Dist: {distance:.3f} | Cosine Sim: {similarity:.3f}"
    cv2.putText(canvas, metrics_text, (canvas_w//2 - 150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.imwrite(save_path, canvas)
