import os
import sys
from pathlib import Path

# append project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.util import generate_synthetic_faces

def main():
    # tao thu muc data neu chua ton tai
    os.makedirs("data", exist_ok=True)
    
    # tao dataset gia lap neu khong co dataset that
    # ham nay tu dong kiem tra neu da co du class thi se bo qua
    data_dir = generate_synthetic_faces("data", num_classes=10, images_per_class=15)
    print(f"[Data] Du lieu da duoc thiet lap san tai: {data_dir}")

if __name__ == '__main__':
    main()
