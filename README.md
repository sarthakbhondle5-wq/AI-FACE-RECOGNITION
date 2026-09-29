# 7. Face Recognition System (Attendance & Security)

This project builds a complete Face Recognition System using **PyTorch Lightning**. The system combines the fast detection capabilities of **YOLOv8-Face** / **OpenCV Haar Cascade** and the powerful feature extraction capabilities of **FaceNet** optimized with **Triplet Loss**.

## 1. Problem Overview
Face recognition is a core task in computer vision, serving various attendance and security applications. The task includes three main challenges:
- **Detection**: Identifying the face location in complex scenes (Bounding Box).
- **Feature Extraction**: Mapping the face image into a 128-dimensional vector space representing biometric features.
- **Vector Matching**: Performing nearest neighbor search in a self-built vector database to output identity classification.

## 2. Dataset
This project uses the cropped **CASIA-WebFace dataset** from Kaggle.
- **Input**: Pre-cropped face images (size 250x250 pixels) of 1,252 subjects.
- **Output**: 128-dimensional biometric embedding vectors and their corresponding identity labels.
- **Preprocessing**:
    - Resize images to a fixed size of 160x160.
    - Normalize using ImageNet mean and standard deviation.
- **Data Augmentation**: Employs Random Rotation, Random Horizontal Flip, and Color Jitter. Automatically pairs inputs into $(Anchor, Positive, Negative)$ triplets for Triplet Loss training.
*Note: If the real dataset is not found at `data/CASIA-WebFace`, the system automatically generates synthetic images for testing.*

## 3. Models Used
The project implements and supports the following models:
1. **YOLOv8-Face (Primary Detector)**: A highly accurate and fast face detector based on the YOLOv8 architecture. [arXiv:2408.15857](https://arxiv.org/pdf/2408.15857.pdf)
2. **OpenCV Haar Cascade (Fallback/Baseline Detector)**: Integrates an upgraded dual-cascade detector combining **Frontal Face (alt2)** and **Profile Face** cascades. This combination significantly improves detection accuracy for tilted, rotated, or profile face angles. It runs extremely fast on CPU without requiring a GPU. [OpenCV Tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
3. **FaceNet ResNet-18 (Recognition Model)**: Employs a pre-trained ResNet-18 backbone to extract face features into a 128-dimensional space, optimized using Triplet Loss. [arXiv:1503.03832](https://arxiv.org/pdf/1503.03832.pdf)

## 4. Training Process
- **Framework**: Fully optimized with **PyTorch Lightning**, automatically managing execution on CPU or GPU.
- **Loss Function**: Uses a combined loss of **Triplet Loss** (margin = 0.5) and **CrossEntropyLoss** (Auxiliary Classifier) to accelerate convergence and clustering of feature embeddings.
- **Speed & Performance (RTX 3060 Optimized)**:
    - **Batch Size**: 128 (parallelized to achieve ~12 batches/sec on GPU, reducing steps per epoch to 713).
    - **Precision**: `16-mixed` (utilizes Mixed Precision AMP on Tensor Cores to double training speed and lower VRAM usage).
    - **Data Loading**: 4 parallel CPU DataLoader workers to prevent data loading bottlenecks.
- **Scheduler**: **ReduceLROnPlateau** automatically halves the learning rate when validation loss plateaus (patience = 5).
- **Early Stopping**: Halts training if validation loss does not improve for 80 epochs (patience = 80).
- **Strategy**: Train for 250+ epochs depending on hardware configuration to ensure complete model convergence.

## 5. Inference Pipeline
The end-to-end inference pipeline is structured as follows:
1. **Detection**: Run YOLOv8-Face or Haar Cascade to locate faces.
2. **Crop & Normalize**: Crop the face regions, resize to 160x160, and normalize color channels.
3. **Feature Extraction**: Feed the face crop to the FaceNet model to obtain a 128-dimensional embedding vector.
4. **Vector Database Matching**:
    - Compare the query vector with registered identities in `face_vector_db.pkl` using L2 distance or Cosine Similarity.
    - Compare the closest match against a decision threshold (threshold = 0.6 for L2). If it exceeds the threshold, label the subject as **"Unknown"**.
5. **Visualization**: Draw bounding boxes and identity labels on the original image.

## 6. Experimental Results Comparison

| System Pipeline (Detector + Recognizer) | Detection Rate | Triplet Matching Accuracy | Processing Speed (GPU FPS) | Avg Distance (Pos / Neg Pairs) |
|---|:---:|:---:|:---:|:---:|
| **OpenCV Haar Cascade + FaceNet (ResNet-18)** | **92.0%** | **95.16%** | **84.0 FPS** | **0.730 / 1.405** |
| YOLOv8-Face + FaceNet (ResNet-18) | 78.0% | 95.16% | 67.1 FPS | 0.730 / 1.405 |

*(You can view actual screenshots of the training process and test results for each version in the [images/](images/) directory).*

> **Conclusion & Analysis**: 
> * Both pipelines achieve a very high Triplet recognition accuracy (**95.16%**) thanks to the well-trained FaceNet (ResNet-18) model checkpoint.
> * The **OpenCV Haar Cascade** pipeline (combining frontal and profile cascades) achieves a detection rate of **92.0%** and an extremely fast processing speed (**84.0 FPS** on GPU).
> * The **YOLOv8** pipeline (using standard YOLOv8n fallback) achieves a detection rate of **78.0%** at a speed of **67.1 FPS**.

## 7. How to Run the Project

### Step 1: Environment Setup
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Download Dataset
```bash
python utils/download_data.py
```

### Step 3: Train FaceNet Model
```bash
python train.py --config config.json
```
*Note: Loss and accuracy metrics will update in real-time. You can start **Tensorboard** to visualize metrics using the following command:*
```bash
tensorboard --logdir saved/log
```

### Step 4: Model Evaluation
After training is complete, the best checkpoint is saved under:
`saved/models/LicensePlateRecognition/latest/` (Filename starts with `best-checkpoint-epoch=xxxx-val_loss=xxxx.ckpt`)

Run the test script to evaluate face detection and recognition performance:
```bash
python test.py --resume saved/models/LicensePlateRecognition/latest/best-checkpoint-epoch=xxxx-val_loss=xxxx.ckpt
```
Accuracy metrics (%) and a Markdown table will print in the Terminal. The system will also save visualization outputs to the directory `saved/models/predictions/`, including:
* **`detection_result.png`**: Face detection/localization output.
* **`match_positive_x.png`** & **`match_negative_x.png`**: Visualization of matching positive and negative face pairs.
* **`embedding_pca.png`**: PCA projection of face embedding vectors.

*Note: You can change the `"detector"` parameter in `config.json` (to `"yolov8"` or `"opencv"`) before running the test script. The script automatically logs metrics and outputs the corresponding Markdown row to help update the Section 6 comparison table.*

### Step 5: Real-time Inference
Run inference on any query image (e.g., the sample image `data/query_face.jpg`). The system automatically detects faces, extracts biometric features, and performs vector similarity search against the database to output the matched label.

* Auto-locate and use the latest best checkpoint:
```bash
python inference.py --config config.json --query data/query_face.jpg
```

* Direct run with a specific checkpoint path:
```bash
python inference.py --config config.json --checkpoint saved/models/LicensePlateRecognition/latest/best-checkpoint-epoch=xxxx-val_loss=xxxx.ckpt --query data/query_face.jpg
```

*The annotated query image will be saved under `saved/models/predictions/inference_result.png` and opened automatically.*
