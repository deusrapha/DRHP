# Detection and Recognition of Herbal Plants from Video Frames

This repository contains the code, dataset structure, and report for the Computer Vision project: **Detection and Recognition of Herbal Plants from Video Frames Using YOLOv8 and Image Enhancement Techniques**.

## Project Overview
This project tackles the challenge of herbal plant detection in diverse, natural environments. We process raw video clips into a robust frame dataset, apply classical computer vision techniques like CLAHE and HSV color correction for feature enhancement, and use **YOLOv8n** for precise object detection.

## Directory Structure
```text
.
├── Raw_Videos/                # Original 21 video clips (Not uploaded due to size limits)
├── dataset/                   # The YOLO-formatted dataset (images and labels)
├── sample_results/            # Key evaluation metrics and prediction screenshots
├── runs/                      # YOLOv8 training outputs and weights
├── create_full_image_labels.py# Script for generating initial full-frame labels
├── data.yaml                  # Dataset configuration for YOLO
├── export_model.py            # Script for exporting the model to ONNX/TFLite
├── frame_extractor.py         # Script to sample frames from videos
├── image_enhancement.py       # Script applying CLAHE and HSV enhancements
├── run_inference.py           # Script to test the trained model on unseen images
├── split_dataset.py           # Script to split dataset into train/val/test
├── train_yolo.py              # Main training script for YOLOv8
├── requirements.txt           # Python dependencies
├── cvpr_report_draft.md       # CVPR formatted project report
└── presentation_outline.md    # PowerPoint presentation outline
```

## Reproducibility and Setup

To reproduce the experimental results on your local machine, follow the setup and execution commands below.

### 1. Installation
Ensure you have Python 3.13 installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
If you are generating the dataset from scratch, first extract the frames and then split them into training, validation, and test sets.

```bash
# Split the dataset into 70% Train, 20% Val, 10% Test
python split_dataset.py
```

### 3. Model Training
Train the YOLOv8 nano model. The `train_yolo.py` script utilizes the `data.yaml` configuration file and trains the model for 50 epochs on a CPU.

```bash
python train_yolo.py
```

### 4. Inference and Evaluation
To run inference on the test set and visualize the bounding box predictions:

```bash
python run_inference.py
```

You can view the resulting metrics (mAP, Precision, Recall) and visual outputs (Confusion Matrix, PR Curves) in the `runs/detect/herbal_plant_model-4/` directory, or see the highlights in the `sample_results/` folder.

## Results Summary
The model achieved outstanding performance on the validation set:
- **Precision**: 0.999
- **Recall**: 1.000
- **mAP@50**: 0.995

For more detailed analysis, please refer to the `cvpr_report_draft.md`.
