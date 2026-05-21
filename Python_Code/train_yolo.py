from ultralytics import YOLO

def train_model():
    """
    Trains the YOLOv8n model using transfer learning on the custom dataset.
    Applies built-in data augmentation (flipping, scaling, etc.) automatically.
    """
    print("Loading pretrained YOLOv8n model...")
    # Load a pretrained YOLOv8 nano model (transfer learning)
    model = YOLO('yolov8n.pt') 

    print("Starting training process...")
    # Train the model using the configuration in data.yaml
    # - data: path to our configuration file
    # - epochs: how many times the model sees the data
    # - imgsz: image size (640 is standard)
    # - batch: batch size (adjust based on your GPU RAM, 8 or 16 is safe)
    # - augment: True applies scaling, flipping, color jitter, etc., by default in Ultralytics
    results = model.train(
        data='data.yaml',
        epochs=50,      # Start with 50 for quick results, increase if underfitting
        imgsz=640,
        batch=16,
        device='cpu',   # Force CPU training due to RTX 5060 CUDA compatibility issue
        name='herbal_plant_model'
    )
    
    print("Training complete!")
    return model

def evaluate_model(model):
    """
    Evaluates the model on the validation dataset.
    Extracts metrics like mAP, precision, and recall.
    """
    print("\nEvaluating model on validation set...")
    metrics = model.val()
    
    print("--- Evaluation Metrics ---")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}") # mean precision
    print(f"Recall:    {metrics.box.mr:.4f}") # mean recall
    print("--------------------------")
    
    # Note: Confusion matrix is automatically generated and saved in the runs/detect/val directory
    print("Confusion matrix and visualizations are saved in the 'runs/detect/val' directory.")

if __name__ == '__main__':
    # Make sure you have installed ultralytics: pip install ultralytics
    # Step 1: Train
    trained_model = train_model()
    
    # Step 2: Evaluate
    evaluate_model(trained_model)
    
    print("\nNext step: Run inference on test clips using 'yolo predict model=runs/detect/herbal_plant_model/weights/best.pt source=dataset/test/images'")
