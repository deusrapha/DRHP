from ultralytics import YOLO

def main():
    # Load the best trained model
    model_path = r"runs\detect\herbal_plant_model-4\weights\best.pt"
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run inference on test dataset images
    test_dir = r"dataset\test\images"
    
    # Run inference and save the results
    print(f"Running inference on images in {test_dir}...")
    results = model.predict(source=test_dir, save=True, save_txt=True, conf=0.5, project="runs/detect", name="test_inference", device="cpu")
    print(f"Inference complete. Results saved to {results[0].save_dir}")

if __name__ == "__main__":
    main()
