from ultralytics import YOLO

def main():
    # Load the best trained model
    model_path = r"runs\detect\herbal_plant_model-4\weights\best.pt"
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export the model to ONNX format
    print("Exporting model to ONNX format...")
    # opset=12 is widely supported by ONNX Runtime and TFLite
    success = model.export(format="onnx", opset=12, device="cpu")
    print(f"Export successful. ONNX model saved at: {success}")
    
    # Export the model to TFLite format
    print("Exporting model to TFLite format...")
    success_tflite = model.export(format="tflite", device="cpu")
    print(f"Export successful. TFLite model saved at: {success_tflite}")

if __name__ == "__main__":
    main()
