from ultralytics import YOLO
import os
import argparse

def train_model(epochs, imgsz, batch_size, model_name, device):
    """
    Trains the YOLOv8 model with specified parameters.
    """
    print(f"Initializing model: {model_name}...")
    model = YOLO(model_name)

    print(f"Starting training on device: {device}")
    print(f"Config: Epochs={epochs}, ImgSz={imgsz}, Batch={batch_size}")
    
    # Ensure config path is correct
    config_path = os.path.join("configs", "dataset.yaml")
    
    # Project name usually defines the root folder for runs
    project_name = "bone_fracture_project"
    run_name = f"yolo_v8_{model_name.replace('.pt', '')}"

    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name=run_name,
        exist_ok=True, # Overwrite existing run with same name
        plots=True,
        save=True
    )
    
    print(f"Training complete. Results saved to {results.save_dir}")
    return results.save_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Bone Fracture Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model (n, s, m, l, x)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu, 0, 1, etc.)")
    
    args = parser.parse_args()
    
    train_model(args.epochs, args.imgsz, args.batch, args.model, args.device)
