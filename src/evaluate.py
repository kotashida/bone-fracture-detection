from ultralytics import YOLO
import os
import argparse

def evaluate_model(split='test'):
    """
    Evaluates the trained YOLOv8 model on the specified data split.
    """
    # 1. Find the best model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(project_root, "bone_fracture_project")
    
    best_pt_files = []
    if os.path.exists(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            if "best.pt" in files:
                full_path = os.path.join(root, "best.pt")
                best_pt_files.append(full_path)
    
    if not best_pt_files:
        print("Error: No trained model found (best.pt). Please run src/train.py first.")
        return

    # Sort by modification time (newest first)
    best_pt_files.sort(key=os.path.getmtime, reverse=True)
    model_path = best_pt_files[0]
    print(f"Loading model from: {model_path}")

    # 2. Load the model
    model = YOLO(model_path)

    # 3. Evaluate
    print(f"Starting evaluation on '{split}' set...")
    # The data arg points to the config file which has 'train', 'val', 'test' paths
    metrics = model.val(data=os.path.join(project_root, "configs", "dataset.yaml"), split=split)

    # 4. Print Summary
    print("\n" + "="*40)
    print(f"Evaluation Results ({split} set)")
    print("="*40)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    evaluate_model(args.split)
