# Bone Fracture Detection: Computer Vision & Quantitative Analysis

## Project Overview

This project implements a sophisticated Computer Vision system designed to automate the detection and localization of bone fractures in X-ray imagery. By leveraging state-of-the-art Deep Learning architectures, specifically **YOLOv8 (You Only Look Once)**, the system addresses the critical medical challenge of diagnostic variability.

The primary objective is to demonstrate the application of **quantitative modeling techniques** and **statistical evaluation frameworks** to unstructured medical data, transforming raw pixel intensity data into actionable diagnostic probabilities.

---

## Key Quantitative Skills

This project demonstrates proficiency in the following analytical and technical domains:

*   **Deep Learning & Computer Vision:**
    *   **Architecture Design:** Implementation of Single-Stage Detectors (YOLOv8) with Feature Pyramid Networks (FPN) for multi-scale feature extraction.
    *   **Transfer Learning:** Leveraging inductive bias from pre-trained COCO weights to accelerate convergence on medical imaging domains.
    *   **Hyperparameter Optimization:** Tuning learning rates, momentum, and anchor box parameters to minimize loss landscapes.
*   **Statistical Evaluation & Metrics:**
    *   **Performance Metrics:** Rigorous calculation of **Mean Average Precision (mAP)** and **Recall** to quantify detection fidelity.
    *   **Threshold Analysis:** Utilizing **Intersection over Union (IoU)** thresholds (0.5 - 0.95) to differentiate between localization errors and classification errors.
    *   **Error Analysis:** Interpreting Confusion Matrices to diagnose False Positive/Negative rates and class-conditional biases.
*   **Data Engineering & Preprocessing:**
    *   **Tensor Manipulation:** Handling high-dimensional image tensors and normalizing pixel intensity distributions.
    *   **Data Augmentation:** Applying geometric (rotation, shear) and photometric (brightness, contrast) transformations to model data variance and improve generalization.
*   **Software Engineering for AI:**
    *   **MLOps:** Modular Python architecture with `argparse` for experiment reproducibility.
    *   **Containerization:** Docker-based deployment ensuring consistent runtime environments.
    *   **Interactive Visualization:** Developing Streamlit dashboards for real-time model inference and probability visualization.

---

## Methodology & Quantitative Approach

### 1. Data Acquisition & Distribution Analysis
The model is trained on the [Bone Fracture Detection Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) dataset.
*   **Class Taxonomy:** 7 distinct fracture types (e.g., `elbow positive`, `humerus fracture`, `forearm fracture`).
*   **Quantitative Justification:** The dataset follows the YOLO format, allowing for direct supervision of bounding box regression. We analyze the class distribution to identify potential biases (class imbalance) that could skew the model towards majority classes, necessitating stratified sampling or class-weighted loss functions during training.

### 2. Model Architecture: YOLOv8
We selected YOLOv8 due to its superior trade-off between **inference latency** and **mean Average Precision (mAP)** compared to two-stage detectors like Faster R-CNN.
*   **Anchor-Free Detection:** Unlike previous iterations utilizing anchor boxes, YOLOv8 employs an anchor-free approach, reducing the number of hyperparameters and simplifying the underlying optimization problem.
*   **Loss Function formulation:** The model optimizes a composite loss function:
    *   **VFL (Varifocal Loss):** For classification, prioritizing high-quality samples.
    *   **DFL (Distribution Focal Loss) + CIoU (Complete IoU):** For bounding box regression. CIoU quantifies the overlap, aspect ratio, and central distance between predicted and ground truth boxes, providing a continuous and differentiable metric for geometric alignment.

### 3. Evaluation Framework
The model's performance is not evaluated on simple accuracy, but rather on statistically robust metrics standard in object detection:
*   **Intersection over Union (IoU):** Defined as $Area(Prediction \cap GroundTruth) / Area(Prediction \cup GroundTruth)$. This measures the geometric fidelity of the detection.
*   **Mean Average Precision (mAP):**
    *   **mAP@0.5:** The area under the Precision-Recall curve when a detection is considered "correct" at an IoU $\ge$ 0.5.
    *   **mAP@0.5:0.95:** A more rigorous metric averaging mAP across IoU thresholds from 0.5 to 0.95 (step 0.05). This penalizes poor localization, ensuring the model is not just detecting the *presence* of a fracture, but accurately defining its *extent*.

---

## Project Structure

The repository is organized to separate concerns between data ingestion, modeling, and deployment:

```
bone_fracture_detection/
├── configs/
│   └── dataset.yaml       # Data path configuration and class definitions
├── src/
│   ├── app.py             # Streamlit inference dashboard (Presentation Layer)
│   ├── download_data.py   # Data ingestion pipeline
│   ├── evaluate.py        # Statistical evaluation script
│   ├── predictor.py       # Inference class encapsulating model logic
│   └── train.py           # Training script with CLI arguments for hyperparameter tuning
├── tests/
│   └── test_inference.py  # Unit tests for prediction logic
├── Dockerfile             # Container definition for reproducibility
├── requirements.txt       # Dependency management
└── README.md              # Documentation
```

---

## Usage

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Pipeline
Execute the download script to fetch the latest dataset revision from KaggleHub and verify integrity.
```bash
python src/download_data.py
```

### 3. Model Training (Optimization)
Train the Deep Learning model. Arguments allow for hyperparameter tuning (Epochs, Batch Size, Model Complexity).
```bash
# Example: Train a Medium (m) model for 50 epochs on GPU
python src/train.py --model yolov8m.pt --epochs 50 --batch 16 --device 0
```
*This process generates training logs, confusion matrices, and precision-recall curves in the `bone_fracture_project/` directory.*

### 4. Inference Application
Launch the interactive dashboard to perform real-time inference on new X-ray images.
```bash
streamlit run src/app.py
```

### 5. Statistical Evaluation
Run the evaluation script to generate performance metrics (mAP, Recall, Precision) on the test set.
```bash
python src/evaluate.py --split test
```

---

## Future Quantitative Expansions

*   **Bayesian Hyperparameter Optimization:** Implementing Optuna to statistically search for optimal learning rates and momentum values.
*   **Ensemble Methods:** Aggregating predictions from models trained on different folds (K-Fold Cross Validation) to reduce variance and quantify uncertainty.
*   **Model Quantization:** Reducing model precision (FP32 to INT8) to analyze the trade-off between computational cost (FLOPS) and statistical accuracy.
