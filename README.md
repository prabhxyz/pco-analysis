# Vision-Language Models To Prevent Posterior Capsule Opacification (PCO)

### ⚠️ The project is currently a work in progress, and the README is outdated.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training](#training)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The **Real-Time Surgical Guidance System** is an advanced tool designed to assist surgeons during cataract surgery by analyzing live video feeds and providing immediate feedback on surgical techniques. Leveraging state-of-the-art machine learning models, the system identifies subtle variations in surgical maneuvers, correlates them with risk factors for Posterior Capsule Opacification (PCO) development, and offers actionable insights to enhance surgical outcomes.

## Features

- **Real-Time Analysis:** Processes surgical video feeds in real-time to provide instant feedback.
- **Technique Assessment:** Evaluates surgical maneuvers across all procedure phases, identifying areas for improvement.
- **PCO Risk Prediction:** Predicts the likelihood of PCO development based on intraoperative observations.
- **Actionable Insights:** Offers recommendations to refine surgical techniques and minimize post-operative complications.
- **Training Tool:** Serves as an educational platform for new surgeons, accelerating the learning curve with personalized feedback.
- **Modular Design:** Comprises distinct modules for dataset handling, segmentation and tracking, vision-language processing, and pipeline orchestration.

## Architecture

The system is divided into four primary components:

1. **Dataset Management (`cataract_dataset.py`):** Handles data loading and preprocessing for the cataract-1k dataset.
2. **Segmentation & Tracking (`segmentation_tracking.py`):** Performs image segmentation and tracks regions of interest using optical flow.
3. **Vision-Language Processing (`vision_language.py`):** Combines visual and textual data to assess surgical techniques and predict PCO risk.
4. **Main Pipeline (`main_pipeline.py`):** Orchestrates data flow, model training, and inference processes.

## Installation

### Prerequisites

- **Python 3.9+**
- **CUDA-enabled GPU** (optional but recommended for faster processing)

### Steps

1. **Clone the Repository:**
    ```
    git clone https://github.com/prabhxyz/pco-analysis.git
    cd real-time-surgical-guidance
   ```

2. **Set Up the Project Structure:**
    Ensure your project directory has the following structure:
    ```
    your_project/
    ├── cataract_dataset.py
    ├── segmentation_tracking.py
    ├── vision_language.py
    ├── main_pipeline.py
    └── cataract-1k/
        ├── images/   # Place your surgical video frames here
        └── masks/    # Place corresponding segmentation masks here
    ```

3. **Create and Activate a Conda Environment:**
    ```
    conda create -n cataract_env python=3.9 -y
    conda activate cataract_env
    ```

4. **Install Dependencies:**
    ```
    pip install torch torchvision torchaudio timm transformers opencv-python tqdm numpy
    ```

## Usage

### Running the Pipeline

Execute the main pipeline script to train the models and run an example inference:

```
python main_pipeline.py
```

### Expected Output

Upon running the pipeline, you will observe logs indicating the progress of:

- **Segmentation Model Training**
- **Vision-Language Model Training**
- **Inference Steps:**
  - Segmentation
  - Optical Flow Calculation
  - Region-Based Tracking
  - Technique Assessment
  - PCO Risk Prediction

### Important Notes

> **Disclaimer:** This prototype uses placeholders for the **cataract-1k** dataset. Modify the dataset paths and structure to match your real data.
>
> - **Data Requirements:** Ensure minimal data is provided, or the system will use mock data for demonstration.
> - **Medical Validation:** This system is **not** medically validated. Use with caution and consult medical professionals for clinical applications.

## Project Structure

```
real-time-surgical-guidance/
├── cataract_dataset.py       # Dataset handling and DataLoader creation
├── segmentation_tracking.py  # Segmentation and tracking modules
├── vision_language.py        # Vision-Language model and related heads
├── main_pipeline.py          # Orchestrates training and inference
├── cataract-1k/              # Dataset directory
│   ├── images/               # Surgical video frames
│   └── masks/                # Segmentation masks
└── README.md                 # Project documentation
```

### File Descriptions

- **`cataract_dataset.py`:**
  - **`Cataract1KDataset`**: Simulates the dataset structure for cataract surgery videos.
  - **`get_cataract_dataloaders`**: Creates training and validation DataLoaders.

- **`segmentation_tracking.py`:**
  - **`SimpleSegmentationModel`**: DeepLabV3-based segmentation model with a ResNet50 backbone.
  - **Training Functions**: Train and perform inference with the segmentation model.
  - **Optical Flow**: Computes optical flow using OpenCV’s Farneback method.
  - **Region-Based Tracking**: Identifies and tracks the centroid of the largest segmented region.

- **`vision_language.py`:**
  - **`VisionLanguageModel`**: Combines ViT (Vision Transformer) and BERT for vision-language tasks.
  - **`TechniqueAssessmentHead`**: Assesses surgical quality indices.
  - **`PCORiskPredictionHead`**: Predicts the probability of PCO development.
  - **Training Functions**: Train the Vision-Language model using contrastive learning.
  - **Inference Functions**: Perform technique assessment and PCO risk prediction.

- **`main_pipeline.py`:**
  - **Workflow Orchestration**: Loads data, initializes models, trains segmentation and vision-language models, and performs inference.

## Models

### Segmentation Model

- **Architecture:** DeepLabV3 with ResNet50 backbone.
- **Purpose:** Segment surgical instruments and anatomical structures in video frames.
- **Training:** Utilizes cross-entropy loss for binary segmentation (background vs. instrument).

### Vision-Language Model

- **Architecture:** Combines Vision Transformer (ViT) and BERT.
- **Purpose:** Aligns visual features with textual surgical notes for technique assessment and PCO risk prediction.
- **Training:** Uses a contrastive loss similar to CLIP to align image and text embeddings.

### Assessment and Prediction Heads

- **TechniqueAssessmentHead:** Outputs vectors representing surgical quality indices, predicted errors, and recommended adjustments.
- **PCORiskPredictionHead:** Outputs the probability of PCO development based on fused embeddings.

## Training

### Segmentation Model Training

The segmentation model is trained using the `train_segmentation_model` function. It processes batches of images and masks, optimizing the model to accurately segment surgical instruments.

### Vision-Language Model Training

The Vision-Language model is trained using the `train_vlm` function. It aligns visual features from surgical frames with corresponding textual prompts, enabling the model to understand and assess surgical techniques.

## Inference

### Example Inference Flow

1. **Segmentation:**
   - The trained segmentation model processes input frames to generate segmentation masks.

2. **Optical Flow Calculation:**
   - Computes the motion between consecutive frames to analyze surgical maneuvers.

3. **Region-Based Tracking:**
   - Identifies and tracks the centroid of the largest segmented region, aiding in instrument tracking.

4. **Technique Assessment:**
   - Uses the Vision-Language model to assess the quality of surgical techniques based on visual and textual data.

5. **PCO Risk Prediction:**
   - Predicts the likelihood of PCO development, allowing proactive measures during surgery.

### Running Inference

The `main_pipeline.py` script demonstrates an example inference using a batch from the validation DataLoader. It showcases the integration of segmentation, optical flow, tracking, technique assessment, and PCO risk prediction.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
    ```
    git checkout -b feature/YourFeature
    ```
3. **Commit Your Changes**
    ```
    git commit -m "Add Your Feature"
    ```
4. **Push to the Branch**
    ```
    git push origin feature/YourFeature
    ```
5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation and tests.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **OpenAI** for providing the foundational GPT models.
- **PyTorch** and **Torchvision** for their robust machine learning frameworks.
- **Transformers** by Hugging Face for state-of-the-art NLP models.
- **OpenCV** for their comprehensive computer vision library.
- **Timm** for efficient and flexible vision models.

---

**Disclaimer:** This system is a prototype and has not been medically validated. It is intended for research and educational purposes only. Consult medical professionals before applying in clinical settings.

---

*Built with ❤️ by [Prabhdeep](https://github.com/prabhxyz)*