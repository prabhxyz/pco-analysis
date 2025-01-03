# Vision-Language Models To Prevent Posterior Capsule Opacification (PCO)

## Project Files Overview

### **1) `cataract_seg_dataset.py`**

Handles loading the segmentation dataset to train models that identify surgical instruments in video frames.

- **Key Features**:
  - Maps instrument classes (e.g., "Pupil", "PhacoTip") to integer IDs.
  - Converts polygon annotations into pixel-wise segmentation masks.
  - Applies data augmentation using **Albumentations**.
- **Purpose**: Prepares `(image_tensor, mask_tensor)` pairs for training segmentation models.

---

### **2) `cataract_phase_dataset.py`**

Handles loading the phase recognition dataset, linking annotation CSVs with corresponding surgical videos.

- **Key Features**:
  - Extracts and samples video frames from annotated phases (e.g., “Incision”).
  - Assigns integer labels to surgical phases.
  - Ensures balanced sampling to reduce redundancy.
- **Purpose**: Prepares `(frame_tensor, phase_label)` pairs for training phase recognition models.

---

### **3) `models.py`**

Contains PyTorch implementations of the models used for segmentation and phase recognition.

- **Models**:
  - `LightweightSegModel`: A **DeepLabv3** model with a **MobileNetV3** backbone for instrument segmentation.
  - `PhaseRecognitionNet`: A **MobileNetV3-Large** model for classifying surgical phases.
- **Purpose**: Provides lightweight, real-time capable architectures optimized for cataract surgery data.

---

### **4) `technique_assessment.py`**

Implements heuristic-based **technique feedback** using recognized phases and instrument presence.

- **Example Rules**:
  - If phase = "Incision" but no "SlitKnife" detected, warn about missing instruments.
  - If phase = "Phacoemulsification" but no "PhacoTip," suggest potential issues.
- **Purpose**: Offers actionable feedback during surgery based on predefined rules.

---

### **5) `pco_assessment.py`**

Estimates **PCO risk** using domain knowledge heuristics.

- **Example Rules**:
  - If "IrrigationAspiration" is missing, flag higher PCO risk.
  - Tracks whether critical steps like "Capsule Polishing" were performed.
- **Output**: Provides a simple risk estimate: “Low,” “Medium,” or “High.”

---

### **6) `assistant.py`**

A minimal **chat assistant** for interacting with the system during surgery.

- **Capabilities**:
  - Answers queries about the current phase, instruments, or PCO risk.
  - Simple rule-based text matching (can be extended with NLP models).
- **Purpose**: Facilitates user interaction during real-time inference.

---

### **7) `train_all.py`**

A unified training pipeline for segmentation and phase recognition models.

- **Features**:
  - Accepts command-line arguments for dataset paths, hyperparameters, and model configurations.
  - Supports data augmentation, mixed-precision training, and learning rate scheduling.
  - Saves trained models as `lightweight_seg.pth` and `phase_recognition.pth`.
- **Purpose**: Streamlines the training process for both tasks.

---

### **8) `real_time_demo.py`**

Demonstrates real-time inference by integrating segmentation, phase recognition, and feedback modules.

- **Workflow**:
  - Loads trained models and processes video or live camera feeds.
  - Displays overlays (segmentation masks, phase labels) on video frames.
  - Provides real-time feedback on technique and PCO risk.
  - Optional terminal-based chat for user queries.
- **Purpose**: Showcases the system's real-time capabilities.

---

## **Getting Started**

### **Data Preparation**

1. Organize data in the following structure:
   - `datasets/Cataract-1k/segmentation`: Instrument segmentation frames + JSON annotations.
   - `datasets/Cataract-1k/phase`: Videos and corresponding phase annotation CSVs.

---

### **Training**

Train the segmentation and phase recognition models using `train_all.py`:

```bash
python train_all.py --seg_data_root datasets/Cataract-1k/segmentation \
                    --phase_data_root datasets/Cataract-1k/phase \
                    --seg_epochs 10 \
                    --phase_epochs 12 \
                    --batch_size 8 \
                    --lr 1e-4
```

- **Outputs**:
  - `lightweight_seg.pth`: Segmentation model weights.
  - `phase_recognition.pth`: Phase recognition model weights.

---

### **Real-Time Inference**

Run real-time inference with `real_time_demo.py`:

```bash
python real_time_demo.py --video /path/to/video.mp4 \
                         --seg_model lightweight_seg.pth \
                         --phase_model phase_recognition.pth \
                         --chat
```

- **Features**:
  - Overlays segmentation results on video frames.
  - Classifies surgical phases and provides real-time feedback.
  - Enables text-based interaction for queries about phase or PCO risk.

---

## **Technical Highlights**

1. **Deep Learning Models**:
   - **DeepLabv3 + MobileNetV3** for segmentation.
   - **MobileNetV3-Large** for phase classification.
2. **Mixed-Precision Training**:
   - Accelerates training with lower memory consumption using `torch.amp`.
3. **Data Augmentation**:
   - Robust augmentations via **Albumentations** to improve model generalization.
4. **Heuristic Modules**:
   - Provides immediate insights into surgical technique and PCO risk.

---

## **Future Improvements**

- Replace heuristic modules with data-driven models for technique assessment and PCO risk.
- Integrate NLP-based language models into `assistant.py` for advanced dialogue capabilities.
- Expand the dataset to cover more surgical cases and phases for better generalization.

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