# 🐾 Dog Breed Detection with CNN and EfficientNetB3

This project is a deep learning–based image classification tool for identifying dog breeds using high-resolution images. Built with PyTorch and deployed via Streamlit, it classifies 120 dog breeds through an interactive web interface.

---

## 🚀 Features

- 🔍 Detects **120** different dog breeds  
- 🧠 Utilizes **transfer learning** with EfficientNetB3  
- 🎯 High accuracy via **label smoothing** & **data augmentation**  
- 🖼️ Robust to real-world variability (backgrounds, angles, lighting)  
- 💻 Supports GPU inference and easy **Streamlit** deployment

---

## 🧠 Model Summary

| Component       | Description                              |
| --------------- | ---------------------------------------- |
| Framework       | PyTorch                                  |
| Backbone        | EfficientNetB3                           |
| Input Size      | 240 × 240                                |
| Augmentations   | Rotation, Flip, Color Jitter, Perspective |
| Loss Function   | CrossEntropy with Label Smoothing        |
| Optimizer       | Adam                                     |
| Scheduler       | ReduceLROnPlateau                        |
| Validation Acc. | ~96%                                     |

---

## 📂 Project Structure

```
deep/
├── app3.py                           # Streamlit application
├── model.py                          # Training & evaluation code
├── split.py                          # Dataset splitting script
├── breed_info.json                   # Breed metadata
├── best_model.pth                    # Best checkpoint weights
├── efficientnet_dog_classifier_final.pth  # Final model weights
├── Dataset/                          # Image folders: train/val/test
└── README.md                         # This documentation
```

---

## 🛠️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Sanyuktha-hansraj/dog-breed-detection.git
   cd dog-breed-detection
   ```

2. **Install dependencies**  
   ```bash
   pip install torch torchvision streamlit pillow matplotlib tqdm
   ```

3. **Run the app locally**  
   ```bash
   streamlit run app3.py
   ```

---

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub (✅ done).  
2. Sign in at [Streamlit Cloud](https://streamlit.io/cloud).  
3. Click **New App**, select your repo, set the main file to `app3.py`, and **Deploy**.

---

## 📊 Dataset

Uses a balanced subset of the Stanford Dogs Dataset (max 100 images per class) for efficient training and validation.

---

## 📌 Requirements

- Python 3.8+  
- `torch`  
- `torchvision`  
- `streamlit`  
- `pillow`  
- `matplotlib`  
- `tqdm`

You can freeze these into a `requirements.txt`:

```text
torch
torchvision
streamlit
pillow
matplotlib
tqdm
```

---

## 👩‍💻 Author

**Biswadas E J**  
MSc Computer Science in specialization with Data Analytics  
[GitHub Profile]((https://github.com/Biswa2Das)

---


