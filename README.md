---

## 📜 About the Project

This project is inspired by the **Siamese Neural Network architecture**, as described in the paper:

> Koch, Gregory. *"Siamese Neural Networks for One-shot Image Recognition."* ICML Deep Learning Workshop (2015).

Siamese networks learn to compare image pairs by producing embeddings and minimizing the distance between similar 
faces while maximizing the distance between different faces.

This notebook (`SetUp.ipynb`) helps create the required dataset for such models by capturing images via a webcam and
organizing them into appropriate folders for training.

---

## 📂 Project Structure

Face-Verification-Data-Setup/
│
├── data/
│ ├── anchor/ # Reference images of a person
│ ├── positive/ # Other images of the same person
│ ├── negative/ # Images of different people
│
├── notebooks/
│ └── SetUp.ipynb # Data collection and preprocessing notebook
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 Features

- Based on **Siamese Neural Network dataset requirements**
- Automatic folder creation for data organization
- Webcam-based image capture
- **Keyboard shortcuts** for fast labeling:
  - `a` → Capture **anchor** image
  - `p` → Capture **positive** image
  - `q` → Quit capture
- Unique image filenames using UUID
- Dataset directly compatible with **TensorFlow data pipelines**

---
