# 🧠 CNN Handwriting Recognition

A deep learning–based Handwriting Recognition system built using Convolutional Neural Networks (CNNs).
This project can recognize **digits, uppercase letters, and lowercase letters** from handwritten images using the **EMNIST dataset**.

---

## 🚀 Features

* 🔤 Recognizes **62 classes** (0–9, A–Z, a–z)
* 🧠 CNN-based architecture with Batch Normalization & Dropout
* 📊 Trained on EMNIST (byclass)
* 🌐 Interactive **Streamlit Web App**
* 🖼️ Supports real-world handwritten image inputs
* 🔍 CLI-based prediction with debug visualization

---

## 📂 Project Structure

```
CNN-project/
│
├── app/
│   └── streamlit_app.py      # Web UI
│
├── src/
│   ├── train.py              # Model training
│   ├── predict.py            # CLI prediction
│   ├── preprocess.py         # Image preprocessing
│   └── model.py              # CNN architecture
│
├── notebooks/
│   └── mnist-dataset.ipynb   # MNIST exploration (optional)
│
├── outputs/                  # Saved models + debug images
├── archive/                  # EMNIST dataset (user must add)
├── requirements.txt
├── test.png
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/tofuux/CNN-project.git
cd CNN-project
```

---

### 2️⃣ Create virtual environment (recommended)

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install tensorflow opencv-python numpy pandas streamlit matplotlib
```

---

## 📁 Dataset Setup

### 🔹 Primary Dataset: EMNIST (byclass)

Download from:
👉 https://www.kaggle.com/datasets/crawford/emnist

Place inside project:

```
CNN-project/
├── archive/
│   ├── emnist-byclass-train.csv
│   └── emnist-byclass-test.csv
```

---

### 🔹 Optional: MNIST

Used only in notebook:

```
notebooks/mnist-dataset.ipynb
```

(No manual download required)

---

## 🏋️ Training the Model

```bash
python src/train.py
```

This will:

* Train CNN model
* Save best model to:

```
outputs/best_model.keras
```

---

## 🔍 Run Prediction (CLI)

```bash
python src/predict.py test.png
```

* Replace `test.png` with your own image
* Output displayed in terminal



## 🧠 Model Architecture

* Conv2D → BatchNorm → MaxPooling
* Conv2D → BatchNorm → MaxPooling
* Fully Connected Dense Layers
* Dropout for regularization
* Softmax output (62 classes)

---

## 📊 Technologies Used

* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* Streamlit
* Matplotlib

---

## ⚠️ Notes

* Model file must exist before prediction:

```
outputs/best_model.keras
```

* Run training first if model not present
* Works best with clean handwritten input

---

## 🤝 Contributing

Feel free to fork the repo and improve the model or UI!

---

## ⭐ Acknowledgements

* EMNIST Dataset
* TensorFlow & OpenCV communities

---


