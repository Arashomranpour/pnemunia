<div align="center">

# 🫁 Pneumonia Detector

**Upload a chest X-ray and a Keras deep-learning classifier tells you whether it looks healthy or shows pneumonia.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## ✨ Features

- 🖼️ Upload a chest X-ray (`jpg`, `jpeg`, `png`).
- 🤖 A trained Keras model (`pneumonia_classifier.h5`) classifies the image and returns the **class name** and a **confidence score**.
- 🏷️ Class labels live in `labels.txt` (`0` pneumonia, `1` healthy).
- 🎨 Optional custom background for the page.

> ⚠️ Educational project. It is **not** a medical device and must not be used for diagnosis.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/pnemunia.git
cd pnemunia
pip install -r requirements.txt
```

The trained model file `pneumonia_classifier.h5` is not stored in this repository. Train your own model on a chest X-ray dataset (for example the Kaggle *Chest X-Ray Images (Pneumonia)* set) and save it next to `st.py`. Then:

```bash
streamlit run st.py
```

## 📁 Project Structure

```
.
├── st.py               # Streamlit UI
├── util.py             # Image preprocessing + classify()
├── labels.txt          # Class labels
└── requirements.txt
```

## 🛠️ Tech Stack

`Keras` · `Streamlit` · `NumPy` · `Pillow` · `Matplotlib`
