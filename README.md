# HitsBE

## 📖 Description

HitsBE is a transformer-based embedding designed for time series. This module aims to recognize patterns in a time series and convert them into "words" that a model can interpret. Essentially, we seek to **paraphrase time series** to enhance their understanding in deep learning models.

This repository contains the source code and an initial model, **`HitsBERTClassifier`**, a BERT-based classifier that integrates the HitsBE module. You will also find an initial usage example to facilitate its implementation.

---

## 🚀 Installation

### 1️⃣ Prerequisites
For this project, you need:

- **Python** `>=3.10, <3.13`
- **Poetry**

💡 The project also includes **Docker** support, allowing deployment in controlled environments.

---

### 2️⃣ Clone the repository
```bash
git clone https://github.com/angelolmn/HitsBE.git
cd HitsBE   
```

### 3️⃣ Install dependencies

```bash
poetry install
```

## 📌 Model Usage

In experiments/hitsbert_example.py, you will find a usage example of HitsBERTClassifier. To execute it from the project root:

```bash
poetry run python experiments/hitsbert_example.py
```

This will run a script that trains the model with a small dataset to verify its functionality.