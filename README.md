# 📊 Research Topic Prediction

A machine learning–based project that predicts **research topics** from input text (abstract/title) using NLP techniques.

## Features
- Text preprocessing and feature extraction
- ML-based topic prediction
- Train and test pipeline included
- Simple and extensible codebase

## Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- NLP (TF-IDF / CountVectorizer)

## Project Structure
```
Research-topic-prediction-main/
├── main.py                 # Entry point
├── train.py                # Model training
├── predict.py              # Topic prediction logic
├── dataset/                # Research text dataset
├── model.pkl               # Trained model
├── requirements.txt        # Dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Train Model
```bash
python train.py
```

## Predict Topic
```bash
python predict.py
```

## Notes
- Ensure dataset path is correct before training
- Model can be replaced with advanced NLP models


