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
├── train.csv               # Model training csv file
├── test.csv                # Model testing csv file
├── submission.csv          # output dataset
├── requirements.txt        # Dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run using the command line:
```bash
python main.py
```

## Notes
- Ensure dataset path is correct before training
- Model can be replaced with advanced NLP models


