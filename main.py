import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_nltk():
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

def load_data():
    print("Loading data...")
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        raise FileNotFoundError("Data files (train.csv, test.csv) not found in current directory.")
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def preprocess_text(df, columns):
    print(f"Preprocessing text for columns: {columns}")
    # Removing Punctuations
    df[columns] = df[columns].replace('[^a-zA-Z]', ' ', regex=True)
    
    # Converting to lower case
    for col in columns:
        df[col] = df[col].str.lower()
        
    # Removing one letter words and multiple blank spaces
    for col in columns:
        df[col] = df[col].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
        df[col] = df[col].replace('\s+', ' ', regex=True)
    
    return df

def main():
    setup_nltk()
    train, test = load_data()
    
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)
    
    target_cols = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    
    # Preprocessing
    test_id = test['ID']
    test = test.drop(['ID'], axis=1)
    
    X = train.loc[:, ['TITLE', 'ABSTRACT']]
    y = train.loc[:, target_cols]
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    
    print("Cleaning training and validation data...")
    X_train = preprocess_text(X_train, ['TITLE', 'ABSTRACT'])
    X_val = preprocess_text(X_val, ['TITLE', 'ABSTRACT'])
    test = preprocess_text(test, ['TITLE', 'ABSTRACT'])
    
    # Combine Title and Abstract
    X_train['combined'] = X_train['TITLE'] + ' ' + X_train['ABSTRACT']
    X_val['combined'] = X_val['TITLE'] + ' ' + X_val['ABSTRACT']
    test['combined'] = test['TITLE'] + ' ' + test['ABSTRACT']
    
    train_lines = X_train['combined'].tolist()
    val_lines = X_val['combined'].tolist()
    test_lines = test['combined'].tolist()
    
    # Vectorization
    print("Vectorizing text data...")
    countvector = CountVectorizer(ngram_range=(1, 2))
    X_train_cv = countvector.fit_transform(train_lines)
    X_val_cv = countvector.transform(val_lines)
    test_cv = countvector.transform(test_lines)
    
    tfidfvector = TfidfTransformer()
    X_train_tf = tfidfvector.fit_transform(X_train_cv)
    X_val_tf = tfidfvector.transform(X_val_cv)
    test_tf = tfidfvector.transform(test_cv)
    
    # Model Training
    print("Training the model...")
    model = LinearSVC(C=0.5, class_weight='balanced', random_state=42)
    multi_model = MultiOutputClassifier(model)
    multi_model.fit(X_train_tf, y_train)
    
    # Validation
    print("Validating the model...")
    val_preds = multi_model.predict(X_val_tf)
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_preds))
    print(f"Accuracy Score: {accuracy_score(y_val, val_preds):.4f}")
    
    # Prediction on Test Set
    print("Predicting on test set...")
    test_preds = multi_model.predict(test_tf)
    
    # Saving results
    print("Saving results to submission.csv...")
    submit = pd.DataFrame({'ID': test_id})
    for i, col in enumerate(target_cols):
        submit[col] = test_preds[:, i]
        
    submit.to_csv('submission.csv', index=False)
    print("Done! Submission file saved as 'submission.csv'.")

if __name__ == "__main__":
    main()
