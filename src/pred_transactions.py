"""
Use bag of words with naive bayes to predict transactions category
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from config import DATA_PATH
import os
from pathlib import Path
import joblib

# def load_model_and_vectorizer():
#     """Return the trained model and vectorizer"""
#     return global_model, global_vectorizer

def return_model_and_vectorizer():
    """Return the trained model and vectorizer"""
    script_dir = Path(__file__).parent
    model_path = script_dir / 'artifacts' / 'model.joblib'
    vectorizer_path = script_dir / 'artifacts' / 'vectorizer.joblib'
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_categories(model, vectorizer, transactions, n=5):
    """
    Predict top n categories and their probabilities for multiple transactions
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        transactions (list): List of transaction description texts
        n (int): Number of top categories to return
        
    Returns:
        tuple: (categories, probabilities) where each is a list of lists
    """
    X = vectorizer.transform(transactions)
    proba = model.predict_proba(X)
    top_n_indices = np.argsort(proba, axis=1)[:, -n:]
    
    categories = []
    probabilities = []
    
    for idx, indices in enumerate(top_n_indices):
        row_categories = []
        row_probabilities = []
        for cat_idx in indices[::-1]:
            row_categories.append(model.classes_[cat_idx])
            row_probabilities.append(proba[idx, cat_idx])
        categories.append(row_categories)
        probabilities.append(row_probabilities)
    
    return categories, probabilities

def top_n_accuracy(model, X, y, n=5):
    top_n = np.argsort(model.predict_proba(X), axis=1)[:, -n:]
    top_n = [model.classes_[i] for i in top_n]

    return np.mean([y[i] in top_n[i] for i in range(len(y))])

if __name__ == '__main__':

    raw_data_path = Path(DATA_PATH)
    raw_data = pd.read_csv(raw_data_path)

    categories = raw_data['Category'].unique()
    y = raw_data['Category'].values
    transactions = raw_data['Name'].values

    # Extract feature
    vectorizer = CountVectorizer()
    vectorizer.fit(transactions)
    X = vectorizer.transform(transactions)

    # Train model
    model = MultinomialNB()
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Save model and vectorizer
    artifact_dir = raw_data_path.parent.parent.parent / 'artifacts' 
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifact_dir, 'model.joblib')) 
    joblib.dump(vectorizer, os.path.join(artifact_dir, 'vectorizer.joblib'))

    # Evaluate
    accuracy = accuracy_score(y, y_pred)

    # Check accuracy in top n predicted categories
    n = 3
    top_n = np.argsort(model.predict_proba(X), axis=1)[:, -n:]
    top_n = [model.classes_[i] for i in top_n]

    top_n_accuracy = np.mean([y[i] in top_n[i] for i in range(len(y))])

    ##############################
    # Get cross validation score #
    ##############################
    from sklearn.model_selection import cross_val_score
    model = MultinomialNB()
    scores = cross_val_score(model, X, y, cv=5)
    print('Cross validation scores:', scores)

    # Check accuracy in top n predicted categories
    scores = cross_val_score(model, X, y, cv=5, scoring=top_n_accuracy)

    # Save model and vectorizer as global variables
    global_model = MultinomialNB()
    global_model.fit(X, y)
    global_vectorizer = vectorizer

