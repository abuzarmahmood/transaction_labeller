"""
Use bag of words with naive bayes to predict transactions category
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

raw_data_path = '/home/abuzarmahmood/projects/transcation_labeller/data/raw/transactions.csv'
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
def top_n_accuracy(model, X, y, n=5):
    top_n = np.argsort(model.predict_proba(X), axis=1)[:, -n:]
    top_n = [model.classes_[i] for i in top_n]

    return np.mean([y[i] in top_n[i] for i in range(len(y))])

scores = cross_val_score(model, X, y, cv=5, scoring=top_n_accuracy)

# Save model and vectorizer as global variables
global_model = MultinomialNB()
global_model.fit(X, y)
global_vectorizer = vectorizer

def load_model_and_vectorizer():
    """Return the trained model and vectorizer"""
    return global_model, global_vectorizer

def predict_categories(transaction_text, n=3):
    """
    Predict top n categories for a transaction description
    
    Args:
        transaction_text (str): Transaction description text
        n (int): Number of top categories to return
        
    Returns:
        list: Top n predicted categories
    """
    model, vectorizer = load_model_and_vectorizer()
    X_new = vectorizer.transform([transaction_text])
    proba = model.predict_proba(X_new)
    top_n_idx = np.argsort(proba[0])[-n:][::-1]
    return [model.classes_[i] for i in top_n_idx]
