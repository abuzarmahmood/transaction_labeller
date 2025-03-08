# Transaction Categorizer

A machine learning-powered tool for automatically categorizing financial transactions with multiple user interfaces.

![image](https://github.com/user-attachments/assets/0a8fd5cb-27d4-479d-b6a5-431ee9020fb5)

## Overview

This project provides a system for categorizing financial transactions using machine learning (Naive Bayes) with both a desktop GUI (Tkinter) and web interface (Streamlit). The system:

- Predicts transaction categories using a bag-of-words approach
- Provides top 5 category suggestions for each transaction
- Allows manual category selection/override
- Supports CSV and Excel file formats
- Saves categorized transactions back to CSV

## Components

- `pred_transactions.py` - Core ML model training and prediction logic
- `transaction_gui.py` - Desktop GUI interface using Tkinter
- `transaction_streamlit.py` - Web interface using Streamlit
- `config.py` - Configuration settings for file paths and model parameters

## Requirements

```
pandas
numpy
scikit-learn
joblib
tkinter
streamlit
```

## Usage

### Desktop GUI

```bash
python transaction_gui.py
```

1. Click "Load Spreadsheet" to select your transaction file
2. Review predicted categories for each transaction
3. Click suggestion buttons or use dropdown to select categories
4. Click "Save" to export categorized transactions

### Web Interface 

```bash
streamlit run transaction_streamlit.py
```

1. Upload your transaction file using the file uploader
2. Review and categorize transactions in the interactive interface
3. Flag transactions for further review
4. Click "Save Results" to download the categorized transactions

## Input File Format

The system expects CSV or Excel files with the following columns:
- `Name` (required) - Transaction description
- `Category` (optional) - Existing categories
- `Date` (optional) - Transaction date
- `Amount` (optional) - Transaction amount
- `Account` (optional) - Account information
- `Flag` (optional) - Boolean flag for marking transactions for review

Example data is provided in `sample_transactions.csv`.

## Model Training

The system uses:
- Bag of words feature extraction with CountVectorizer
- Multinomial Naive Bayes classifier
- Top-5 category suggestions with probability scores
- Cross-validation for model evaluation

Model and vectorizer are automatically saved to an `artifacts` directory as defined in `config.py`.

## Features

- Automatic category prediction
- Top-5 category suggestions with confidence percentages
- Manual category override
- Transaction flagging for review
- Alternating row colors for readability
- Responsive interfaces for both desktop and web
- Cross-platform compatibility
- Persistent model storage
- CSV/Excel file support

## Project Structure

```
.
├── artifacts/
│   ├── model.joblib
│   └── vectorizer.joblib
├── data/
│   └── raw/
│       └── transactions.csv
├── config.py
├── pred_transactions.py
├── transaction_gui.py
├── transaction_streamlit.py
└── sample_transactions.csv
```

## Implementation Details

### Prediction Logic
The `predict_categories` function returns both the top category predictions and their associated probabilities, allowing users to see the model's confidence in each suggestion.

### User Interfaces
- **Tkinter GUI**: Provides a desktop application with scrollable transaction list, category suggestions as buttons, and dropdown selection.
- **Streamlit Web App**: Offers a responsive web interface with similar functionality plus transaction flagging and alternating row colors for better readability.

### Configuration
The `config.py` file centralizes path management and ensures required directories exist, making deployment more robust across different environments.
