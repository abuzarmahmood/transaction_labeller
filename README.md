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
3. Click "Save Results" to download the categorized transactions

## Input File Format

The system expects CSV or Excel files with the following columns:
- `Name` (required) - Transaction description
- `Category` (optional) - Existing categories
- `Date` (optional) - Transaction date
- `Amount` (optional) - Transaction amount
- `Account` (optional) - Account information

## Model Training

The system uses:
- Bag of words feature extraction
- Multinomial Naive Bayes classifier
- Top-5 category suggestions
- Cross-validation for model evaluation

Model and vectorizer are automatically saved to an `artifacts` directory.

## Features

- Automatic category prediction
- Top-5 category suggestions
- Manual category override
- Alternating row colors for readability
- Sort and filter capabilities
- Cross-platform compatibility
- Persistent model storage
- CSV/Excel file support

## Project Structure

```
.
├── artifacts/
│   ├── model.joblib
│   └── vectorizer.joblib
├── pred_transactions.py
├── transaction_gui.py
└── transaction_streamlit.py
```
