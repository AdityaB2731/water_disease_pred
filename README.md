# water_disease_pred

A machine learning project to predict water-borne diseases from water quality and environmental features.

---

## Overview

This project uses water quality parameters and environmental factors to predict whether waterborne diseases may occur.  
It includes scripts for:

- Training a machine learning model  
- Saving and reusing the trained model  
- Running predictions on new input data  
- Performing quick sanity checks  

---

## Project Structure

```
water_disease_pred/
│
├── water_disease_data.csv       # Dataset used for training
├── train_model.py                # Train and save the ML model
├── predict_from_doc.py           # Run predictions from input documents
├── check.py                      # Quick utility script for testing
├── dummy_input.txt               # Example input for predictions
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Data

- **water_disease_data.csv** — Dataset containing features (water quality, rainfall, health symptoms, etc.) and disease labels.  
- **dummy_input.txt** — Example input file to demonstrate prediction usage.  

---

## Features

The dataset includes:

- **Physical parameters**: Turbidity, pH  
- **Biological indicators**: Bacteria presence  
- **Environmental attributes**: Rainfall, season, past cases  
- **Patient symptoms**: Fever, diarrhea, abdominal pain  

The **target variable** is the presence or type of waterborne disease.

---

## Requirements

Install Python 3.x and dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:

- pandas  
- numpy  
- scikit-learn  
- joblib  

---

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This script will:

- Load and preprocess the dataset  
- Train a machine learning model  
- Save the trained model for later use  

---

### 2. Run Predictions

To predict from a new input file:

```bash
python predict_from_doc.py --input dummy_input.txt
```

For a quick check:

```bash
python check.py
```

Both scripts will load the trained model and output predictions.

---

## Contributing

Contributions are welcome!  
You can help by improving preprocessing, testing new ML models, adding evaluation metrics, or building a frontend/API.

Steps to contribute:

1. Fork the repository  
2. Create a new branch  
3. Commit your changes  
4. Submit a pull request  

---

## License

This project is open-source. Please add a LICENSE file (MIT, Apache, or your choice).

---

## Contact

**Author**: Aditya Bajaj  
**GitHub**: [AdityaB2731](https://github.com/AdityaB2731)
