#  Diabetes Prediction using Machine Learning  
*FS25 Foundations of Data Science*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/Models-KNN%2C%20RF%2C%20SVM%2C%20LogReg-orange)

---
## Table of Contents

## 📚 Table of Contents

1. [About the Project](#about-the-project)  
   └ [Built With](#built-with)  
2. [Getting Started](#getting-started)  
   └ [Dependencies](#dependencies)  
   └ [Installation](#installation)  
3. [Usage](#usage)  
   └ [Repository Structure](#repository-structure)  
4. [Authors](#authors)  

---

## About The Project
This project implements a robust end-to-end machine learning pipeline for predicting the likelihood of diabetes onset, using the [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)—a widely recognized benchmark in clinical data science. The workflow includes standardized preprocessing, modular pipeline construction, and multiple classification algorithms (KNN, SVM, Random Forest, and Logistic Regression). To ensure unbiased model selection and reliable generalization estimates, we employ nested cross-validation, fully encapsulating both hyperparameter tuning and evaluation within a reproducible scikit-learn pipeline.



## Getting Started

To get a local copy up and running follow these simple steps.

### Dependencies

Below are the main Python libraries and tools required to run this project.  
We recommend installing them in a clean virtual environment using `conda`.

- **Python 3.10**

- **Pandas 2.2.1**

- **NumPy 1.26.4**

- **matplotlib 3.8.4**

- **Seaborn 0.13.2**

- **Scikit Learn 1.4.2**

- **Imbalanced-learn 0.12.2**






## Usage

To run the diabetes prediction pipeline:

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/mettlrob/FoDS-Diabetes-Diagnosis.git
   cd FoDS-Diabetes-Diagnosis


2. **Set up the environment**
We recommend using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Run the main script**
   ```bash
   python main.py


## Repository Structure

FoDS-Diabetes-Diagnosis/
│
├── main/                    # Contains main pipeline script (model_pipeline.py)
│   └── model_pipeline.py    # Core script for training and evaluation
│
├── data/                    # Raw dataset and processed files
├── pipeline_output/         # Outputs: plots, SHAP values, evaluation metrics
├── support/                 # Helper functions and utilities
├── archive/                 # Deprecated or older scripts
│
├── README.md                # This file
├── diabetes_Pima_paper.pdf  # PDF writeup of the project



## Authors

Tobias Herrmann

Michael Keller

Robin Mettler

Georg Weber










