# Global Superstore MLflow Project

## Overview
This repository contains Jupyter Notebooks for training and deploying a machine learning model using MLflow with the Global Superstore dataset. The project includes two main notebooks:

1. **Global_superstore_deploy.ipynb** - Focuses on deploying a trained model using MLflow.
2. **Global_super_store_MLflow.ipynb** - Covers data exploration and model training with MLflow tracking.

---

## Notebook Summaries

### 1. Global_super_store_MLflow.ipynb
- **Purpose**: Prepares and trains a model while logging experiments with MLflow.
- **Key Features**:
  - Loads and explores the Global Superstore dataset.
  - Uses Pandas, Seaborn, and Matplotlib for data analysis.
  - Implements MLflow to track experiments and model performance.

### 2. Global_superstore_deploy.ipynb
- **Purpose**: Deploys a trained ML model using MLflow.
- **Key Features**:
  - Configures environment variables for MLflow tracking.
  - Searches for existing MLflow runs.
  - Deploys the trained model using `mlflow models serve`.

---

## Prerequisites
To run these notebooks, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- MLflow
- Pandas, NumPy, Matplotlib, and Seaborn

---

## Usage
1. Run `Global_super_store_MLflow.ipynb` to train and track your model.
2. Run `Global_superstore_deploy.ipynb` to deploy the trained model.

---

## Notes
- Update the dataset path in `Global_super_store_MLflow.ipynb` before running.
- Make sure MLflow is configured correctly before deployment.

For further details, refer to the respective notebooks.


