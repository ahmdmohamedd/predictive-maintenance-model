# Predictive Maintenance Model

## Overview

This project implements a **Predictive Maintenance** system aimed at forecasting the Remaining Useful Life (RUL) of industrial equipment using two machine learning models:
- **Random Forest Regressor**
- **Long Short-Term Memory (LSTM)**

The system leverages sensor and operational data to predict when equipment is likely to fail, enabling proactive maintenance strategies.

---

## Table of Contents

- [Project Description](#project-description)
- [Getting Started](#getting-started)
- [Data](#data)
- [Model Development](#model-development)
  - [Random Forest Regressor](#random-forest-regressor)
  - [LSTM Model](#lstm-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)

---

## Project Description

Predictive maintenance is a critical aspect of industrial operations, where the goal is to predict equipment failures before they occur. This project uses data from sensors that measure various operational settings and sensor values to forecast the **Remaining Useful Life (RUL)** of machines.

Two models have been developed:
- **Random Forest Regressor**: A tree-based model that can handle complex, non-linear relationships in the data.
- **Long Short-Term Memory (LSTM)**: A deep learning model designed to work with time-series data, capturing temporal dependencies in the sequence of sensor values.

The performance of both models is evaluated using **Mean Absolute Error (MAE)** and **R^2 Score** metrics.

---

## Getting Started

### Prerequisites

To run this project, ensure you have the following:
- Python 3.x
- Required Python libraries (see `requirements.txt` or the `install` section below)
- Jupyter Notebook (optional, for running the notebook interactively)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ahmdmohamedd/predictive-maintenance-model.git
    cd predictive-maintenance-model
    ```

2. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook predictive_maintenance_system.ipynb
    ```

---

## Data

The data used for this project is based on the **Turbofan engine degradation simulation dataset**. It contains data collected from sensors installed on turbofan engines, including operational settings and sensor readings. The goal is to predict the **Remaining Useful Life (RUL)** of engines based on these sensor values.

The dataset consists of the following columns:
- `engine_no`: Unique identifier for each engine
- `cycle`: The operating cycle number
- `operational_setting_*`: Operational settings
- `sensor_*`: Sensor readings
- `RUL`: Remaining Useful Life (target variable)

---

## Model Development

### Random Forest Regressor

The Random Forest Regressor model was trained to predict the RUL by learning the relationships between the operational settings, sensor readings, and the RUL values. It is a robust model suitable for regression tasks and can handle both linear and non-linear relationships.

#### Key Steps:
1. **Data Preprocessing**: Handling missing values and feature scaling.
2. **Model Training**: Split data into training and testing sets, trained using the Random Forest algorithm.
3. **Evaluation**: Model performance was evaluated using MAE and R^2 score.

### LSTM Model

The LSTM model is designed to work with sequential data, which makes it suitable for time-series forecasting. This model learns from the sequence of sensor readings and operational settings to predict the RUL.

#### Key Steps:
1. **Data Preparation**: Reshaping data into time-series sequences suitable for LSTM.
2. **Model Architecture**: A simple LSTM-based architecture was built for regression.
3. **Training**: Trained on the prepared time-series data.
4. **Evaluation**: Model performance was evaluated using MAE and R^2 score.

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

The MAE is used to measure the average magnitude of the errors in the predictions. It’s the average of the absolute differences between the predicted and actual values:
- **Lower MAE** indicates better model performance.

### R^2 Score

The R^2 score indicates how well the model explains the variance in the data:
- **Higher R^2** indicates a better model fit.
- **Negative R^2** indicates a poor model, worse than a simple mean-based predictor.

---

## Results

After training both models, the evaluation metrics for each are as follows:

### Random Forest Regressor:
- **MAE**: 29.57
- **R^2 Score**: 0.62

### LSTM Model:
- **MAE**: 54.49
- **R^2 Score**: -0.07

The **Random Forest** model outperformed the **LSTM** model, as indicated by the lower MAE and positive R^2 score. The LSTM model did not perform well due to issues like data preprocessing and the nature of the data, suggesting that more advanced tuning or more data might improve its performance.

---

## Contributing

If you would like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. Contributions can include:
- Improving model performance through hyperparameter tuning or feature engineering
- Adding new models or techniques for predictive maintenance
- Fixing bugs or enhancing the codebase

---
