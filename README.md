# Energy Price Prediction

This project predicts the future price of crude oil using historical data and a machine learning model. It uses Python, Pandas, Scikit-learn, and yfinance to fetch and analyze data.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Overview
The goal of this project is to predict the future price of crude oil based on historical price data. The model is trained using a Random Forest Regressor and achieves a Mean Squared Error (MSE) of **43.75**.

---

## Features
- Fetches historical crude oil price data using `yfinance`.
- Preprocesses the data and creates features like moving averages and price changes.
- Trains a machine learning model using Scikit-learn.
- Visualizes the actual vs. predicted prices using Matplotlib.

---

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RayaYasmin1/Energy-Price-Prediction.git
   cd Energy-Price-Prediction
   python oil_price_prediction.py
##Set up a virtual environment (optional but recommended):
 python -m venv venv
 source venv/bin/activate  # On Windows: venv\Scripts\activate
 
