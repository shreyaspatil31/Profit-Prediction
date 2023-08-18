# Exposys-Profit-Prediction
Explored regressions predicting profits using R&amp;D, Admin, and Marketing data. Streamlit app offers dynamic model selection &amp; instant predictions. Metrics guide optimal algorithm choice for precise profit estimation.


# Profit Prediction Web App README

This repository contains a Streamlit web application that predicts company profits using regression algorithms based on R&D Spend, Administration Cost, and Marketing Spend data. The project also assesses model performance and offers a practical solution for businesses aiming to refine resource allocation and strategic decision-making.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Algorithm Choices](#algorithm-choices)
5. [Evaluation Metrics](#evaluation-metrics)
6. [User Input](#user-input)
7. [Predicted vs. Actual Values](#predicted-vs-actual-values)

## Getting Started

Clone this repository to your local machine to access the Streamlit web application for profit prediction.

## Installation

1. Install the required packages using the following command:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

2. Download the dataset ("50_Startups.csv") and place it in the root directory of the repository.

## Usage

1. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

2. Open a web browser and navigate to the provided link (usually http://localhost:8501).

## Algorithm Choices

Select an algorithm for profit prediction from the available options:

- Linear Regression
- Gradient Boosting Regressor
- k-Nearest Neighbors Regressor
- Decision Tree Regressor
- Elastic Net Regression

## Evaluation Metrics

Upon algorithm selection, the app computes and displays the following evaluation metrics for the chosen model:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

## User Input

Provide input values for the following features:

- R&D Spend
- Administration
- Marketing Spend

Enter numeric values to obtain an instant profit prediction based on the selected model.

## Predicted vs. Actual Values

The app also visualizes the predicted profit values versus the actual profit values using a scatter plot. Blue points represent predicted profits, while red points represent actual profits.

---

Feel free to customize this readme as needed and add any additional information that you think would be relevant for users.
