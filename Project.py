import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
dataframe = pd.read_csv("D:\\INTERNSHIP\\50_Startups.csv")

# Checking if the dataset contains null values
if dataframe.isnull().values.any():
    print(dataframe.isnull().sum())
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)

