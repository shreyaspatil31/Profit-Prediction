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

# Split the dataset into features (X) and target (y)
X = dataframe[["R&D Spend", "Administration", "Marketing Spend"]]
y = dataframe["Profit"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=60
)

# Run Streamlit app
st.set_page_config(page_title="Profit Prediction App")

st.title("Profit Prediction App")

# Streamlit app starts here
algorithm_choice = st.selectbox(
    "Choose an Algorithm for Prediction:",
    [
        "Linear Regression",
        "Gradient Boosting Regressor",
        "k-Nearest Neighbors Regressor",
        "Decision Tree Regressor",
        "Elastic Net Regression",
    ],
)

if algorithm_choice == "Linear Regression":
    model = LinearRegression()
elif algorithm_choice == "Gradient Boosting Regressor":
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=35
    )
elif algorithm_choice == "k-Nearest Neighbors Regressor":
    model = KNeighborsRegressor(
        n_neighbors=4, weights="uniform", algorithm="auto", leaf_size=30, p=1
    )
elif algorithm_choice == "Decision Tree Regressor":
    model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2)
elif algorithm_choice == "Elastic Net Regression":
    model = ElasticNet()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"## {algorithm_choice} Metrics:")
st.write("Mean Squared Error:", mse)
st.write("Mean Absolute Error:", mae)
st.write("R-squared:", r2)

# User input section
st.write("\n## Provide Input Values:")
user_input = {}
for feature in ["R&D Spend", "Administration", "Marketing Spend"]:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_input[feature] = value

user_input_df = pd.DataFrame([user_input])
user_input_scaled = scaler.transform(user_input_df)

predicted_output = model.predict(user_input_scaled)
st.write("\n## Prediction Result:")
st.write(f"Predicted Profit: {predicted_output[0]:.2f}")

# Visualize predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted")
plt.scatter(y_test, y_test, color="red", label="Actual")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Predicted vs. Actual Profit")
plt.legend()
st.pyplot(plt)