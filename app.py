import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
st.title("Stock Price Prediction")
st.write("This app predicts stock prices using a simple linear regression model.")
option =st.selectbox("Select a stock dataset 1 manually or 2 upload your own dataset", ("1", "2"))
if option == "1":
    # Sample dataset
    data = pd.read_csv("stock.csv")
    X = data[['Open_Price','High_Price','Low_Price','Volume']]
    y = data['Close_Price']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)

    # Manual Input
    open_price = st.number_input("Open Price")
    high_price = st.number_input("High Price")
    low_price = st.number_input("Low Price")
    volume = st.number_input("Volume")

    if st.button("Predict"):

        input_data = pd.DataFrame({
            "Open_Price":[open_price],
            "High_Price":[high_price],
            "Low_Price":[low_price],
            "Volume":[volume]
        })

        prediction = model.predict(input_data)

        st.success(f"Predicted Close Price: {prediction[0]:.2f}")
        st.success("Model trained successfully!")
elif option == "2":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())
        st.write("Dataset Summary:")
        st.write(data.describe())
        st.write("Correlation Heatmap:")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)
        st.write("Scatter Plot of Open vs Close Price:")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Open_Price', y='Close_Price', data=data)
        st.pyplot(plt)
        st.write("Training Linear Regression Model...")
        x = data[['Open_Price', 'High_Price', 'Low_Price', 'Volume']]
        y = data['Close_Price']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train, y_train)
        st.success("Model trained successfully! You can now use the input fields to predict stock prices.")


