import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegretion import load_data, train_model, plot_data, plot_regression, evaluate_model, get_coefficients

# Title of the application
st.title("Analysis of Grades vs. Study Hours")
st.write("This application shows the relationship between study hours and grades obtained by students.")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file with the data", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Preview of the data:")
    st.write(df.head())

    # Check that the necessary columns are present
    if 'Study_hours' in df.columns and 'Grades' in df.columns:
        # Graph the original data
        st.write("### Scatter graph: Study hours vs. grades")
        fig, ax = plt.subplots()
        plot_data(df['Study_hours'], df['Grades'], 'Study hours vs. grades', 'Study hours', 'Grades')
        st.pyplot(fig)

        # Training the linear regression model
        X = df[['Study_hours']]
        y = df['Grades']
        model = train_model(X, y)

        # Predict
        y_pred = model.predict(X)

        # Graph the regression line
        st.write("### Linear Regression: Study Hours vs Grades")
        fig, ax = plt.subplots()
        plot_regression(X, y, y_pred, 'Regresión lineal: Horas de estudio vs Calificaciones', 'Study hours', 'Grades')
        st.pyplot(fig)

        # Show model metricsLinear Regression: Study Hours vs Grades
        st.write("### Model metrics")
        mse, r2 = evaluate_model(y, y_pred)
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Coefficient of Determination (R²):** {r2:.2f}")

        # Show model coefficients
        st.write("### Model coefficients")
        intercept, slope = get_coefficients(model)
        st.write(f"**Intercept (β₀):** {intercept:.2f}")
        st.write(f"**Pending (β₁):** {slope:.2f}")
    else:
        st.error("The CSV file must contain the columns 'Study_hours' y 'Grades'.")
else:
    st.write("Please upload a CSV file to get started.")
