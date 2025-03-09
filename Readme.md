# Project: Simple Linear Regression

This project shows how to use simple linear regression to predict students' grades as a function of study hours.

## Project Structure
- SyntheticDataSet.py`: Generate a synthetic dataset (`grades.csv`).
- LinearRegression.py`: Calculates linear regression and graphs the results.
- app.py`: Interactive web application to visualise the data and model results.

## Install features

- python -m venv myenv
- pip install Flask
- myenv\Scripts\activate
- pip install pandas
- pip install numpy
- pip install matplotlib
- pip install scikit-learn
- pip install streamlit


## How to Run the Project
1. Generate the synthetic dataset:
   ````bash
   python SyntheticDataSet.py

2. Deploy App.
   ````bash
   streamlit run app.py


