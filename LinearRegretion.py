import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(X, y):
  model = LinearRegression()
  model.fit(X, y)
  return model

def plot_data(X, y, title, xlabel, ylabel):
    plt.scatter(X, y, color='blue', label='Data points')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_regression(X, y, y_pred, title, xlabel, ylabel):
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def evaluate_model(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2

def get_coefficients(model):
    return model.intercept_, model.coef_[0]

