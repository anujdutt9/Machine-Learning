# Linear Regression: Fit a best fit line to the data

import pandas as pd
import quandl
import numpy as np
import math

# Preprocessing used to normalize or scale the data
# CrossValidation: To shuffle data and divide data into Training and Validation data
from sklearn import preprocessing, model_selection, svm

# Linear Regression
from sklearn.linear_model import LinearRegression

# Sklearn Metrics to see how accurate the Classification is
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score



# Load Google Stock prices as a DataFrame (Table)
df = quandl.get('WIKI/GOOGL')
print(df.head())
print('Raw Stock Data: \n',df.describe())

# df.to_csv('Google-Stock.csv')

# Select useful features from the given features
# Create a new data frame with selected features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# High Low Percent Change
df['HighLow_Percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

# Calculate the percent change in stock prices
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Create a New dataframe with the required features
# Features:
df = df[['Adj. Close','HighLow_Percent','Percent_Change','Adj. Volume']]
print(df.head())
print('Feature Selected Data: \n',df.describe())

forecast_col = 'Adj. Close'

# Replace the "NaN" data with some value
df.fillna(-99999, inplace=True)

# Predict stock prices for next 10 days (0.1 => 10 days)
forecast_out = int(math.ceil(0.01*len(df)))

# print('Forecast out: ',forecast_out)

# Labels:
# Shifted columns of forecast_col -vely (upwards)
# Labels are the Adj. Closed Prices in the future
df['label'] = df[forecast_col].shift(-forecast_out)

# df.to_csv('Labels.csv')
df.dropna(inplace=True)
print(df.head())

# Features
X = np.array(df.drop(['label'],1))

# preprocessing.scale(): Scale the data such that final data has "zero mean" and "unit variance"
X = preprocessing.scale(X)

# Labels
y = np.array(df['label'])

df.dropna(inplace = True)
print(len(X), len(y))

# Divide data into Training and Test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = LinearRegression()

clf = svm.SVR()

# Train the Classifier on Training Data
clf.fit(X_train, y_train)

# Find the Predicted Outputs
pred = clf.predict(X_test)
print('Predicted Output: \n', pred)

# Compute Accuracy of Classifier
accuracy = clf.score(X_test, y_test)
print('\nAccuracy: ',accuracy)