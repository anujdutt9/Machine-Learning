

# Logistic Regression using Scikit-Learn

# In this tutorial, we will implement Logistic Regression using Scikit-Learn. 
# So, let's get started. 
# 
# Step-1 Import Dependencies

# numpy: 
# for numerical calculations
# 
# pandas: 
# for data modelling and access data from files
# 
# sklearn.preprocessing: 
# Preprocessing data, scaling data points to bring all data points to same scale for easy calculations.
# 
# sklearn.model_selection:  
# for cross validation, divide data into testing and training set.
# 
# sklearn.linear_model.LogisticRegression: 
# Logistic Regression Classifier.
# 
# sklearn.metrics.accuracy_score: 
# to calculate accuracy score of the classifier.
# 
# matplotlib:  
# to plot data


# Import Dependecies

import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Step-2: Load Dataset
# 
# Now that we are ready with our dependecies, it's time to load our dataset.
# 
# This time we'll be using "Wisconsin Breast Cancer Dataset". You can download the dataset from the dataset folder from the Github repository or download from this link [https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)]
# 
# Let's have a look at our data.
# 
# This dataset has the following attributes:
# 1. Sample code number            id number
# 2. Clump Thickness               1 - 10
# 3. Uniformity of Cell Size       1 - 10
# 4. Uniformity of Cell Shape      1 - 10
# 5. Marginal Adhesion             1 - 10
# 6. Single Epithelial Cell Size   1 - 10
# 7. Bare Nuclei                   1 - 10
# 8. Bland Chromatin               1 - 10
# 9. Normal Nucleoli               1 - 10
# 10. Mitoses                       1 - 10
# 11. Class:                        (2 for benign, 4 for malignant)
# 
# It has 11 features. These features contribute together to tell whether the tumor is "Malignant : 4" or "Benign : 2".
# 
# Let's load the data and get some insight into it.



# Load Data

df = pd.read_csv('dataset/breast-cancer-wisconsin-data.csv')


# Let's have a look at the data
# Let's print first 5 row values from data.

print(df.head())


# Step-3: Adding Names to Dataset
# 
# As we can see that the data does not have any name for the column. So, either we can add the names to the data in csv file itself or go the second way.
# 
# I will be explaining the second way here. Let's see.


# Load data with Column names
# Here we provide the names for each Column.

df = pd.read_csv('dataset/breast-cancer-wisconsin-data.csv', names=['id', 'clump_thickness','unif_cell_size',
                                                                           'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
                                                                           'bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses','class'])



# Let's check the data again

print(df.head())


# Perfect. Now we atleast have some names to refer to while processing the data.
# 
# Step-4 Feature Selection
# 
# Now, since we have a lot of features i.e. each column is a feature. Hence, we need to find out which features are the most informative i.e. which features selected will lead to maximizing accuracy.
# 
# * Correlation:
# Let's check the correlation between the data points of different features.



# Correlation between different features
correlation = df.corr()
print(df.corr())


# Let's see this in a heatmap

plt.figure(figsize=(15, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# We can clearly see that when we take the correlation of features, almost all the Correlation coefficients are positive values but only "id" has a negative correlation with all the featues in the data.
# 
# So, this shows that, the "id" column is of no use to us. Also, removing this will not affect the accuracy at all.
# 
# So, we will remove the "id" column from the dataset. This is what we call "Feature Selection".


# Feature Selection

df.drop(['id'], 1, inplace=True)

# Filtered Dataset
print(df.head())


# Are we all done ?? Well not yet. This dataset contains empty or NaN values which have been repressented as "?". We have two options: Either replace the empty spaces by the Mean or Median value of the Column or replace the empty value with a very large negative value i.e. an outlier or remove the entire row consisting of empty value.
# 
# I'll go with the second option. Why replace it with an outlier value you ask ?? Well, we went over this in Linear Regression discussion and saw that the outliers are neglected. So, let's do it.


# Replace empty values with Outlier value

df.replace('?', -99999, inplace=True)


# Well we are all done with the basic Data Modelling. Next step is to form the features and labels and train the Classifier.
# 
# 
# Step-5: Train the Classifier
# 
# So, let's train the Classifier. Firstly, we need to define the features and the labels.
# 
# Features: All column values except the "Class" which we need to predict i.e. "Malignant" or "Benign"


# Features
X = np.array(df.drop(['class'],1))

# Labels
y = np.array(df['class'])


# Let's have a look at our Features and Labels
print('Features: \n',X)
print('Labels: \n',y)


# Perfect. We are now ready to do the Cross-Validation step.


# Cross Validation
# Test Data: 20% of total data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)


# Now, we have the training and testing set. It's time to Train the Classifier and test the predictions.


# Define the Classifier
clf = linear_model.LogisticRegression(n_jobs = 10)

# Train the Classifier
clf.fit(X_train,y_train)


# Well, now we have a trained Classifier, we need to check out the predictions for the Test Data and then see how accurate the predictions are.
# 
# So, let's get started.


# Predcitions on Test Data
y_pred = clf.predict(X_test)
print('Predicted Labels: ',y_pred)


# We have the true labels i.e. y_test and the predicted labels y_pred. Now is a good time to see that how good our Classifier is i.e. how many labels did the Classifier Misclassify or classify correctly. So, let's calculate this.

# Number of Misclassified Labels
print('Number of Misclassified Labels: {}'.format((y_pred != y_test).sum()))


# Well our Classifier Misclassified only "6" values. But out of how many ??

print('Total Predicted Values: ',len(y_pred))


# Woww !! Out of 140 values, only 6 values were Misclassified by our Trained Classifier. That's pretty impressive. Let's calculate the confidence and accuracy score for this.

# Confidence
confidence = clf.score(X_test, y_test)
print('Confidence of Classifier: ',confidence)


# So, our Classifier is 95% confident of its predictions. That is what we saw above where only 6 values were misclassified from 140. Pretty good.

# Accuracy Score
acc = accuracy_score(y_test,y_pred)
print('Accuracy Score of Classifier: ',acc)


# Let's check the Confusion Matrix to see the number of True positive and False Positives.

# Confusin Matrix
conf = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n',conf)


# These values show that we have 91 True Positive values, 5 False Positive Values. Also, we have 1 False Negative Value and 43 True negative Values predicted.
