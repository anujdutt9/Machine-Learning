import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('dataset/iris-dataset.csv')

print('Dataset Description: \n',df.describe())

print('\nCorrelation:\n',df.corr())

df.dropna(inplace=True)

d = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica' : 3}
df['class'].replace(d,inplace=True)

X = np.asarray(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)

print('Number of Training Data Points: ',len(X_train))
print('Number of Training Labels: ',len(y_train))

print('\nNumber of Test Data Points: ',len(X_test))
print('Number of Test Labels: ',len(y_test))


# Classifier
clf = GaussianNB()

clf.fit(X_train,y_train)

pred = clf.predict(X_test)
print('\nPredicted Labels: ',pred)


# Number of Misclassified Labels
print('\nNumber of Misclassified Labels: {}'.format((pred != y_test).sum()))

confidence = clf.score(X_test,y_test)
print('\nClassifier Confidence: ',confidence)

accuracy = accuracy_score(y_test,pred)
print('\nAccuracy Score: ',accuracy)

conf = confusion_matrix(y_test,pred)
print('\nConfusion Matrix: \n',conf)