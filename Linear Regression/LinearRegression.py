# Linear Regression Code from Scratch
# 
# In this tutorial, we will be writing the code from scratch for Linear Regression using the First Approach that we studied i.e. using the R-Squared Error and then we will move on to plotting the "Best Fit Line" using Gradient Descent.
# So, let's get started.
# 
# ### Step-1: Import the Dependencies.
# Mean: to calculate the mean of the data points
# 
# Numpy: for numerical calculations
# 
# Matplotlib: to plot the data
# 
# Pandas: to load the data and modify it


# Import Dependencies
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# We discussed in the starting tutorials that a straight line is represented by the equation: 
# y = mx + b
# where,
# m: Slope of the line
# b: bias
# 
# 
# Step2: Fit a line to Input Data
# 
# Also, we studied that to find out the "Best Fit Line" we require the values of m and b. So, to find the best fit line, we used the formula:
# 
# Slope(m) = (mean(x)*mean(y) - mean(x*y)) / (mean(x)^2 - mean(x^2))
# Bias(b) = mean(y) - m*mean(x)
# 
# and then put these values in the equation of a straight line to get the new values for y or what we called in the tutorial as y_hat. 

# In[50]:

# Equation for a Straight Line:  y = mx + b
# Function to predict the Best Fit Slope
# Slope(m) = (mean(x)*mean(y) - mean(x*y))/(mean(x)^2 - mean(x^2))
# Bias(b) = mean(y) - m*mean(x)

def best_fit_slope(X,y):
    slope_m = ((mean(X)*mean(y)) - mean(X*y))/(mean(X)**2 - mean(X**2))
    bias_b = mean(y) - slope_m*mean(X)
    return slope_m, bias_b


# In the above function, we have calculated the hard coded values for slope(m) and the bias(b). Now, let's input our data and see how this function performs to form a "Best Fit Line".
# 
# 
# Step-3: Load Dataset
# For this code, we will be taking the "Swedish Insurance Dataset". This is a very simple dataset to start with and involves predicting the total payment for all the claims in thousands of Swedish Kronor (y) given the total number of claims (X). This means that for a new number of claims (X) we will be able to predict the total payment of claims (y).
# 
# Let's load our data and have a look at it.


# Load the data using Pandas

df = pd.read_csv('dataset/Insurance-dataset.csv')



# Let's have a look at the data, what it looks like, how many data points are there in the data.

print(df.head())


# Data is in the form of two columns, X and Y. X is the total number of claims and Y represents the claims in thousands of Swedish Kronor.
 
# Now, let's describe our data.

df.describe()


# So, both the columns have equal number of data points. Hence, the dataset is stable. No need to modify the data. We also get the mean, max values in both columns etc.
 
# Now, let's put the data in the form to be input to the function we just defined above.

# Load the data in the form to be input to the function for Best Fit Line

X = np.array(df['X'], dtype=np.float64)
y = np.array(df['Y'], dtype=np.float64)


# Step-4: Scatter Plot of Input Data
# Before going any further, let's first plot our data and see if it's linear or not. Remember, we require linear data to plot a best fit line and neglect the Outliers.
# So, let's plot the data.



# Scatter Plot of the Input Data

fig,ax = plt.subplots()
ax.scatter(X,y)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Input Data Scatter Plot')
plt.show()


# From the above plot, we can see that the data is pretty linear except for 1 or 2 points. But that's ok. We can work with that.
 
# Now, lets use the function we defined above and find the values for m and b to get the line that best fits the data.


m,b = best_fit_slope(X,y)

print('Slope: ',m)
print('Bias: ',b)


# Step-5: Calculate Values of y_hat
# Using these values of m and b, we can now find out the values of y which we call as y_hat and when we plot them with X, we get our line.


# Calculate y_hat

y_hat = m*X + b

print('y_hat: ', y_hat)


# Step-6: Plot Line of Best Fit
# Now, we have got the values for y_hat, m and b. Now, we can go ahead and plot the line that fits the data.


# Scatter Plot of the Input Data with the line fit on Input Data

fig,ax = plt.subplots()
ax.scatter(X,y)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.plot(X,y_hat)
ax.set_title('Line fit to Input Data')
plt.show()


# Step-7: Find the Squared Error and R-Squared Error
# So, in the above plot we get the plot of the line fit to input data points. But is this line really a good fit line ??
# Now, we will write a function to check the accuracy of this line and see how really best fit is this line.
# So, how to proceed from here. Remember, we talked about the Square Error and the R-Squared Error equations which help to provide the accuracy for the so called best fit line. Let's implement these functions.
# So, what is the equation for Squared Error? 
 
# Squared Error = sum((y_hat - y)^2) 

# Let's implement this.


# Squared Error
# Squared Error = sum((y_hat-y)^2)
def squared_Error(y, y_hat):
    return sum((y_hat - y)**2)


# Now, we have our squared error. Next we want to calculate the R-Squared Error using Squared Error function. So, what was the formula for this.

# R-Squared Error  =  1 - ((Squared_Error(y_hat)) / (Squared_Error(mean(y))))
 
# So, let's implement it.

# R-Squared Error

def r_squared(y,y_hat):
    # Mean of all data points forms the mean line
    mean_line = mean(y)
    
    # Squared Error between Regression Line (y_hat) and the data points
    y_hat_Squared_Error = squared_Error(y,y_hat)
    
    # Squared Error between Mean Line and data points
    y_Squared_Error = squared_Error(y,mean_line)
    
    return 1 - (y_hat_Squared_Error/y_Squared_Error)


# So, now since we have implemented both the functions, it's time to evaluate our results. Let's see how accuate our current line is.

print('R-Squared Error: ', r_squared(y,y_hat))


# So, the R-Squared error is 0.833. But wait a minute. In the tutorial we explained that R-Squared Error is actually the accuracy of the line. So what is our error ??

# Error%

error = 1 - r_squared(y,y_hat)
print('Error% : ', error*100)


# Accuracy%
print('Accuracy% : ', r_squared(y,y_hat)*100)


# So, our line is 83.33% accurate, which is pretty great but we would still like to improve the accuracy.
