# Linear Regression using Gradient Descent

# In this tutorial, we will be writing the code from scratch for Linear Regression using the Second Approach that we studied i.e. using Gradient Descent and then we will move on to plotting the "Best Fit Line".
# So, let's get started.


# Import Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# Load the data using Pandas
df = pd.read_csv('dataset/Insurance-dataset.csv')


# Let's have a look at the data, what it looks like, how many data points are there in the data.
print(df.head())


# Data is in the form of two columns, X and Y. X is the total number of claims and Y represents the claims in thousands of Swedish Kronor.
# Now, let's describe our data.

# Describe the Data
df.describe()


# Load the data in the form to be input to the function for Best Fit Line
X = np.array(df['X'], dtype=np.float64)
y = np.array(df['Y'], dtype=np.float64)


# Cost Function
def cost_Function(m,b,X,y):
    return sum(((m*X + b) - y)**2)/(2*float(len(X)))



# Gradient Descent
# X,y: Input Data Points
# m,b: Initial Slope and Bias
# alpha: Learning Rate
# iters: Number of Iterations for which we need to run Gradient Descent.

def gradientDescent(X,y,m,b,alpha,iters):
    # Initialize Values of Gradients
    gradient_m = 0
    gradient_b = 0
    # n: Number of items in a row
    n = float(len(X))
    a = 0
    # Array to store values of error for analysis
    hist = []
    # Perform Gradient Descent for iters
    for _ in range(iters):
        # Perform Gradient Descent
        for i in range(len(X)):
            gradient_m = (1/n) * X[i] * ((m*X[i] + b) - y[i])
            gradient_b = (1/n) * ((m*X[i] + b) - y[i])
        m = m - (alpha*gradient_m)
        b = b - (alpha*gradient_b)
        # Calculate the change in error with new values of "m" and "b"
        a = cost_Function(m,b,X,y)
        hist.append(a)
    return [m,b,hist]





# Main Function
if __name__ == '__main__':
    # Learning Rate
    lr = 0.0001

    # Initial Values of "m" and "b"
    initial_m = 0
    initial_b = 0

    # Number of Iterations
    iterations = 900

    print("Starting gradient descent...")

    # Check error with initial Values of m and b
    print("Initial Error at m = {0} and b = {1} is error = {2}".format(initial_m, initial_b, cost_Function(initial_m, initial_b, X, y)))


    # Run Gradient Descent to get new values for "m" and "b"
    [m,b,hist] = gradientDescent(X, y, initial_m, initial_b, lr, iterations)


    # New Values of "m" and "b" after Gradient Descent
    print('Values obtained after {0} iterations are m = {1} and b = {2}'.format(iterations,m,b))


    # Calculating y_hat
    y_hat = (m*X + b)
    print('y_hat: ',y_hat)

    # Testing using arbitrary Input Value
    predict_X = 76
    predict_y = (m*predict_X + b)
    print('predict_y: ',predict_y)

    # Plot the final Outptu
    fig,ax = plt.subplots(nrows=1,ncols=2)
    ax[1].scatter(X,y,c='r')
    ax[1].plot(X,y_hat,c='y')
    ax[1].scatter(predict_X,predict_y,c='r',s=100)
    ax[0].plot(hist)
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('y')
    ax[1].set_title('Best Fit Line Plot')
    ax[0].set_title('Cost Function Over Time')
    plt.show()
