# Linear Regression Model with R-Squared Error

# Equation for a line: y = mx + b
# where:
# m => Slope,  b => y intercept

# m = (mean(x).mean(y) - mean(x.y)) / (mean(x)**2 - mean(x**2))
# b = mean(y) - m.mean(x)

# Equation for R-Squared Error or Coefficient of Determination
# r2 = 1 - (SqError(y_hat) / SqError(mean(y)))



# Import Dependencies
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random



# Generate Random Data
def generateDataset(num_values, variance, step_size, correlation=False):
    val = 1
    y = []
    for i in range(num_values):
        u = val + random.randrange(-variance, variance)
        y.append(u)
        if correlation and correlation == 'pos':
            val += step_size
        elif correlation and correlation == 'neg':
            val -= step_size
    x = [i for i in range(len(y))]
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


# Function to Calculate Slope of the Line: 'm'
# and to Calculate the Y Intercept: 'b'
def bestFitSlope_m_and_y_Intercept(x,y):
    # Calculate Numerator
    num = (mean(x)*mean(y)) - (mean(x*y))
    # Calculate Denominator
    den = (mean(x)**2) - (mean(x**2))
    # Calculate Slope of Line
    m = (num)/(den)
    # Calculate the Y Intercept for the Line
    b = mean(y) - (m * (mean(x)))
    return m,b


# Function to Calculate Squared Error
# Squared Error: Sum of squared distance of data points from Regression Line / Mean Line
def Squared_Error(y,y_hat):
    return sum((y_hat-y)**2)


# Function to Calculate R-Squared Error
def coeff_of_Determination(y,y_hat):
    # Mean of all data points forms the mean line
    mean_line = mean(y)
    # Squared Error between Regression Line and Data Points
    y_hat_SquaredError = Squared_Error(y,y_hat)
    # Squared Error between Mean Line and Data Points
    y_mean_SquaredError = Squared_Error(y,mean_line)
    return 1 - (y_hat_SquaredError/y_mean_SquaredError)




if __name__ == '__main__':
    # Generate Random Data
    # Decrease the Variance to get better R-Squared Value
    x,y = generateDataset(num_values=40, variance=20, step_size=2, correlation = 'pos')

    # Slope of Line
    m,b = bestFitSlope_m_and_y_Intercept(x,y)
    print('Slope of line: ',m)

    # Y Intercept
    print('Y Intercept: ',b)

    # Predicted Output: "Best Fit Line" or "Regression Line"
    y_hat = m * x + b
    print('y_hat: ',y_hat)

    # R-Squared Error: Higher the output value, more better the Regression Line
    r_squaredError = coeff_of_Determination(y,y_hat)
    print('R-Squared Error: ',r_squaredError)

    fig,ax = plt.subplots()
    ax.scatter(x,y)
    ax.plot(y_hat)
    plt.title('Linear Regression: Best Fit Line')
    plt.show()

# ------------------ EOC ---------------------