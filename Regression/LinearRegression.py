# Simple Linear Regression Model from Scratch

# Equation for a line: y = mx + b
# where:
# m => Slope,  b => y intercept

# m = (mean(x).mean(y) - mean(x.y)) / (mean(x)**2 - mean(x**2))
# b = mean(y) - m.mean(x)


# Import Dependencies
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt



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



if __name__ == '__main__':
    # Input Data from "Diameter of Sand Granules Vs. Slope on Beach" Dataset
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

    # Slope of Line
    m,b = bestFitSlope_m_and_y_Intercept(x,y)
    print('Slope of line: ',m)

    # Y Intercept
    print('Y Intercept: ',b)

    # Predicted Output: "Best Fit Line"
    y_hat = m * x + b
    print('y_hat: ',y_hat)

    # Plotting the "Best Fit Line"
    # plt.scatter(x,y)
    # plt.plot(y_hat)
    # plt.title('Linear Regression: Best Fit Line')

    fig,ax = plt.subplots()
    ax.scatter(x,y)
    ax.plot(y_hat)
    plt.title('Linear Regression: Best Fit Line')
    plt.show()

# ------------------ EOC ---------------------