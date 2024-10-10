# Error Analysis

import matplotlib.pyplot as plt
# Import necessary packages
import numpy as np
from numpy import pi


# Define the cardinal function (to be completed)
def cardinal(xdata, x):
    """
    cardinal(xdata, x): 
    In: xdata, array with the nodes x_i.
        x, array or a scalar of values in which the cardinal functions are evaluated.
    Return: l: a list of arrays of the cardinal functions evaluated in x. 
    """
    n = len(xdata)  # Number of evaluation points x
    l = []
    for i in range(n):  # Loop over the cardinal functions
        li = np.ones_like(x)
        for j in range(n):  # Loop to make the product for l_i
            if i != j:
                li *= (x - xdata[j]) / (xdata[i] - xdata[j])
        l.append(li)  # Append the array to the list
    return l


# Define the Lagrange interpolation function
def lagrange(ydata, l):
    """
    lagrange(ydata, l):
    In: ydata, array of the y-values of the interpolation points.
         l, a list of the cardinal functions, given by `cardinal(xdata, x)`
    Return: An array with the interpolation polynomial (evaluated at `x`). 
    """
    poly = 0
    for i in range(len(ydata)):
        poly += ydata[i] * l[i]
    return poly


# Define the function of interest (to be completed)
def f(x):
    # return np.sin(x)  # Example function
    return 2**(x**2 - 4 - x)


# Data for the final plot and for
a = -5  # Define interval start
b = 5  # Define interval end
x = np.linspace(a, b, 401)  # Generate x values
y = f(x)  # Evaluate the function
n = 10  # Number of nodes (example)

# **a)** Use the given (equidistant) nodes to compute our polynomial p
xdata = np.linspace(a, b, n + 1)  # Equidistant nodes
ydata = f(xdata)  # Function values at the nodes

# Compute the cardinal function
l = cardinal(xdata, x)

# Compute the polynomial
p = lagrange(ydata, l)

# Compute the error and its maximum
e_p = np.abs(y - p)
e_p_max = np.max(e_p)
print(f"The maximal error is {e_p_max:.2e}")

# Plot f and p - and for illustration, the interpolation points
plt.plot(x, y, label='f(x)')
plt.plot(x, p, label='p(x)')
plt.plot(xdata, ydata, 'o', label='Interpolation points')
plt.legend()

# plt.show()


# **b)** Compute the Chebyshev nodes and their function values
def chebyshev_nodes(a, b, n):
    """
    chebyshev_nodes(a, b, n)
    
    return $n+1$ Chebyshev nodes $x_0,\ldots,x_{n-1}$ on $[a,b]$
    """
    i = np.array(range(n))
    x = np.cos((2 * i + 1) / (2 * n) * pi)  # Nodes over the interval [-1,1]
    return 0.5 * (a + b) + 0.5 * (b - a) * x  # Nodes over the interval [a,b]


xcheb = chebyshev_nodes(a, b, n + 1)
ycheb = f(xcheb)
print(xcheb)
print(ycheb)

# **c)** Perform the same analysis as in b, interpolate and create g(x) and compute the error
l = cardinal(xcheb, x)
g = lagrange(ycheb, l)
e_g = np.abs(y - g)
e_g_max = np.max(e_g)
print(f"The maximal error is {e_g_max:.2e}")

# Plot f, g - and for illustration, the interpolation points
plt.plot(x, y, label='f(x)')
plt.plot(x, g, label='g(x)')
plt.plot(xcheb, ycheb, 'o', label='Chebyshev points')
plt.legend()
# plt.show()

# Plot the error (on the x-axis, i.e. with zero error) for both the equidistant and the Chebyshev nodes
z = np.zeros(n + 1)
z_cheb = np.zeros(n + 1)
plt.plot(xdata, z, 'o', label='Equidistant nodes')
plt.plot(xcheb, z_cheb, 'o', label='Chebyshev nodes')
plt.legend()
# plt.show()
