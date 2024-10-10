import matplotlib.pyplot as plt
import numpy as np


# Function definition
def u(x):
    return x * np.cos(x)


def u_prime_exact(x):
    return np.cos(x) - x * np.sin(x)


h_values = [0.1 * (2**-i) for i in range(10)]
approx_values = []
errors = []

# Computing the approximations and errors
for h in h_values:
    u_at_x = u(np.pi / 2)
    u_at_x_h = u(np.pi / 2 - h)
    u_at_x_2h = u(np.pi / 2 - 2 * h)

    u_prime_approx = (3 * u_at_x - 4 * u_at_x_h + u_at_x_2h) / (2 * h)
    approx_values.append(u_prime_approx)

    error = abs(u_prime_exact(np.pi / 2) - u_prime_approx)
    errors.append(error)

# Plotting
plt.loglog(h_values, errors, marker='o')
plt.xlabel('Step Size h')
plt.ylabel('Error |e(h)|')
plt.title('Convergence Plot of Numerical Differentiation Error')
plt.grid(True)
plt.show()
