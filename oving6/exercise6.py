import matplotlib.pyplot as plt
import numpy as np


def u(x):
    return x * np.cos(x)


h_values = [0.1 * 2**(-i) for i in range(5)]

approximations = []
errors = []

exact = -np.pi / 2

for h in h_values:
    approx = (3 * u(np.pi / 2) - 4 * u(np.pi / 2 - h) +
              u(np.pi / 2 - 2 * h)) / (2 * h)

    error = abs(approx - exact)
    print("------------------")
    print(error)
    print(approx)

    approximations.append(approx)
    errors.append(error)

plt.figure(figsize=(16, 9))
plt.title('Convergence Plot of Finite Difference Approximation')
plt.loglog(h_values, errors, marker='o', label='Error')
plt.axhline(y=1 / 600 * np.pi,
            color='r',
            linestyle='--',
            label='Target Error |e(h)|')
plt.xlabel('Step Size (h)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()
