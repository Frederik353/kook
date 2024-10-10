import matplotlib.pyplot as plt
import numpy as np


def total_energy(x, m1, m2, k1, k2):
    x1, x2, x3, x4 = x  # x1 = u, x2 = u', x3 = v, x4 = v'
    kinetic_energy = 0.5 * m1 * x2**2 + 0.5 * m2 * x4**2
    potential_energy = 0.5 * k1 * x1**2 + 0.5 * k2 * (x1 - x3)**2
    return kinetic_energy + potential_energy


energies_euler = {}
energies_heun = {}

plt.figure(figsize=(15, 10))

for h in step_lengths:
    energies_euler[h] = [total_energy(x, m1, m2, k1, k2) for x in x_euler]
    energies_heun[h] = [total_energy(x, m1, m2, k1, k2) for x in x_heun]

plt.figure(figsize=(15, 10))

for h in step_lengths:
    plt.subplot(2, 1, 1)
    plt.plot(t_euler, energies_euler[h], label=f'Euler, h={h}', linestyle='--')
    plt.title("Energy Conservation Euler's")
    plt.xlabel("Time")
    plt.ylabel("Total Energy (E)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_heun, energies_heun[h], label=f'Heun, h={h}')
    plt.title("Energy Conservation  Heun's")
    plt.xlabel("Time")
    plt.ylabel("Total Energy (E)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
