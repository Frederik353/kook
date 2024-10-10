import matplotlib.pyplot as plt
import numpy as np

k1 = 100
k2 = 200
m1 = 10
m2 = 5


def derivatives(t, x):
    x1, x2, x3, x4 = x
    dx1_dt = x2
    dx2_dt = (-k1 * x1 + k2 * (x3 - x1)) / m1
    dx3_dt = x4
    dx4_dt = (-k2 * (x3 - x1)) / m2
    return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])


def eulers(derivatives, t_range, x0, h):
    t_values = np.arange(t_range[0], t_range[1] + h, h)
    x_values = np.zeros((len(t_values), len(x0)))
    x_values[0] = x0

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + h * derivatives(
            t_values[i - 1], x_values[i - 1])

    return t_values, x_values


def heuns(derivatives, t_range, x0, h):
    t_values = np.arange(t_range[0], t_range[1] + h, h)
    x_values = np.zeros((len(t_values), len(x0)))
    x_values[0] = x0

    for i in range(1, len(t_values)):
        x_prev = x_values[i - 1]
        t_prev = t_values[i - 1]

        x_pred = x_prev + h * derivatives(t_prev, x_prev)

        avg_slope = (derivatives(t_prev, x_prev) +
                     derivatives(t_prev + h, x_pred)) / 2
        x_values[i] = x_prev + h * avg_slope

    return t_values, x_values


x0 = [0, 1, 0, 1]
t_range = (0, 3)
step_lengths = [0.1, 0.01, 0.001]

plt.figure(figsize=(15, 10))

for h in step_lengths:
    t_euler, x_euler = eulers(derivatives, t_range, x0, h)

    t_heun, x_heun = heuns(derivatives, t_range, x0, h)

    plt.subplot(2, 1, 1)
    plt.plot(t_euler, x_euler[:, 0], label=f'Euler, h={h}', linestyle='++')
    plt.plot(t_heun, x_heun[:, 0], label=f'Heun, h={h}')
    plt.title("Displacement of m1 (u) over time")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_euler, x_euler[:, 2], label=f'Euler, h={h}', linestyle='--')
    plt.plot(t_heun, x_heun[:, 2], label=f'Heun, h={h}')
    plt.title("Displacement of m2 (v) over time")
    plt.xlabel("Time")
    plt.ylabel("v")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
