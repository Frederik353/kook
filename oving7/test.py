import matplotlib.pyplot as plt
import numpy as np


# Function for the ODE
def f(t, y):
    return -2 * t * y  # dy/dt = -2ty


# Exact solution for the ODE
def exact_solution(t):
    return np.exp(-t**2)  # y(t) = e^(-t^2)


# Implementing the RK4 method
def rk4(f, y0, t0, t_end, h):
    t_vals = np.arange(t0, t_end + h, h)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = y0

    for i in range(1, len(t_vals)):
        t_n = t_vals[i - 1]
        y_n = y_vals[i - 1]
        k1 = f(t_n, y_n)
        k2 = f(t_n + h / 2, y_n + h / 2 * k1)
        k3 = f(t_n + h / 2, y_n + h / 2 * k2)
        k4 = f(t_n + h, y_n + h * k3)  # Corrected

        y_vals[i] = y_n + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_vals, y_vals


# Plotting function and convergence order verification
def plot():
    y0 = 1
    t0 = 0
    t_end = 2  # Adjusted for a reasonable range to see the decay
    h_vals = [0.1, 0.05, 0.025, 0.0125]
    max_errors = []

    plt.figure(figsize=(10, 6))

    for h in h_vals:
        t_vals, y_vals = rk4(f, y0, t0, t_end, h)
        exact_y_vals = exact_solution(t_vals)
        max_err = np.linalg.norm(y_vals - exact_y_vals, ord=np.inf)
        max_errors.append(max_err)

        plt.plot(t_vals, y_vals, label=f"RK4 h={h}")

    # Calculate and plot the exact solution
    t_fine = np.linspace(t0, t_end, 1000)
    plt.plot(t_fine,
             exact_solution(t_fine),
             label="Exact Solution",
             color='black',
             linewidth=2)

    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('4th Order Runge-Kutta Method vs Exact Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

    orders = []
    for i in range(1, len(max_errors)):
        order = np.log(max_errors[i - 1] / max_errors[i]) / np.log(
            h_vals[i - 1] / h_vals[i])
        orders.append(order)

    print("Max Errors for each step size:", max_errors)
    print("Estimated Orders of Convergence:", orders)


plot()
