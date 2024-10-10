import numpy as np

f = lambda t, y: 2 / t * y

t0, tend = 1, 2
y0 = 1
N = 10
h = (tend - t0) / N

y = np.zeros(N + 1)
t = np.zeros(N + 1)

y[0] = y0
t[0] = t0

for n in range(N):
    k1 = f(t[n], y[n])
    k2 = f(t[n] + 0.5 * h, y[n] + 0.5 * h * k1)
    k3 = f(t[n] + 0.75 * h, y[n] + 0.75 * h * k2)
    y[n + 1] = y[n] + h * (2 * k1 + 3 * k2 + 4 * k3) / 9
    t[n + 1] = t[n] + h

print('t= ', t)
print('y= ', y)
