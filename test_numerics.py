from numerics import *
from matplotlib import pyplot as plt

def f(y, t):
    return -2 * t * y**2

def g(x):
    return 1 / (x**2 + 1)

y0 = 1
t0 = 0
tN = 1
N = 10

x = np.linspace(0, 1, 101)
plt.plot(x, g(x), label="exact")

solveODE(f, y0, (t0, tN))

plt.plot(t, y, label=f"")

plt.legend()
plt.show()


def f(y, t):
    predator = 0.1 * y[0] * y[1] - 1 * y[0]
    prey = 4 * y[1] - 1 * y[0] * y[1]
    return np.array([predator, prey])

y0 = np.array([3, 5])
t0 = 0
tN = 10
N = 100

t, y =
plt.plot(t, y.T[0], label="predator")
plt.plot(t, y.T[1], label="prey")

plt.legend()
plt.show()