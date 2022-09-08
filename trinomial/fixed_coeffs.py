from matplotlib import pyplot as plt
import numpy as np

eps = 1e-5
def f(x: float):
    return 5.00000 * x**4 - 4.00000 * x**3 - 6.00000 * x**2 - 2.00000 * x + 2.00000

start = -2.0
stop  =  2.0
count = 100

x = np.linspace(start, stop, num=count, endpoint=True, dtype=np.float64)
y = f(x)
y_diff = np.diff(y)

fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x[:-1], y_diff)

ax.grid(True, which='both')

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

plt.show()

# from graph :
#   1 root
#   x_1 in [-1.0, 0.0]

l = 1.18019
r = 1e9
xn = (r + l) / 2

f_xn = f(xn)
while abs(f_xn) >= eps:
    if f_xn < 0:
        l = xn
    else:
        r = xn
    xn = (l + r) / 2
    f_xn = f(xn)

print(f'root {xn=:.5f}')