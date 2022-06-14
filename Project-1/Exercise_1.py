from scipy.optimize import minimize
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import numdifftools as nd


def func():
    return lambda x: (math.e**x[0]) * (4 * (x[0]**2) + 2 * x[1]**2 + 4 * x[0] * x[1] + 2 * x[1] + 1)

def constraints():
    return ({'type': 'ineq', 'fun': lambda x: - x[0] * x[1] + x[0] + x[1] - 1.5},
            {'type': 'ineq', 'fun': lambda x: x[0] * x[1] + 10})


x0 = [(0,0), (10, 20), (-10, 1), (-30, -30)]

results = []
x1_results = []
x2_results = []

print("SOLUTIONS WITHOUT JACOBIAN")
print()

for i in range(4):
    print("Result when initial point is " + str(x0[i]) + " is the following:")
    print()
    start_time = time.time()
    solution = minimize(func(), x0[i], method = 'SLSQP', constraints = constraints(), options={'disp': False})
    print()
    print("--- %s seconds ---" % (time.time() - start_time))
    results = np.append(results, solution.fun)
    x1_results = np.append(x1_results, solution.x[0])
    x2_results = np.append(x2_results, solution.x[1])
    print()
    print(solution)
    print()
    print()


def f(x):
    return (math.e**x[0]) * (4 * (x[0]**2) + 2 * x[1]**2 + 4 * x[0] * x[1] + 2 * x[1]  + 1)

#PLOT IN 3D
yl = np.arange(-10, 3, 0.7)
xl = np.arange(-10, 3, 0.7)
X, Y = np.meshgrid(xl, yl)
Z = np.array(f((X, Y)))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)


#Plot the mins of the different inital guesses
for i in range(len(results)):
    ax.scatter(x1_results[i], x2_results[i], results[i], color="red")
    

# Plot a 3D surface
ax.plot_surface(X, Y, Z, cmap = "cool")

plt.show()

import numdifftools as nd
jac = nd.Gradient(func())

print("SOLUTIONS WITH JACOBIAN")
print()

for i in range(4):
    print("Result using Jacobian when initial point is " + str(x0[i]) + " is the following:")
    print()
    start_time = time.time()
    solution = minimize(func(), x0[i], method = 'SLSQP', constraints = constraints(), jac = jac, options={'disp': True})
    print()
    print("--- %s seconds ---" % (time.time() - start_time))
    print()
    print(solution)
    print()
    print()

vector = [(-3,-3), (-2,-2), (-1,-1), (0,0), (1,1), (3,3)]
for point in vector:
    hessian = nd.Hessian(f)
    H = hessian(point)
    print("Matrix for point ", point, ": ")
    print(H)
    print()