import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt



def func():
    return lambda x: x[0]**2 + x[1]**2

def f(x):
    return x[0]**2 + x[1]**2

def constraints():
    return ({'type': 'ineq', 'fun': lambda x: -0.5 + x[0]},
            {'type': 'ineq', 'fun': lambda x: x[0] + x[1] + 1},
            {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1},
            {'type': 'ineq', 'fun': lambda x: 9*x[0]**2 + x[1]**2 - 9},
            {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]**2 - x[0]})

x0 = (1,1)
            
print("Solution with initial point " + str(x0))
solution = minimize(func(), x0, method = 'SLSQP', constraints = constraints(), options={'disp': True})
print()
print(solution)
print()

import numdifftools as nd

print("Solution with Jacobian and initial point " + str(x0))
solution = minimize(func(), x0, method = 'SLSQP', constraints = constraints(), jac = nd.Gradient(func()), options={'disp': False})
print()
print(solution)
print()

#PLOT IN 3D
yl = np.arange(-10, 10, 0.7)
xl = np.arange(-10, 10, 0.7)
X, Y = np.meshgrid(xl, yl)
Z = np.array(f((X, Y)))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

results = []
x1_results = []
x2_results = []

results = np.append(results, solution.fun)
x1_results = np.append(x1_results, solution.x[0])
x2_results = np.append(x2_results, solution.x[1])


#Plot the mins of the different inital guesses
for i in range(len(results)):
    ax.scatter(x1_results[i], x2_results[i], results[i], color="red")
    

# Plot a 3D surface
ax.plot_surface(X, Y, Z, cmap = "cool")

#plt.show()

vector = [(-3,-3), (-2,-2), (-1,-1), (0,0), (1,1), (3,3)]
for point in vector:
    hessian = nd.Hessian(f)
    H = hessian(point)
    print("Matrix for point ", point, ": ")
    print(H)
    print()