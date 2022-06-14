from cvxpy import *
import numpy as np

x = Variable(2, name = "x")

f1 = power(x[0] - 1,2) + power(x[1] - 1, 2) # Define both constraints of the problem
f2 = power(x[0] - 1,2) + power(x[1] + 1, 2)

constraints = [f1 <= 1., f2 <= 1.]

f0 = power(x[0], 2) + power(x[1], 2) # Define the objective function
obj = Minimize(f0)

prob = Problem(obj, constraints) # Create the model of the problem

print("Solution " + str(prob.solve()))  # Returns the optimal value.
print("Status: " + str(prob.status))
print("Optimal value p* = " + str(prob.value))
print("Optimal var: x1 = " + str(x[0].value) + " x2 = " + str(x[1].value))
print("Optimal dual variables lambda1 = " + str(constraints[0].dual_value))
print("Optimal dual variables lambda2 = " + str(constraints[1].dual_value))