from cvxpy import *

x = Variable(1, name = "x") # Definition of the variable x


f1 = power(x, 2) - 6*x + 8 # Definition of the constraint that the model has
constraints = [f1 <= 0]

f0 = power(x, 2) + 1 # Definition of the Objective function
obj = Minimize(f0)

prob = Problem(obj, constraints) # Creation of the model

print("Solution " + str(prob.solve()))  # Returns the results
print("Status: " + str(prob.status))
print("Optimal value p* = " + str(prob.value))
print("Optimal var: x = " + str(x.value))
print("Optimal dual variables lambda1 = " + str(constraints[0].dual_value))

