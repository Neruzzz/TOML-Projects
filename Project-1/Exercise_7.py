from cvxpy import *

x = Variable(3, name = "x")
c = [1, 2, 1, 2, 1] # Channel capacity array

f1 = x[0] + x[2] # x1+x3
f2 = x[0] + x[1] # x1+x2
f3 = x[2] # x3

constraints = [f1 <= c[0], f2 <= c[1], f3 <= c[4], x[0] >= 0, x[1] >= 0, x[2] >= 0] # Define all constraints

f0 = 0
for i in range(3):
    f = cvxpy.log(x[i])
    f0 = f0 + f

obj = Maximize(f0) # Define the objective function

prob = Problem(obj, constraints) # Create the model

print("Solution " + str(prob.solve()))  # Returns the solutions and the lagrange multipliers
print("Status: " + str(prob.status))
print("Optimal value p* = " + str(prob.value))
print("Optimal var: x1 = " + str(x[0].value) + "; x2 = " + str(x[1].value) + "; x3 = " + str(x[2].value))
print("Optimal dual variables lambda1 = " + str(constraints[0].dual_value))
print("Optimal dual variables lambda2 = " + str(constraints[1].dual_value))
print("Optimal dual variables lambda3 = " + str(constraints[2].dual_value))
print("Optimal dual variables lambda4 = " + str(constraints[3].dual_value))
print("Optimal dual variables lambda5 = " + str(constraints[4].dual_value))
