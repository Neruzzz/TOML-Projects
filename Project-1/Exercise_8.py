from cvxpy import *


x = Variable(3, name='x') # Define the two variables
r = Variable(3, name='r')

f1 = x[0] + x[1] # Define all the constraints of the model
f2 = x[0]
f3 = x[2]
f4 = r[0] + r[1] + r[2]

constraints = [f1 <= r[0], f2 <= r[1], f3 <= r[2], f4 <= 1]

f0 = sum(log(x)) # Define the objective function
prob = Problem(Maximize(f0), constraints) # Create the model

print("Solution", prob.solve())  # Returns the optimal value.
print("Status:", prob.status)
print("Optimal value p* = ", prob.value)
print("Optimal var: x1 = ", x[0].value)
print("Optimal var: x2 = ", x[1].value)
print("Optimal var: x3 = ", x[2].value)
print("Optimal var: r12 = ", r[0].value)
print("Optimal var: r23 = ", r[1].value)
print("Optimal var: r32 = ", r[2].value)
print("Optimal dual variables lambda1 = ", constraints[0].dual_value)
print("Optimal dual variables lambda2 = ", constraints[1].dual_value)
print("Optimal dual variables lambda3 = ", constraints[2].dual_value)
print("Optimal dual variables lambda4 = ", constraints[3].dual_value) # μ1