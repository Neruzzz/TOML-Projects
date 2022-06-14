from scipy.optimize import newton
from sympy import Point2D

def func1():
    return lambda x: 2*x**2 - 0.5

def func2():
    return lambda x: 2*x**4 - 4*x**2 + x - 0.5

def GradientDescendMethod(cur_x, rate, precision, previous_step_size, max_iters, func):
    import numdifftools as nd
    initial = cur_x # Save the initial point to print it later
    iters = 0 #iteration counter
    df = nd.Gradient(func) #Gradient of our function 
    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - rate * df(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count
        
    print("The local minimum for initial point", initial, "occurs at", cur_x, "at iteration", iters)

print()
print("RESULTS FOR FUNCTION 1")

GradientDescendMethod(3, 0.01, 0.0001, 1, 10000, func1())

x0 = [-2, -0.5, 0.5, 2]

print()
print("RESULTS FOR FUNCTION 2 USING BACKTRACKING LINE SEARCH METHOD")

for i in range(4):
    GradientDescendMethod(x0[i], 0.01, 0.0001, 1, 10000, func2())


def f_1(x): #First derivative of the function
    return  (8 * (x ** 3)) - (8 * x) + 1

def f_2(x): #Second derivative of the function
    return  (24 * (x ** 2)) - (8 * x)

def newton_method(x0):
    g= newton(func=f_1,x0=x0,fprime=f_2, tol=0.0001, full_output=True) # Newtons method using the library scipy
    r= g[1]
    print("The local minimum for initial point", x0, "occurs at", r.root, "at iteration", r.iterations)

print()
print("RESULTS FOR FUNCTION 2 USING NEWTON'S METHOD")

for points in x0:
    newton_method(points)

print()