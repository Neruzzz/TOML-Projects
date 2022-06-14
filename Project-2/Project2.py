from variables import *
from gpkit import *
import matplotlib.pyplot as plt

### Sampling frequency
Fs = 0
def fs(time):
    return 1/(time*60*1000)   # e.g. Min traffic rate 1 pkt/half_hour = 1/(60*30*1000) pk/ms

# Sleep period: Parameter Bounds
Tw_max  = 500.       # Maximum Duration of Tw in ms
Tw_min  = 100.       # Minimum Duration of Tw in ms
Tw0 = np.linspace(Tw_min, Tw_max)


# Definition of Id
def Id(d):
    if d == D:
        return 0
    elif d == 0:
        return C
    else:
        return (2*d + 1)/(2*d -1)

# Definition of Fout
def Fout(d):
    if d == D:
        return Fs
    else:
        return Fs*((D**2 - d**2 + 2*d -1)/(2*d - 1))


# Definition of FI
def FI(d):
    if d == 0:
        return Fs*(D**2)*C
    else:
        return Fs*((D**2 - d**2)/2*d - 1)

# Definition of FB
def FB(d):
    return (C - np.abs(Id(d)))*Fout(d)

# Definition of the alphas
def alphas(d):
    alpha1 = Tcs + Tal + (3/2)*Tps*((Tps + Tal)/2 + Tack + Tdata)*FB(d)
    alpha2 = Fout(d)/2
    alpha3 = ((Tps + Tal)/2 + Tcs + Tal + Tack + Tdata)*Fout(d) + ((3/2)*Tps + Tack + Tdata)*FI(d) + (3/4)*Tps*FB(d)
    return alpha1, alpha2, alpha3

# Definition of the betas
def betas(d):
    beta1 = sum([1/2]*D)
    beta2 = sum([Tcw/2 + Tdata]*d)
    return beta1, beta2


########## 1. a) Plot the energy as a function of Tw ##########
def energy(Tw):
    d = 1
    alpha1, alpha2, alpha3 = alphas(d)
    return alpha1/Tw + alpha2*Tw + alpha3

time = [5, 10, 15, 20, 25, 30]

for minutes in time:
    Fs = fs(minutes)
    label = str(round(1/minutes, 3)) + "pkt/min"
    plt.plot(Tw0, energy(Tw0), label = label)
plt.title("Energy vs Tw plot")
plt.xlabel("Tw in ms")
plt.ylabel("Energy in J")
plt.legend()
plt.show()

########## 1. b) Plot delay as a function of Tw ##########
def delay(Tw):
    d = D
    beta1, beta2 = betas(d)
    return beta1*Tw + beta2

plt.plot(Tw0, delay(Tw0))
plt.xlabel("Tw in ms")
plt.ylabel("Delay in ms")
plt.title("Delay function vs Tw plot")
plt.show()

########## 1. c) Plot the curve E-L ##########
Tw0 = np.linspace(0, Tw_max)
for minutes in time:
    Fs = fs(minutes)
    label = str(round(1/minutes, 3)) + "pkt/min"
    plt.plot(energy(Tw0), delay(Tw0), label = label)
plt.title("Delay vs Energy plot")
plt.xlabel("Energy in J")
plt.ylabel("Delay in ms")
plt.legend()
plt.show()

########## 2. OPTIMIZATION AND GRAPH ##########
L = [500, 750, 1000, 2500, 5000]
E = np.linspace(Emax,Emin)
Tw0 = np.linspace(Tw_min, Tw_max)


SolutionsP1 = []
SolutionsP2 = []

time = [5, 10, 15, 20, 25, 30]

for l in L:
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for minutes in time:
        Fs = 1.0 / (minutes * 60 * 1000)

        x = Variable('x')

        
        alpha1, alpha2, alpha3 = alphas(1)
        beta1, beta2 = betas(D)

        f1 = beta1*x + beta2 # Define all the constraints
        f2 = x
        f3 = abs(Id(0))*(Tcs + Tal + ((x/(Tps + Tal))*((Tps + Tal)/2) + Tack + Tdata))*Fout(1)

        f01 = alpha1/x + alpha2*x + alpha3 # Objective function P1
        constraintsP1 = [f1 <= l, f2 >= Tw_min, f3 <= 1/4]
        P1 = Model(f01, constraintsP1)
        sol1 = P1.solve()
        SolutionsP1.append(sol1["cost"])
        plt.plot(Tw0, energy(Tw0), label='E(Tw) for Fs('+str(minutes)+'min)')
        ax.scatter(sol1['variables'][x], sol1['cost'], color="red")

    plt.xlabel('Tw in ms')
    plt.ylabel("Energy in J")
    plt.legend(loc='upper right')
    plt.title("Energy vs Tw with Lmax = "+str(l))
    plt.show()


E = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

for e in E:
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for minutes in time:
        Fs = 1.0 / (minutes * 60 * 1000)

        x = Variable('x')

        
        alpha1, alpha2, alpha3 = alphas(1)
        beta1, beta2 = betas(D)

        f1 = alpha1/x + alpha2*x + alpha3 # Define all the constraints
        f2 = x
        f3 = abs(Id(0))*(Tcs + Tal + ((x/(Tps + Tal))*((Tps + Tal)/2) + Tack + Tdata))*Fout(1)

        f02 =  beta1*x + beta2 # Objective function P2
        constraintsP2 = [f1 <= e, f2 >= Tw_min, f3 <= 1/4]
        P2 = Model(f02, constraintsP2)
        sol2 = P2.solve()
        SolutionsP2.append(sol2["cost"])
        plt.plot(Tw0, delay(Tw0), label='E(Tw) for Fs('+str(minutes)+'min)')
        ax.scatter(sol2['variables'][x], sol2['cost'], color="red")
        print()
        print("Delay value", sol2["cost"])
        print("Tw value", sol2["variables"])
        print()


    plt.xlabel('Tw in ms')
    plt.ylabel("Delay in ms")
    plt.legend(loc='upper right')
    plt.title("Delay vs Tw with Ebudget = "+str(e))
    plt.show()


######### 3. Nash Bargaining Scheme ##########
import cvxpy as cvx

Tw0 = np.linspace(50, 300, 100)

x = cvx.Variable(3, name='x')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


for item in L:

    Ew = 0.05
    Lw = item

    f0 = - cvx.log(Ew - x[0]) - cvx.log(Lw - x[1])

    f1 = Ew
    f2 = x[0]
    f3 = Lw
    f4 = x[1]
    f5 = x[2]
    f6 = abs(Id(0))*(Tcs + Tal + (x[2]/(Tps + Tal)*(Tps + Tal)/2 + Tack + Tdata))*Fout(1)

    constraints = [f1 >= (alpha1 * cvx.power(x[2], -1) + alpha2 * x[2] + alpha3),
                    f2 >=  (alpha1 * cvx.power(x[2], -1) + alpha2 * x[2] + alpha3),
                    f3 >= (beta1 * x[2] + beta2),
                    f4 >= (beta1 * x[2] + beta2),
                    f5 >= Tw_min,
                    f6 <= (1 / 4)]

    P3 = cvx.Problem(cvx.Minimize(f0), constraints)


    if x[0].value and x[1].value != None:
        if x[0].value <= 0.06 and x[1].value <= 800:
            ax.scatter(x[0].value, x[1].value, label = 'Tradeoff Point with Lmax=' + str(item))
            print("Optimal value = ", P3.solve())
            print("Optimal var: E1 = ", x[0].value)
            print("Optimal var: L1 = ", x[1].value)
            print("Optimal var: Tw = ", x[2].value)
            print()

plt.plot(energy(Tw0), delay(Tw0), color='b')
plt.xlabel("Energy in J")
plt.ylabel("Delay in ms")
plt.legend(loc="upper right")
plt.show()