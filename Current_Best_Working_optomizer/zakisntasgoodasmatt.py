#The MATT method (Multi-channel Allocation for Transmission Tuning)
from gekko import GEKKO
import numpy as np
import math

# Problem parameters
K = 96                 # total channels
fidelity_limit = 0.7
N_links = 12
y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]

# Create GEKKO model. Setting remote=True uses GEKKO's online solver service.
m = GEKKO(remote=True)
m.options.IMODE = 3  # steady-state optimization mode
m.options.SOLVER = 1  # use APOPT (which supports MINLP)

# Decision variable: mu (continuous, positive)
mu = m.Var(value=0.001, lb=1e-9)

# Decision variables: k_i (integer, at least 1, and upper bounded by K)
k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=K) for i in range(N_links)]

# Constraint: Sum of channels must equal K
m.Equation(sum(k_vars) == K)

# Fidelity constraints for each link:
#  0.25*(1 + (3*mu*k_i)/(mu^2*k_i^2 + mu*k_i*(2*(y1_i+y2_i)+1) + 4*y1_i*y2_i)) >= fidelity_limit
for i in range(N_links):
    expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
    m.Equation(0.25*(1 + (3*mu*k_vars[i]) / expr) >= fidelity_limit)

# Objective: maximize the sum over links of log10(expr_i)
# Since log10(x) = ln(x)/ln(10) and logarithm is monotonic,
# maximizing the sum of ln(expr) is equivalent.
# GEKKO minimizes by default so we minimize the negative of the sum.
obj = 0
for i in range(N_links):
    expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
    obj += m.log(expr)
m.Obj(-obj)  # maximize sum(log(expr))

# Solve the MINLP
m.solve(disp=True)

# Output the solution:
print("Optimal mu =", mu.value[0])
print("Optimal channel allocation (k_i):")
for i in range(N_links):
    print(f" Link {i+1}: k = {int(round(k_vars[i].value[0]))}")

# Compute and print the objective value (in base 10)
objective_value = 0
for i in range(N_links):
    expr_val = mu.value[0]**2 * (k_vars[i].value[0])**2 \
               + mu.value[0] * (k_vars[i].value[0]) * (2*(y1[i]+y2[i]) + 1) \
               + 4*y1[i]*y2[i]
    objective_value += math.log(expr_val) / math.log(10)
print("Objective value =", objective_value)
print("Matt has successfully beaten Zak")