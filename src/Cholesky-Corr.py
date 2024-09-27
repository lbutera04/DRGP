import numpy as np
from scipy.stats import pearsonr

# Parameters
n = 100  # Number of data points
a = 2    # Intercept
b = 3    # Slope
desired_correlation = 0.8

# Step 1: Generate the original series X
x = np.random.uniform(low=-10, high=10, size=n)

# Step 2: Define the regression parameters a and b
# Already defined above

# Step 3: Generate the new series Y with added noise
# Create the covariance matrix
cov_matrix = np.array([[1.0, desired_correlation], [desired_correlation, 1.0]])
print(cov_matrix)
# Perform Cholesky decomposition
L = np.linalg.cholesky(cov_matrix)
print(L)

# Generate uncorrelated random series
uncorrelated = np.random.normal(size=(n, 2))

# Correlate the random series using the Cholesky decomposition matrix
correlated = np.dot(uncorrelated, L.T)

# Extract the noise component
noise = correlated[:, 1]

# Combine the deterministic part with the noise
y = a + b * x + noise

# Step 5: Verify the correlation
correlation, _ = pearsonr(x, y)
print(f"Desired Correlation: {desired_correlation}, Achieved Correlation: {correlation}")

# Optionally, plot the results
import matplotlib.pyplot as plt

plt.scatter(x, y, alpha=0.5)
plt.title(f"Scatter plot of Y vs. X with correlation {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

import re

# Prompt for input
newcoln = input('List new correlated feature names, separated by commas: ')
print(newcoln)

# Perform the substitution
newcoln = re.sub(r'\,(\s)', ',', newcoln)
print(newcoln)

x.size
plt.scatter(x, uncorrelated, alpha=0.5)
plt.title(f"Scatter plot of Y vs. X with correlation {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

# plt.scatter(x, y, alpha = 0.5)
# plt.show()

corr_mat = np.array([[1.0, 0.7], [0.7, 1.0]])
L = np.linalg.cholesky(corr_mat)

# tau = np.diag(np.array([1, 3]))

# lamb = tau@L.T

Z = np.array([x, y])

X = L@Z

# plt.scatter(X[0], X[1], alpha = 0.5)
# plt.show()

correlation, _ = pearsonr(X[0], X[1])
print(f"Desired Correlation: {0.7}, Achieved Correlation: {correlation}")


X = np.array()
corrcoef = []
for i in range(2000):
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)

    corr_mat = np.array([[1.0, 0.7], [0.7, 1.0]])
    L = np.linalg.cholesky(corr_mat)

    Z = np.array([x, y])

    X = L@Z

    correlation, _ = pearsonr(X[0], X[1])
    corrcoef.append((correlation))

plt.hist(corrcoef, alpha = 0.5)
plt.show()

plt.scatter(X[0], X[1], alpha = 0.5)
plt.show()








x = np.random.normal(500, 150, 1000)
y = np.random.normal(0, np.std(x), 1000)

corr_mat = np.array([[1.0, 0.7], [0.7, 1.0]])
L = np.linalg.cholesky(corr_mat)

Z = np.array([x, y])

X = L@Z

correlation, _ = pearsonr(X[0], X[1])
print(f"Desired Correlation: {0.7}, Achieved Correlation: {correlation}")



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 1000  # Number of samples
z = np.random.normal(0, 1, (3, n))

# Step 2: Define the desired correlation matrix
corr_mat = np.array([
    [1.0, 0.9, 0.7],
    [0.9, 1.0, 0.3],
    [0.7, 0.3, 1.0]
])

L = np.linalg.cholesky(corr_mat)

# Step 3: Apply the Cholesky decomposition
X = L @ z

ax.scatter(X[0], X[1], X[2], marker='o')
plt.show()


from sympy import sympify, symbols
expr = "x**2 + 3*x + 2"
expr1 = sympify(expr)
expr1
expr1.subs('x', 2)



from scitools.StringFunction import StringFunction

f = StringFunction('1+V*sin(w*x)*exp(-b*t)', independent_variables=('x', 't'))
f.set_parameters(V=0.1, w=1, b=0.1)

result = f(1.0, 1.0)
print(result)

import sympy as sp
from sympy import symbols, Eq, simplify
equation = "3*x + 4*y - 2*z + 3*x*z + 5"
variables = "y x z"

var_symbols = symbols(variables)
eq = simplify(equation)
coeffs = {str(var): eq.coeff(var) for var in var_symbols}
print(coeffs)


import sympy as sp

def get_equation():
    user_input = input("Enter an equation (e.g., '2*x**2 + 3*x + 4 = 0'): ")
    lhs, rhs = user_input.split('=')
    equation = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
    return equation

def extract_coefficients(equation, var):
    poly = sp.Poly(equation.lhs - equation.rhs, var)
    return poly.all_coeffs()

def main():
    equation = get_equation()
    print(f"Parsed Equation: {equation}")

    # Assuming the variable is x
    x = sp.symbols('x')
    
    coefficients = extract_coefficients(equation, x)
    print(f"Coefficients: {coefficients}")
    
    # Solve the equation
    solutions = sp.solve(equation, x)
    print(f"Solutions: {solutions}")

if __name__ == "__main__":
    main()
