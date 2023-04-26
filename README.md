# CCA
Self-implemented CCA (Canonical Correlation Analysis)

## Functions
1. `pow_mat()`: Calculate the power of the matrix
2. `order_eig()`: Order eigenvalues and their corresponding eigenvectors
3. `cca_population()`: Conanical Correlation Analysis for population
4. `cca_sample()`: Conanical Correlation Analysis for sample

## Example
```Python
import pandas as pd
from cca import cca_sample

# read data
df = pd.read_csv("stiffness.dat", delim_whitespace=True, header=None, dtype="float64")
X, Y = df.iloc[:, :2], df.iloc[:, 2:4]

# do cca with samples
info, combinations = cca_sample(X, Y)

# rho, a, b, E, Var = info.values()
# U, V = combinations.values()
```