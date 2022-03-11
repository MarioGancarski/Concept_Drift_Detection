# Concept_Drift_Detection : Supervised and Unsupervised Concept Drift Detection Methods

by Mario Gancarski, 2022.
First work : re-implementing two existent algorithms in order to understand how they work out.

**Sample run :**

```
from DDM import DDM
from D3 import D3
from datasets import import_dataset
import matplotlib.pyplot as plt

X, y = import_dataset('datasets/ELEC.csv')
window_size = 100
rho = 0.1
threshold = 0.51

d3 = D3(X, y, window_size, rho, threshold)
ddm = DDM(X, y)

error_rate_list, accuracy = d3.run_all_steps()
print(accuracy)
plt.plot(error_rate_list, color = 'blue')
plt.show()

```



