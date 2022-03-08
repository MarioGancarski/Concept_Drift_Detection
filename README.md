# Concept_Drift_Detection : Supervised and Unsupervised Concept Drift Detection Methods

by Mario Gancarski, 2022.
First work : re-implementing two existent algorithms in order to understand how they work out.

## DDM

**Sample run :**

```
from DDM import PredictionManager
from dataset import dataset
X, y = dataset('datasets/ELEC.csv')
pm = PredictionManager(X, y)
error_rate_list, accuracy = pm.run_all_steps()
print("Accuracy :", accuracy)
plt.plot(error_rate_list, color = 'blue')
plt.show()
```



## D3

**Sample run :**

```
from D3 import *
from dataset import *
X, y = dataset('datasets/ELEC.csv')
w_size = 100
new_data_size = 10
threshold = 0.70
pm = PredictionManager(X, y, w_size, new_data_size, threshold)
accuracy = pm.run_all_steps()
print("Accuracy : ", accuracy)
```


