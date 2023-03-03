import numpy as np
from sklearn.datasets import make_blobs
import scipy
test_data = np.random.randint(-9, 9, size=(1, 2,))
print(test_data)

dataset = make_blobs(centers=3)
print(dataset)