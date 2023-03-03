from sklearn.datasets import make_blobs
import numpy as np
import scipy

dataset = make_blobs(centers=2)
features = dataset[0]
labels = dataset[1]
test_data = np.random.randint(-9, 9, size=(1, 2))
print("features")
print(features)
print("labels")
print(labels)
print("test_data")
print(test_data)

k = 3
diff = np.linalg.norm(test_data - features, axis=1)
print("diff")
print(diff)
# 最も近いk個のラベルを抽出
nearest_labels = labels[np.argsort(diff)[:k]]
print("nearest_labels")
print(nearest_labels)
# 最も多い数含まれるラベルを抽出
predict = scipy.stats.mode(nearest_labels)[0]

print(predict)