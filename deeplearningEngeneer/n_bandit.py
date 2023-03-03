import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 手法を定義する関数です
def randomselect():
    slot_num = np.random.randint(0, 5)
    return slot_num

# 環境を定義する関数です
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数です
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record


# 初期変数を設定しています
times = 10000
results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
record = np.zeros(times)
print(record)

# slot_numを取得して、results,recordを書き換えてください
for time in range(0, times):
    slot_num = randomselect()
    results, record = reward(record, results, slot_num, time)

# 各マシーンの試行回数と結果を出力しています
print(results)

# recordを用いて平均報酬の推移をプロットしてください
plt.plot(np.cumsum(record) / np.arange(1, record.size + 1))

# 表を出力しています
plt.xlabel("試行回数")
plt.ylabel("平均報酬")
plt.title("試行回数と平均報酬の推移")
plt.show()
