import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def costFunctionJ(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    j = 1 / (2 * m) * np.sum(sqrErrors)
    return j

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)
    temp = np.zeros((n, num_iters))
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X.dot(theta)
        temp[:, i] = theta - (alpha / m) * X.T.dot(h - y).reshape(-1)
        theta = temp[:, i]
        j_history[i] = costFunctionJ(X, y, theta)
    return theta, j_history

# 从CSV文件加载数据
data = pd.read_csv('airfoil_noise_samples.csv')

# 提取特征和目标变量列
X = data[['Frequency', 'Angle', 'Displacement', 'Chord length', 'Velocity', 'Thickness']].values
y = data['Sound Pressure'].values

# 添加偏置列（x0），全为1
X = np.column_stack((np.ones(len(X)), X))

# 初始化theta
theta = np.zeros(X.shape[1])

# 求代价函数值
j = costFunctionJ(X, y, theta)
# print('代价值：', j)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 1], y, c='r', label='real data')
plt.plot(X[:, 1], X.dot(theta), label='test data')
plt.legend(loc='best')
plt.title('before')

theta, j_history = gradientDescent(X, y, theta, 0.0001, 1000)
print('最终j_history值：\n', j_history[-1])
print('最终theta值：\n', theta)
print('每次迭代的代价值：\n', j_history)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 1], y, c='r', label='real data')
plt.plot(X[:, 1], X.dot(theta), label='predict data')
plt.legend(loc='best')
plt.title('after')

# 添加回归曲线
x_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_predict = theta[0] + theta[1] * x_range
plt.plot(x_range, y_predict, label='regression line', c='g')

plt.show()
