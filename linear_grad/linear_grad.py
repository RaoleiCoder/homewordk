import numpy as np
import matplotlib.pyplot as plt

# 步骤 2: 生成随机数据集
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 步骤 3: 初始化参数
theta = np.random.randn(2, 1)

# 步骤 4: 定义代价函数
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    print(cost)
    return cost

# 步骤 5: 定义梯度下降函数
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / m
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# 步骤 6: 构建输入矩阵 X，并添加一列全为 1，以便计算截距
X_b = np.c_[np.ones((100, 1)), X]

# 设置学习率和迭代次数
learning_rate = 0.01
iterations = 1000

# 调用梯度下降函数
theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# 步骤 7: 画出散点图与回归线
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X, X_b.dot(theta), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
