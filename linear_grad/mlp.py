import numpy as np

# 定义sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义sigmoid函数的导数，用于反向传播
def sigmoid_derivative(x):
    return x * (1 - x)

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 定义数据集
inputs = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

# 期望的输出
targets = np.array([[0],
                     [1],
                     [1],
                     [0]])

# 定义超参数
epochs = 10000
learning_rate = 0.1
input_size = inputs.shape[1]
hidden_size = 4
output_size = 1

# 初始化权重和偏置
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_input_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_hidden_output = np.zeros((1, output_size))

# 训练模型
for epoch in range(epochs):
    # 前向传播
    hidden_inputs = np.dot(inputs, weights_input_hidden) + bias_input_hidden
    hidden_outputs = sigmoid(hidden_inputs)

    final_inputs = np.dot(hidden_outputs, weights_hidden_output) + bias_hidden_output
    final_outputs = sigmoid(final_inputs)

    # 计算误差
    error = targets - final_outputs

    # 反向传播
    output_delta = error * sigmoid_derivative(final_outputs)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_outputs)

    # 更新权重和偏置
    weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate
    bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
    bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# 预测
new_inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

hidden_layer_activation = sigmoid(np.dot(new_inputs, weights_input_hidden) + bias_input_hidden)
output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output) + bias_hidden_output)

# 打印最终预测结果
print("Final predictions:")
print(output_layer_activation)
