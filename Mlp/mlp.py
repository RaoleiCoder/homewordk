import torch
import torch.nn as nn
import torch.optim as optim

# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 定义训练数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 初始化模型
input_dim = 2
hidden_dim = 4
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss() # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练模型
epochs = 10000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, Y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 1000 == 0:
         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions = model(test_data)
    print(predictions)
