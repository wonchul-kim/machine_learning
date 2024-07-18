import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity


    
# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# 입력 데이터와 타겟 데이터 정의
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y = torch.tensor([[0.], [0.], [0.]])

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# forward pass
output = model(x)
loss = criterion(output, y)

# backward pass
optimizer.zero_grad()
loss.backward()

# Gradient 확인
for param in model.parameters():
    print('1. param: ', param)
    print(param.grad)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    optimizer.step()

prof.export_chrome_trace('optimizer_trace.json')

# Gradient 확인
for param in model.parameters():
    print('2. param: ', param)
