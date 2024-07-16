import torch

print("Trial 1: with python float")
w = torch.randn(3,5,requires_grad = True) * 0.01

x = torch.randn(5,4,requires_grad = True)

y = torch.matmul(w,x).sum(1)
w.retain_grad()
y.backward(torch.ones(3))

print("w.requires_grad:",w.requires_grad)
print("x.requires_grad:",x.requires_grad)

print("w.grad",w.grad)
print("x.grad",x.grad)

print("Trial 2: with on-the-go torch scalar")
w = torch.randn(3,5,requires_grad = True) * torch.tensor(0.01,requires_grad=True)

x = torch.randn(5,4,requires_grad = True)

y = torch.matmul(w,x).sum(1)

y.backward(torch.ones(3))

print("w.requires_grad:",w.requires_grad)
print("x.requires_grad:",x.requires_grad)

print("w.grad",w.grad)
print("x.grad",x.grad)

print("Trial 3: with named torch scalar")
t = torch.tensor(0.01,requires_grad=True)
w = torch.randn(3,5,requires_grad = True) * t

x = torch.randn(5,4,requires_grad = True)

y = torch.matmul(w,x).sum(1)

y.backward(torch.ones(3))

print("w.requires_grad:",w.requires_grad)
print("x.requires_grad:",x.requires_grad)

print("w.grad",w.grad)
print("x.grad",x.grad)
