import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.01)


def f_step():
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    x = list(range(100))
    y = []
    for epoch in range(100):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print (epoch, lr)
        y.append(lr)
    
    return x, y

def f_multistep():
    scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 80], gamma=0.98)
    x = list(range(100))
    y = []
    for epoch in range(100):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print (epoch, lr)
        y.append(lr)
    
    return x, y

def f_exp():
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    x = list(range(100))
    y = []
    for epoch in range(100):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print (epoch, lr)
        y.append(lr)
    
    return x, y

def f_cosine():
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,eta_min=4e-08)
    x = list(range(100))
    y = []
    for epoch in range(100):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print (epoch, lr)
        y.append(lr)
    
    return x, y


x, y = f_step()


plt.figure()
plt.plot(x, y)
plt.show()
