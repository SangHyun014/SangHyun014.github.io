---
layout: post
title: "[Pytorch] 파이토치(PyTorch)배우기 - 예제"
categories: [Pytorch]
tags: [Pytorch, Example]
---

# 텐서(Tensor)
Numpy를 사용하여 신경망을 구성해보도록 하겠습니다.

```python
import numpy as np
import math

# 무작위로 입력과 출력 데이터를 생성
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 무작위로 가중치를 초기화
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계 : 예측값 y를 계산합니다.
    # y = a + bx+ cx^2+ dx^3
    y_pred = a + b*x + c*x**2 +d*x**3

    # 손실(loss)를 계산하고 출력합니다.
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # 손실(loss)에 따른 a, b, c, d의 변화도 (gradient)를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치를 계산합니다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y ={a}, {b} x + {c} x^2 + {d} x^3')
```

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/567a1a62-bbd4-4d54-b900-f1b85d8ae14c)

이번에는 동일한 코드를 Torch를 이용해서 작성해보겠습니다.
```python
import torch
import math

dtype = torch.float
device = torch.device('cpu')
#device = torch.device('cuda:0') #GPU 사용을 하려면 해당 device를 사용

# 무작위로 입력과 출력 데이터 생성
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 무작위로 가중치를 초기화
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계 : 예측값 y를 계산합니다.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 손실(loss)를 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 a, b, c, d의 변화도(gradient)를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치를 갱신합니다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.itme()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```
![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/53a4b960-14ff-4f46-889b-27406915a71c)

## Autograd
이번에는 동일한 과정을 자동미분(Autograd)를 이용하여 진행해보겠습니다.
`Autograd`를 사용하면, 신경망의 순전파 단계에서 **연산 그래프**(**Computational Graph**)를 정의하게 됩니다. 이 그래프의 노드(node)는 텐서(tensor)이고, 엣지(edge)는 입력 텐서로부터 출력 텐서를 만들어내는 함수가 됩니다.

```python
import torch
import math

dtype = float
device = torch.device('cpu')
# device = torch.device('cuda:0') # GPU 사용을 하려면 앞 주석을 제거합니다.

# 입력값과 출력값을 갖는 텐서들을 생성합니다.
# requires_grad = False 가 기본값으로 설정되어 역전파 단계 중에 이 텐서들에 대한 변화도(gradient)를 계산할 필요가 없음을 나타냅니다.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치를 갖는 임의의 텐서를 생ㅇ성합니다.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계 : 텐서들 간의 연산을 사용하여 예측값 y를 계산합니다.
    y_pred = a + b * x + c * x **2 + d * x ** 3

    # 손실(loss)를 계산합니다. 이 때, 손실은 (1,) shape를 갖는 텐서입니다.
    # loss.item()으로 손실이 갖는 스칼라 값을 가져올 수 있습니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd를 사용해 역전파를 계산합니다. 이는 requires_grad=True를 갖는 모든 텐서들에 대한 손실의 변화도(gradient)를 계산합니다.
    loss.backward()

    # 경사하강법(gradient descent)을 사용하여 가중치를 직접 갱신합니다.
    # torch.no_grad()로 감싸는 이유는, 가중치들이 requires_grad=True이지만 autograd에서는 이를 추적하지 않을 것이기 때문입니다.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 가중치 갱신 이후에는 변화도를 0으로 만듦
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/c2534af5-9ac4-4328-9d61-54a8e574924a)

## nn Module 이용하기
이번에는 파이토치(PyTorch)의 nn Module을 이용해보겠습니다.

신경망을 구성하는 것을 종종 연산을 **계층**(Layer)에 배열하는 것으로 생각하는데, 이 중 일부는 학습 도중 최적화가 될 **학습 가능한 매개변수**를 갖고 있습니다.

텐서플로우(Tensorflor)에서는 `Keras`와 `TensorFlow-Slim`,`TFLearn` 같은 패키지들이 연산 그래프를 고수준(high-level)으로 추상화하여 제공하므로 신경망을 구축하는데 유용합니다.

파이토치(PyTorch)에서는 `nn` 패키지가 동일한 목적으로 제공됩니다. `nn` 패키지는 신경망 계층(Layer)과 거의 비슷한 **Module**의 집합을 정의합니다. Module은 입력텐서를 받고 출력 텐서를 계산하는 한편, 학습 가능한 매개변수를 갖는 텐서들을 내부 상태(internal stage)로 갖습니다. `nn` 패키지는 또한 신경망을 학습시킬 때 주로 사용하는 유용한 손실 함수(loss function)들도 정의하고 있습니다.

```python
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 이번 예제에서는 출력 y는 (x, x^2, x^3) 의 선형 함수이므로, 선형 계층 신경망으로 간주할 수 있습니다.
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
# 위 코드에서 x.unsqueeze(-1)은 (2000, 1)의 shape을, p가 (3,) shapeㅇ을 가지므로 브로드캐스트(broadcast)가 적용되어 (2000, 3)의 shape을 갖는 텐서가 됩니다.

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction = 'sum')

learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계 전 변화도(gradient)를 0으로 만듭니다.
    model.zero_grad()

    loss.backward()

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다.
    # 각 매개변수는 텐서이므로, 이전처럼 변화도에 접근할 수 있습니다.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# list의 첫번째 항목에 접근하는 것처럼 `model`의 첫번째 계층에 접근할 수 있습니다.
linear_layer = model[0]

print(f'Result y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

```

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/12ddd802-961d-486f-927b-adf52a3b45ea)

## Pytorch: optim
지금까지는 `torch.no_grad()`로 학습 가능한 매개변수를 갖는 텐서들을 직접 조작하여 모델의 가중치(weight)를 갱신하였습니다. 이것은 확률적 경사하강법(SGD, Stochastic Gradient Descent)와 같은 간단한 최적화 알고리즘에서는 크게 부담이 되지 않지만, 실제로 신경망을 학습할 때는 AdaGrad, RMSProp, Adam 등과 같은 더 정교한 옵티마이저(Optimizer) 를 사용합니다.

이번에는 RMSProp Algorithm 을 사용하겠습니다.
```python
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# nn 패키지를 사용하여 model 과 loss function을 정의합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# RMSProp를 사용하겠습니다.
learning_rate = 1e-6
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # 순전파 단계 : 모델에 x를 전달하여 prediction y를 계산합니다.
    pred_y = model(xx)

    # 손실(loss)를 계산하고 출력합니다.
    loss = loss_fn(pred_y, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/fcd76734-2a9a-4f5d-b6a9-c973ebeb1c9a)

## PyTorch : 사용자 정의 nn.Module
더 복잡한 모델을 구성해야할 때 사용합니다. 

```python
import torch
import math

class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x **2 + self.d * x ** 3
    
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Polynomial3()

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    pred_y = model(x)

    loss = loss_fn(pred_y, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```
