---
layout: post
title: "[Pytorch] pytorch tutorial 따라하기 - 신경망 모델 구성하기"
categories: [Pytorch]
tags: [Pytorch, tutorial]
---

# 신경망 모델 구성하기
이번에는 파이토치(Pytorch)를 이용한 신경망 모델을 구성해보곘습니다.
신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(Module)로 구성되어 있습니다.
`torch.nn`은 신경망을 구성하는데 필요한 모든 구성 요소를 제공합니다. PyTorch의 모든 모듈은 `nn.Module`의 하위 클래스(subclass)입니다. 신경망은 다른 모듈(계층; layer)로 구성된 모듈입니다.

동일하게 FashionMNIST dataset을 이용해 이미지들을 분류하는 신경망을 구성해보도록 하겠습니다.
```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from otrchvision import datasets, transforms
```

우선 학습하기 전 GPU/CPU setting부터 하겠습니다.
```python
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using {device} device')
```

## 클래스(class) 정의하기
신경망 모델을 `nn.Module`의 하위클래스로 정의하고, `__init__`에서 신경망 계층들을 초기화합니다.
`nn.Module`을 상속받은 모든 클래스는 `forward` 메소드에 입력 데이터에 대한 연산들을 구현합니다.
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

`NeuralNetwork`의 인스턴스(instance)를 생성하고 이를 `device`로 이동한 뒤, 구조(structure)를 출력합니다.

```python
model = NeuralNetwork().to(device)
print(model)
```
![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/e4a2320a-113b-4a28-b9d1-200950c7079f)

해당 모델을 사용하기 위해 입력 데이터를 전달합니다.

```python
X = torch.rand(1, 28, 28, device = device)
logits = model(X)
pred_probab=nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f'Predicted class: {y_pred}')
```
![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/eb32cf2e-599c-4b16-b05a-22ade6085c47)

## 모델 계층(Layer)
이번에는 모델의 계층들을 살펴보겠습니다.
```python
input_image = torch.rand(3, 28, 28)
print(input_image.size())
```

## nn.Flatten
`nn.Flatten` 계층을 초기화하여 각 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환합니다.

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

## nn.Linear
`nn.Linear`는 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈입니다.
```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
pritn(hidden1.size())
```

## nn.ReLU
비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듭니다. 비선형 활성화는 선형 변환 후에 적용되어 비선형성(nonlinearitu)을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 돕습니다.
```python
print(f'Before ReLU: {hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLU: {hidden1}')
```

## nn.Sequential
`nn.Sequential`은 순서를 갖는 모듈의 컨테이너입니다. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됩니다. 순차 컨테이너(Sequential Container)를 사용하여 아래의 `seq_modules`와 같은 신경망을 빠르게 만들 수 있습니다.
```python
seq_models = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
```

## nn.Softmax
신경망의 마지막 선형 계층은 `nn.Softmax` 모듈에 전달될 [$-\infty$,$\infty$] 범위의 raw value인 logits를 반환합니다.
logits는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록[0, 1] 범위로 조정(scale)됩니다.
```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```