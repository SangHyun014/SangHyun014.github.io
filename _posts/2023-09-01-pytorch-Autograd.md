---
layout: post
title: "[Pytorch] pytorch tutorial 따라하기 - 모델 매개변수 최적화하기"
categories: [Pytorch]
tags: [Pytorch, tutorial]
---

# 모델 매개변수 최적화하기
모델과 데이터를 준비하고 데이터의 매개변수를 최적화하여 하여 모델을 학습하고, 검증하고, 테스트 해보겠습니다. 모델을 학습하는 과정은 반복적인 과정을 거칩니다. 각 반복 단계에서 모델은 출력을 추측하고, 추측과 정답 사이의 오류(loss)를 계산하고, 매개변수에 대한 오류의 도함수(derivative)를 수집한 뒤, 경사하강법을 사용하여 파라미터들을 **최적화(optimize)**합니다.

## 기본 코드
앞선 튜토리얼들의 코드들을 가져와 작성합니다.
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

## 하이퍼파라미터(Hyperparameter)
하이퍼파라미터(Hyperparameter)는 모델 최적화 과정에서 제어할 수 있는 매개변수입니다. 서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율(Convergence rate)에 영향을 미칠 수 있습니다.

학습 시 다음과 같은 하이퍼 파라미터들을 정의합니다.
- 에폭(epoch) : 데이터셋을 반복하는 횟수
- 배치 크기(batch size) : 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
- 학습률(learning rate) : 각 배치/에폭에서 모델의 매개변수를 조절하는 비율.

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 최적화 단계(Optimization Loop)
최적화 단계를 통해 모델을 학습하고 최적화할 수 있습니다. 최적화 단계의 각 반복(iteration)을 **에폭(epoch)** 이라 부릅니다.

하나의 에폭은 다음 두 부분으로 구성됩니다.
- 학습 단계(train loop) : 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴합니다.
- 검증/테스트 단계(validation/test loop) : 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)합니다.

## 손실 함수(Loss Function)
손실 함수(Loss Function)는 획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarity)를 측정하며, 학습 중에 이 값을 최소화하려고 합니다. 주어진 데이터 샘플을 입력으로 계산한 예측과 정답(label)을 비교하여 손실(loss)를 계산합니다.

> 일반적으로 회귀 문제(regression task)에 사용하는 `nn.MSELoss`(평균 제곱 오차(MSE; Mean Square Error)), 분류 문제(classification task)에 사용하는 `nn.NLLLoss(음의 로그 우도(Negative Log Likelihood))`, 그리고 `nn.LogSoftmax`와 `nn.NLLoss`를 합친 `nn.CrossEntropyLoss` 등이 있습니다.

```python
# 손실 함수(Loss Function)을 초기화 합니다.
loss_fn = nn.CrossEntropyLoss()
```

## 옵티마이저(Optimizer)

최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정입니다. **최적화 알고리즘**은 이 과정이 수행되는 방식(SGD; Stochastic Gradient Descent)을 정의합니다. 모든 최적화 절차는 `optimizer` 객체에 캡슐화(encapsulate)됩니다. 여기서는 SGD 옵티마이저를 사용하고 있으며, PyTorch에는 ADAM이나 RMSProp과 같은 다른 종류의 모델과 데이터에서 잘 동작하는 다양한 옵티마이저(Optimizer)가 있습니다.

학습하려는 모델의 매개변수와 학습률(learning rate) 하이퍼파라미터를 등록하여 옵티마이저를 초기화합니다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

학습 단계(loop)에서 최적화는 세단계로 이루어집니다.
- `optimizer.zero_grad()`를 호출하여 모델 매개변수의 변화도를 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.
- `loss.backwards()`를 호출하여 예측 손실(prediction loss)을 역전파합니다. PyTorcch는 각 매개변수에 대한 손실의 변화도를 저장합니다.
- 변화도를 계산한 뒤에는 `optimizer.step()`을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정합니다.

## 전체 구현

최적화 코드를 반복하여 수행하는 `train_loop`와 테스트 데이터로 모델의 성능을 측정하는 `test_loop`를 정의하였습니다.
```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        #역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

```
손실 함수와 옵티마이저를 초기화하고 `train_loop`와 `test_loop`에 전달합니다.
모델의 성능 향상을 알아보기 위해 자유롭게 에폭(epoch) 수를 증가시켜 볼 수 있습니다.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f'Epoch {t+1}\n------------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print('Done!')
```

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/f71477fc-312c-44f2-9ddd-2ba10916ae2c)

다음과 같이 학습이 진행이 되고, 테스트도 진행해 볼 수 있습니다.
마지막으로 그렇게 학습된 모델을 저장과 불러오는 방법은 다음과 같습니다.

```python
torch.save(mode, 'model.pth')  # 모델 저장하기
torch.load('model.pth') # 모델 불러오기
```