---
layout: post
title: "[Pytorch] pytorch tutorial 따라하기 - 빠른시작"
categories: [Pytorch]
tags: [Pytorch, tutorial]
---

# Pytorch Tutorial(빠른 시작)
pytorch에 대한 이해도가 부족하다고 생각하여 tutorial을 진행하며 다시 이해해보자.

먼저 첫번째 Tutorial인 빠른 시작(Quick Start)부터 시작할 것이다.

우선 다음과 같이 필요한 라이브러리를 불러온다.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

파이토치(Pyroch)에는 데이터를 작업하는 요소 <code>torch.utils.data.DataLoader</code>와 <code>torch.utils.data.Dataset</code> 두가지가 있다. <code>Dataset</code>은 샘플과 정답(label)을 저장하고, <code>DataLoader</code>는 <code>Dataset</code>을 순회 가능한 객체(iterable)로 감싼다.

해당 예제는 이미지 데이터셋인 TorchVision 데이터셋의 FashionMNIST를 사용한다.

먼저 데이터셋을 불러오도록 하자.
```python
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
```
해당 <code>datasets.FashionMNIST()</code>의 인자 중 <code>transform=ToTensor()</code>를 이용해 Tensor형태로 변환을 해준다. 왜 변환을 해주어야 할까?
- PIL 이미지,Numpy의 배열은 **H x W x C**로 이루어져있지만 파이토치의 이미지의 경우 **C x H x W**의 구조를 가진다. 따라서 Channel의 위치가 변경되는 문제가 발생한다.
-  이미지 픽셀 밝기 값이 0~255의 범위에서 0~1의 범위로 변경된다. 

그리고 불러온 <code>Dataset</code>을 <code>DataLoader</code>를 이용해 묶습니다.
```python
batch_size = 64

# DataLoader 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f'Shape of y: {y.shape} {y.dtype}')
    break
```
![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/427eb359-c93a-44bd-8a40-0b24e79a81a1)

다음과 같이 결과를 확인할 수 있고, 볼 수 있듯이, **C x H x W**의 결과를 가지고 있음을 알 수 있다.