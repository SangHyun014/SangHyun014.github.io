---
layout: post
title: "[Pytorch] pytorch tutorial 따라하기 - Dataset과 Dadaloader"
categories: [Pytorch]
tags: [Pytorch, tutorial]
---

이번에는 두번째로 데이터를 다루는 tutorial을 진행해보자.

# Dataset과 Dataloader

이번 데이터셋도 동일하기 Fashion-MNIST로 진행하겠습니다. Fashion-MNIST는 Zalando의 기사 이미지 데이터셋으로 60,000개의 학습 에제와 10,000개의 테스트 예제로 이루어져 있습니다. 각 예제는 흑백(grayscale)의 28x28 이미지와 28개 분류(class) 중 하나인 정답(label)으로 구성됩니다.

다음 인자들을 통해 데이터셋을 불러옵니다.
- <code>root</code> 는 학습/테스트 데이터가 저장되는 경로
- <code>train</code> 은 학습용 또는 테스트용 데이터셋 여부를 지정
- <code>download=True</code>는 <cdoe>root</code>에 데이터가 없는 경우 인터넷에서 다운로드 합니다.
- <code>trasform</code>과<code>target_transform</code>은 특징(feature)과 정답(label) 변형을 지정합니다. ToTensor에 관한 설명은 앞선 post인 빠른 시작()