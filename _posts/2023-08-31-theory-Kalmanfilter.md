---
layout: post
title: "[MOT] 칼만 필터(Kalman filter) 이해하기"
categories: [Theory]
tags: [Kalman filter,로봇, 자율주행]
---

# 칼만 필터(Kalman filter)란?
칼만 필터(Kalman filter)는 베이즈 필터(Bayes filter)를 기반으로 만들어져 적분에 어려움이 있을 수 있다는 한계를 해결하기 위해 제시된 알고리즘이다.
- 칼만 필터는 일부 동적 시스템에 대한 정보가 확실하지 않은 곳에서 사용할 수 있으며 다음 수행 할 작업에 대한 정확한 추측을 할 수 있다.
- 센서를 통해 추측한 움직임에 노이즈(noise)가 있더라도 제거에 있어 좋은 역할을 한다.
- 이는 연산 과정이 빨라 real-time, embedded system에 적합하다.

칼만 필터는 다음 2가지의 가정이 성립할 때 사용할 수 있다.

1. 