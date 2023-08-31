---
layout: post
title: "[MOT] 칼만 필터(Kalman filter) 이해하기"
categories: [Theory]
tags: [Kalman filter,로봇, 자율주행]
---

# 칼만 필터(Kalman filter)란?
칼만 필터는 1960년대 초 Rudolf E. Kalman이 개발한 알고리즘으로 NASA의 Apolo Project에서 네이게이션 개발 시에 사용되었다. 현재는 GPS, 날씨 예측 등 다양한 곳에서 사용되고 있다.

칼만 필터는 최소 자승법(최소 제곱법, Least Square Method)를 사용한다.
과거와 현재의 값을 기준으로 재귀적 연산을 통해 최적값을 추적하는 것이다.

여기서 측정 데이터 혹은 신호는 잡음(noise)를 가지게 되는데 칼만 필터는 필요한 신호나 정보를 고르는 알고리즘으로 볼 수 있다.
