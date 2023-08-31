---
layout: post
title: "[MOT] 베이즈 필터(Bayes filter) 이해하기"
categories: [Theory]
tags: [Bayes filter,로봇, 자율주행]
---

먼저, 베이즈 필터(Bayes Filter)는 칼만 필터(Kalman Filter)와 파티클 필터(Particle Filter) 개념의 기초가 되는 필터(Filter)로 베이즈 필터를 통해 먼저 이해하고 넘어가도록 하자.

# 베이즈 필터(Bayes Filter)란?


- 베이즈 필터는 베이즈 이론(Bayes theorem)을 재귀적으로 따르게 된다.
    - 베이즈 이론을 이용하면 사전 확률(prior probability)와 가능도(likelihood)를 이용해 사후 확률(posterior probability)를 구할 수 있다. 이를통해 구한 사후 확률은 다음 스텝에서의 사전 확률로 사용되는 것이다.

    - 베이즈 필터의 식은 다음과 같다.

    ![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/7e74f3ba-8a5b-4c07-9e0e-5969cf11ee0a)


$$
bel(x_t)=p(x_t|z_{1:t},u_{1:t})
$$

- 베이즈 필터의 목적을 한 줄의 수식으로 표현하자면 위와 같다.
    - $x_t$ : $t$ 시점에서의 상태
    - $z_t$ : $t$ 시점에서의 센서 입력 값
    - $u_t$ : $t$ 시점에서의 제어 입력 값
    - 따라서 시작부터 현재시점 $t$까지의 센서와 제어 입력 값으로부터 현재 상태를 확률적으로 추정하는 것으로 이해할 수 있다.

- 즉, 제어값을 사용하여 물리적인 모델을 통해 현재 상태를 예측하고, 센서값을 통해 예측하여 두 가지를 같이 사용하여 결과를 예측한다.
    - Ex. 이전 위치 상태 + 속도 + 제어값을 이용해 먼저 예측을 하고, 센서를 통해 결과값을 보완한다.
    - 이때, 노이즈가 작은 값을 반영하는 것이 높은 성능을 얻기 위해 고려할 점이다.

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/8a1c5289-1f15-47e5-86d1-70cf4f009d37)

코드를 통해 베이즈 필터를 이해해보자.
- 먼저 반복문의 첫번째 줄부터 살펴보자
    - 첫번째 줄의 변수가 제어 입력 값인 $u_t$와 이전 상태 값을 사용하여 현재 상태를 예측하기 때문에 **control** 또는 **prediction**으로 불린다. 
    - 두번째 줄은 $z_t$로 측정한 센서값을 사용하여 업데이트하여 **measurement updata**로 불리거나 prediction의 결과를 센서값을 통해 보정하여 **correction**이라고도 불린다.

![image](https://github.com/SangHyun014/SangHyun014.github.io/assets/87685922/453c165b-b525-422d-8559-a4704334111b)

1. $bel(x_{t-1})$ : 이전 상태
2. $p(x_t|u_t,x_{t-1})$ :이전 상태와 제어값이 주어졌을 때, 현재 상태의 확률 분포
3. $\overline{bel}(x_t)$ : control update(prediction)
4. $p(z_t|x_t)$ : 현재 상태의 센서값 확률 분포
5. $bel(x_t)$ : measurement update(correction)

- 1,2 를 이용하여 3 control update(prediction)을 계산하고, 3,4를 이용하여 5 measurement update(correction)을 계산한다.
- 최종적으로 얻은 measurement update는 다음 step의 입력이 되어 재귀적인 필터를 구성할 수 있다.

## 베이즈 필터의 한계와 개선 방법
- prediction 과정에서의 적분 연산
    - 적분 연산은 경우에 따라 연산이 복잡하거나 적분이 불가능한 경우가 있다.
    - Discrete의 경우 $\sum$을 이용해 계센하지만 Continuous의 경우 $\int$를 사용하기에 계산적인 문제가 발생한다.
    - 다음을 해결하는데에는 2가지 방법이 있다.
    1. 몬테 카를로 적분(Monte Carlo Integration)
        - **랜덤 샘플링 방식을 통해 근사화**하는 방법 $\to$ 파티클 필터(Particle Filter)
    2. 적분이 가능한 식만 사용(가우시안 분포)
        - 베이즈 필터의 제어값과 센서값의 노이즈가 정규 분포를 따른다고 가정하고, 노이즈의 평균=0, 표준편차 = $\sigma$로 가정한다.$\to$ 칼만 필터(Kalman Filter)


### Reference
https://gaussian37.github.io/autodrive-ose-bayes_filter/