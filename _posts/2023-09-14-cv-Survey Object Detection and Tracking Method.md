---
layout: post
title: "[Paper Review] A Survey on Moving Object Detection and Tracking Methods"
category: Paper
tags: [Paper, Computer Vision, Object Tracking, Survey, Object Detection, Method]
---

MOT(Multi Object Tracking) 알고리즘에 대해 Aiffel 과정의 프로젝트를 진행하면서 대략적으로 공부했지만, 정확한 공부를 했다고 생각하지 않으며 추가적인 공부 및 실험이 필요하다고 판단했습니다. 그래서 새마음 새뜻으로 진행을 해 볼 생각입니다.

시작하기에 앞서 먼저 Obejct Tracking의 기본부터 들어가는 생각으로 해당 Survey 논문을 읽어볼 것입니다. 해당 논문은 Object Detection과 Tracking의 기본적인 방법론에 대해 이야기합니다.

# A Survey on Moving Object Detection and Tracking Methods

## Abstract
- Object Tracking Algorithm의 목적은 **비디오 장면으로부터 관심 영역을 분할하고 물체(Object)의 움직임, 위치 그리고 가려짐(Occlusion)을 추적하는 것**입니다.
- Object Tracking이 수행되기 전에는 우선, sequence image로부터 물체를 탐지(Detection)하고 분류(Classification)가 되어야합니다.
- Object Tracking은 객체(Object)의 존재 여부(presence), 위치(position), 크기(size), 모양(shape) 등과 같은 공간적(spatial), 시간적(temporal) 변화를 모니터링함으로써 수행됩니다.

## Introduction
- 각 프레임(frame)에 적용되는 모든 이미지 처리 기술과 두 연속적인 프레임의 내용은 밀접한 관계가 있습니다.
- 일반적인 Object Detection Algorithm을 사용하는 것은 바람직하지만, 잘 모르는 객체(Unknown Object) 혹은 Color, Shape, Texture 등 다양한 변수가 있는 객체를 다루는 것은 어렵습니다.
- 비디오 시퀀스(Video Sequence)로부터의 이미지는 두가지 complimentary sets of pixels로 나누어 집니다.
    - First Set : 전경 객체(Foreground Object)에 대한 픽셀 정보
    - Second SEt : 배경 객체(Background Object)에 대한 픽셀 정보
    - 다음과 같이 처리된 이미지는 Binary Image 혹은 Mask 형태의 결과로 나타납니다.
- Object Tracking은 보안과 감시와 같은 여러 중요한 일에서 사람을 인식할 수 있고, 시각적 정보를 사용해 더 나은 정보를 제공할 수 있다는 점에서 중요한 의미를 가집니다.
- Object Tracking은 기본적으로 다음 과정들을 거치게 됩니다.
    1. Object Detection
        - Object Detection은 video sequence에서 관심있는 객체(Object)를 인식하고, 이러한 객체들의 pixels을 cluster하는 과정입니다.
        - frame differencing, optical flow, background subtraction과 같은 다양한 기술들로 수행할 수 있습니다.
    2. Object Classification
        - 객체(Object)는 다양한 움직이는 객체들로 분류될 수 있습니다.
        - 방법으로는 Shaped-based classification, Motion-based classification, Color-based classification, Texture-based classification 등이 있습니다.
    3. Object Tracking
        - tracking은 장면내 이미지 영역에서 객체의 경로(path) 근사치를 구하는 문제라 할 수 있습니다.
        - point tracking, kernel tracking, silhouette과 같은 방법들이 있습니다.
        - tracking을 할 때는 아래의 5가지를 주의하며 수행해야 합니다.
            1. 2D 이미지에서 3D 영역의 추정에 의한 증거 손실(evidence of loss)
            2. 이미지의 노이즈(noise)
            3. 객체 움직임의 어려움(Difficult object motion)
            4. 객체 가려짐(occlusion)에 의한 불완전성(Imperfect)
            5. 객체 구조의 복잡성(Complex objects structures)


![image](https://github.com/SangHyun014/Datathon/assets/87685922/24b39d9d-ebcb-4752-bda3-2f2d43d6ae82)

- 과정을 도식화하면 위의 그림과 같습니다.

## Object Detection Methods
- Object Tracking의 과정은 첫번째 과정은 video sequence 내 관심 객체(object of interest)를 인식하고 그러한 객체들의 픽셀들을 cluster하는 것입니다.
- 이것을 수행하는 방법은 3가지가 있습니다.
    1. Frame differencing

        - 두 개의 연속되는 이미지 사이의 차이를 계산함으로써 움직이는 객체가 있다고 판단하는 방법입니다.  
        - 이러한 방법은 동적인 환경(dynamic environment)의 다양성에 대해 강하게 적응할 수 있지만, 움직이는 객체의 outline을 완벽하게 포함하는 것이 어렵습니다. 또한 empty phenomenon(빈 현상)이 나타나 움직이는 객체의 detection이 정확하지 않습니다.

    2. Optical flow
        - Opticla Flow는 Optical field를 구하기 위하여 이전 프레임과 현재 프레임의 차이를 이용하고 픽셀값과 주변 픽셀들과의 관계를 통해 각 픽셀의 이동(motion)을 계산하여 추출합니다. 이를 통해 움직임을 구별해 내는 것입니다. 
        - 이 방법은 움직임 정보를 완벽하게 얻을 수 있고 배경으로부터 움직이는 객체를 탐지할 수 있습니다. 하지만 큰 연산량, 노이즈에 대해 민감한 단점을 가지고 있습니다. 또한 노이즈가 없는 영상에 대해서도 좋지 않은 결과를 보이며, real-time에 적용하기 적합하지 않은 방식입니다.
        ![image](https://github.com/SangHyun014/Datathon/assets/87685922/6c88c324-d614-4592-8eda-dbafbffaa97f)

    3. Background Subtraction
        -   Background Subtraction의 First Step은 background modeling입니다. Background modeling은 움직이는 객체를 충분히 인식해야 한다는 점이 핵심입니다. Background modeling 은 reference model로 사용되며 현재 비디오 프레임과 reference frame 간의 변화가 움직이는 객체의 존재를 나타내게 됩니다. 현재, mean filter와 median filter가 background modeling 으로 주로 사용됩니다.

        - 이러한 방법은 외부 환경 변화에 민감하고 간섭 방식 능력(anti-interference ability)가 좋지 않습니다. 하지만 background의 정보에 대해 잘 알고 있는 경우 객체에 대한 정보를 완벽하게 얻을 수 있습니다.

![image](https://github.com/SangHyun014/Datathon/assets/87685922/2a9175cf-a433-4de4-810c-7af07e60950a)
해당 이미지는 Object Detection Method를 정리한 표입니다.

## Object Classification Methods
- 서로 다른 종류의 객체들을 추출하기 위한 다양한 Classification 알고리즘들이 있습니다. 본 논문에서는 shape feature을 이용하여 영역에서 객체를 추출하는 접근법들에 대해 알아보겠습니다.
    1. Shape-based classification
        - point, box, blob 등과 같은 feature들로 shape가 이루어지게 됩니다.이러한 정보들로 움직이는 물체를 Classification을 수행합니다.
    
    2. Motion-based classification
        - 관절형 객체는 주기적인 특성을 가지게 되고, 이는 객체를 분류하는 강력한 단서(cue)로 사용되었습니다.
        - optical flow는 object classification에 유용하게 사용될 수 있으며, residual flow는 움직이는 객체의 강성(rigidity)와 주기성(periodicity)를 분석하는데 사용할 수 있습니다.
    
    3. Color-based classification
         - 컬러 기반 분류는 viewpoint의 변화와 밀접한 관계가 있습니다. 
         - color 정보는 항상 적절한 정보를 가져오는 것은 아니지만, 알고리즘의 낮은 계산적 비용으로 적절할 때 사용하면 좋은 특징(feature)입니다.
         - 가우시안 혼합 모델(Gaussian Mixture Model)에 따르면 이미지 시퀀스와 배경,객체에서 color의 분포(distribution)을 표현할 수 있다고 합니다.

    4. Texture-based classification
        - Texture 기반 분류 방법은 이미지의 지역적으로 퍼져있는 부분에서 기울기의 방향(gradient orientation)을 계산할 수 있습니다.

![image](https://github.com/SangHyun014/Datathon/assets/87685922/ea73c3a2-141b-490a-b3fe-c7467bd38203)
위는 Object Classification Method를 정리한 표입니다.

##  Object Tracking Methods
- Tracking은 이미지 영역 내의 움직임, 경로(path)를 추적하는 문제로 정의할 수 있습니다.
- Object Tracking의 목적은 비디오의 단일 모든 단일 frame에서 객체의 위치를 찾아 경로(route)를 생성하는 것입니다.
- Object Tracking은 Point tracking, kernel based tracking, silhouette based tracking으로 나누어집니다.

![image](https://github.com/SangHyun014/Datathon/assets/87685922/412e8e9c-2c6b-4dba-a07d-1108078bdaf7)

- **Point Tracking**
    - 이미지 구조에서 움직이는 객체는 tracking을 하는 동안 특징점(feature points)에 의해 표현됩니다.
    - Point Tracking은 객체의 Occlusion 발생으로 복잡한 문제입니다. 하지만 Thresholding을 이용함으로써 객체 인식은 간단하게 수행될 수 있습니다.
        1. *Kalman Filter*
        - 칼만 필터(Kalman Filter)는 최적 재귀 데이터처리 알고리즘(Optimal Reculsive Data Processing Algorithm)에 기반한 방법입니다.
        - 칼만 필터는 수학적 방정식(mathematical equation)의 집합이고, tracking을 위한 효과적인 계산을 할 수 있습니다.
        - 이는 상황을 추적하고, 노이즈 측정에 대해 피드백을 제공해줍니다.
        - 칼만 필터 방정식은 두가지 그룹으로 나누어 수행됩니다.
            - time update equation : 다음 상태를 예측하기 위해 현재 상태 및 오류 공분산 추정치를 예측하는 역할을 합니다.
            - measurement update equation : 피드백을 제공하는 역할을 합니다.

        - 이 방식은 항상 Optimal Solution을 제공해줍니다.

        2. *Particle Filtering*
        - particle filtering은 다음 변수가 움직이기 전에 한 변수에 대해 모델을 생성합니다.
        - 이 방식은 무한한(unboundedly) 다양한 변수들을 다룰 수 있고, 동적인 상황의 변수들이 있을 때 유리한 알고리즘입니다.
        - 칼만 필터의 한가지 단점은 상태가 정규분포(Gaussian)를 따르지 않는 상황에서 근사치가 좋지 않다는 것입니다. $\to$ 해당 알고리즘은 이러한 부분을 개선할 수 있습니다.
        - 이 알고리즘은 주로 contours, color feature, texture mapping에서 사용됩니다.
        - 이 방식은 Kalman Filtering과 마찬가지로 prediction과 update 구 단계로 구성됩니다.

- **Kernel Based Tracking**
    - Kernel Tracking은 하나의 프레임에서 다음 프레임까지 초기의(embryonic) 객체 영역에 의해 표현되는 움직이는 객체를 계산함으로써 수행됩니다.
        - 객체의 움직임은 translation, conformal, affine 등 parametric motion의 형태입니다.
    - Real-time에서는 기하학적 형태(geometric shape)를 이용하여 객체를 표현하는 것이 일반적입니다.
        - 한가지 제한사항은 정의된 모양이 외부에 남아 있을 수 있고, 배경의 일부가 내부에 존재할수 있다는 점입니다.
    - 이 방법은 물체의 표현, 특징, 모양 및 외관을 기반으로 하는 대규모 tracking techniques입니다.
        1. *Simple Template Matching*
        - Template mathcing은 관심 영역을 brute force method를 이용하여 추적하는 방법입니다.
        - Tracking은 비디오에서 단일 객체에 대해서 수행할 수 있고, 부분적으로 overlapping된 객체에 대해서도 수행할 수 있습니다.
        - 이 방법은 매치된 이미지의 작은 부분을 탐지해 digital image를 처리하거나 프레임 안에서 template과 일치하는 동등한 이미지 모델을 추적하여 처리하는 방법입니다.

        2. *Mean Shift Medthod*
        - Mean Shift Algorithm은 미리 전처리된 모델과 locally하게 나타내는 모델의 유사점을 찾는 방법입니다.
        - 즉, Tracker를 현재 영역과 모델 사이의 유사 정도가 최대가 되는 위치를 찾으며 수행하게 됩니다.
        - tracking된 이미지 영역은 히스토그램으로 표현될 수 있습니다.

        3. *Support Vector Machine(SVM)*
        - SVM은 positive와 negative 훈련값을 제공하는 broad classification method 입니다.
        - positive sample에는 추적된 이미지 객체가 포함되고, negative sample에는 추적되지 않은 나머지 모든 항목들이 포함됩니다.
        
        4. *Layering based tracking*
        - multiple object에서 kernel based tracking을 수행할 때 사용되는 방법입니다.
            - 각 레이어(layer)는 translation, rotation 같은 움직임과 타원(ellipse)모양을 포함하며, layer appearance, based on intensity를 포함합니다.
        - 모든 픽셀의 확률은 객체의 앞선 움직임과 shape feature을 기반으로 계산됩니다.
        - 이것은 multiple image 혹은 완전히 가려진 객체를 tracking 할 수 있습니다.

- **Silhouette based tracking approach**
    - Silhouette-based object tracking의 목표는 이전 프레임에서 생성되는 object model의 의미(means)에 대해 모든 프레임에서 객체 영역을 찾는 것입니다.
    - 이는 객체의 다양성, 가려짐(Occlusion) 그리고 객체의 분리(split)와 합성(merge)들을 다루는데 용이합니다.
        1. *Contour Tracking*
        - Contour Tracking 방식은 현재 프레임에서 새로운 position에 대해 이전 프레임에서의 주된 윤곽(contour)을 반복적으로 처리하는 알고리즘입니다.
        - 이 방식은 두가지 접근법을 사용하여 수행될 수 있습니다.
            - 윤곽 형태과 움직임을 모델링하기 위해 상태 공간 모델(state space model)을 사용합니다.
            - 경사 하강법(gradient descent)과 같은 방법을 사용해 윤곽 에너지(contour energy)를 최소화함으로써 윤곽을 직접적으로 보여줍니다.
        - 이는 객체의 다양한 shape에 대하여 유연하게 다룰 수 있습니다.

        2. *Shape Matching*
        - 이 접근법은 존재하는 프레임 안에서 object model에 대해 검사하는 방식으로 진행됩니다.
            - 방식은 kernel 방식의 template based tracking 방식과 유사합니다.
        - 또 다른 접근법은 두 연속 프레임에서 탐지된 일치하는 실루엣(silhouette)을 찾는 것입니다.
            - 방식은 point matching과 유사합니다.
            - 실루엣 기반 탐지(detection)은 배경 추출에 의해 수행됩니다.
        - 단일 객체 그리고 가려지는 객체를 다룰 수 있고, Hough transform 기술과 함께 수행될 수 있습니다.
     