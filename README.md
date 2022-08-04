# 2022 K-ICT 빅데이터센터 딥러닝 초중급 과정
본 교육은 딥러닝 기초부터 중급까지를 모두 다룹니다. 판교 스타트업캠퍼스 1동 6층에 소재한 빅데이터센터 교육장에서 강의가 진행됩니다. 강의는 3일간, 오전 10시~오후 6시까지(총 21시간) 과정으로 진행되며, 이론과 실습과정으로 운영됩니다.

<br>

## 1. 수강 대상
본 강의는 인공지능과 머신러닝 그리고 딥러닝 관련하여 관심 있는 수강생을 대상으로 교육을 진행됩니다.    
> 본 강의는 파이썬에 대한 기초 지식이 없는 수강생에게는 다소 어려울 수 있습니다. 이 점 양해 부탁드립니다.

<br>

## 2. 교육 일시 및 장소
본 강의는 `2022.8.31(수)~9.2(금)`으로 `3일간` 진행됩니다. 각 일자별로 `오전 10시 ~ 오후 6시까지(총 21시간)`로 진행됩니다. 본 교육은 [판교 스타트업캠퍼스 1동 6층](http://kko.to/ZorLrT4fH)에 소재한 빅데이터센터 교육장에서 강의(온라인)가 진행됩니다. 

<br>

<div align='center'><div style="font:normal normal 400 12px/normal dotum, sans-serif; width:640px; height:392px; color:#333; position:relative"><div style="height: 360px;"><a href="https://map.kakao.com/?urlX=523421.0&amp;urlY=1084809.0&amp;name=%EA%B2%BD%EA%B8%B0%20%EC%84%B1%EB%82%A8%EC%8B%9C%20%EB%B6%84%EB%8B%B9%EA%B5%AC%20%ED%8C%90%EA%B5%90%EB%A1%9C289%EB%B2%88%EA%B8%B8%2020&amp;map_type=TYPE_MAP&amp;from=roughmap" target="_blank"><img class="map" src="http://t1.daumcdn.net/roughmap/imgmap/41031ba4eb5bfcfbfed6787cbad70521c9e47b492e4f704de694cc8060ff0416" width="638px" height="358px" style="border:1px solid #ccc;"></a></div></div></div>

<br>

## 3. 강의 세부 안내
본 강의는 파이썬에 대한 기초 지식과 머신러닝에 대한 전반적인 이론과 마지막으로 이론에 대한 실습으로 운영됩니다. 본 강의 자료 및 실습 자료를 [한꺼번에 다운](https://github.com/minsuk-sung/2022-NIA-Deep-Learning-Lecture/archive/refs/heads/main.zip)받기 위해선 아래 그림과 같이 [Download ZIP](https://github.com/minsuk-sung/2022-NIA-Deep-Learning-Lecture/archive/refs/heads/main.zip)을 눌러주시길 바랍니다.

![](https://i.imgur.com/LamEaXl.png)

<br>

### 실습환경
본 강의의 실습시간을 위해서 아래와 같은 라이브러리 버전을 맞춰주셔야 원할한 실습이 진행됩니다. 기본적으로 Google Colab 환경에서 실습을 진행해주는걸 권장합니다.

<br>
<div align='center'>
<!-- https://naereen.github.io/badges/ -->
<img src="https://img.shields.io/badge/python-v3.7-blue">
<img src="https://img.shields.io/badge/NumPy-1.20.2-blue">
<img src="https://img.shields.io/badge/Pandas-1.2.3-green">
<img src="https://img.shields.io/badge/Scikit--learn-0.24.2-yellowgreen">
</div>
<br>

Anaconda를 이용하여 가상환경을 생성할 경우 아래와 같은 명령어를 한줄 한줄씩 실행해주세요.
```
>> (base) conda create -n k-ict python=3.7 -y
>> (base) conda activate k-ict
>> (k-ict) conda install -c anaconda jupyter
>> (k-ict) conda install numpy==1.20.2
>> (k-ict) conda install pandas==1.2.3
>> (k-ict) conda install matplotlib==3.3.4
>> (k-ict) conda install scikit-learn==0.24.2
```

<br>

### 1일차 교육 - 8/31(수)
1일차에서는 머신러닝에 대한 간단한 개요와 기초 이론 그리고 학습 파이프라인에 대해서 다룹니다. 실습시간에는 파이썬 기초를 복습하고 NumPy, Pandas, 그리고 Scikit-learn과 같은 데이터 분석을 위한 라이브러리들을 가볍게 다루는 시간을 갖습니다. 

#### 1일차 이론 내용
- ㅇㅇ

#### 1일차 실습 내용
- 파이썬 기초 복습 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](]) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://nbviewer.jupyter.org/github/minsuk-sung/2021-NIA-K-ICT-AI-Lecture/blob/main/day1/%281%EC%9D%BC%EC%B0%A8%29%202021%EB%85%84%20NIA%208%EC%9B%94%20%EB%B6%84%EC%84%9D%EC%9D%B8%ED%94%84%EB%9D%BC%EA%B5%90%EC%9C%A1%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20-%20%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C%20%281%29%20%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%EA%B8%B0%EC%B4%88%20%EB%B3%B5%EC%8A%B5.ipynb)

<br>

### 2일차 교육 - 9/1(목)
2일차에서는 머신러닝에서의 회귀와 분류와 관련된 내용을 강의합니다. 지도학습에서의 대표적인 모델 중 하나인 k-최근접 이웃 알고리즘와 나이브 베이즈 그리고 서포트 벡터 머신에 대해서 배웁니다. 실습시간에는 Boston 주택 가격 데이터, Iris 데이터, 와인 품질 데이터 그리고 KOSPI 지수 데이터를 이용해서 이론 시간에 학습한 모델을 활용하는 시간을 가져봅니다.

#### 2일차 이론 내용
- 회귀(Regression)
- 분류(Classification)
- k-최근접 이웃 알고리즘(K-Nearest Neighbor, KNN)
- 나이브 베이즈(Naive Bayes)
- 서포트 벡터 머신(Support Vector Machine, SVM)

#### 2일차 실습 내용
- Boston 주택 가격 데이터를 통해서 알아보는 머신러닝 예제(1) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](]) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://nbviewer.jupyter.org/github/minsuk-sung/2021-NIA-K-ICT-AI-Lecture/blob/main/day2/%282%EC%9D%BC%EC%B0%A8%29%202021%EB%85%84%20NIA%208%EC%9B%94%20%EB%B6%84%EC%84%9D%EC%9D%B8%ED%94%84%EB%9D%BC%EA%B5%90%EC%9C%A1%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20-%20%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C%20%281%29%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%20%EA%B0%80%EA%B2%A9%20%EB%8D%B0%EC%9D%B4%ED%84%B0.ipynb)

<br>

### 3일차 교육 - 9/2(금)
3일차에서는 지도학습에서 가장 중요한 의사결정나무와 앙상블 방법에 대해서 알아봅니다. 마지막으로 비선형적인 문제를 해결하기 위해서 인공신경망에 대해서 다뤄봅니다. 실습시간에는 당뇨병 데이터나 유방암 데이터와 같은 간단한 데이터와 타이타닉 생존 데이터나 MNIST 숫자 데이터를 통해서 앞서 배운 이론을 적용하는 시간을 가져봅니다.

#### 3일차 이론 내용
- 의사결정나무(Decision Tree)
- 앙상블(Ensemble): 보팅(Voting), 배깅(Bagging) 및 부스팅(Boosting)
- 인공신경망(Artificial Neural Network)

#### 3일차 실습 내용
- 당뇨병 데이터를 통해서 알아보는 머신러닝 예제(5) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](]) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://nbviewer.jupyter.org/github/minsuk-sung/2021-NIA-K-ICT-AI-Lecture/blob/main/day3/%283%EC%9D%BC%EC%B0%A8%29%202021%EB%85%84%20NIA%208%EC%9B%94%20%EB%B6%84%EC%84%9D%EC%9D%B8%ED%94%84%EB%9D%BC%EA%B5%90%EC%9C%A1%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20-%20%EC%8B%A4%EC%8A%B5%EC%9E%90%EB%A3%8C%20%281%29%20%EB%8B%B9%EB%87%A8%EB%B3%91%20%EB%8D%B0%EC%9D%B4%ED%84%B0.ipynb)

<br>

## 4. 질의응답
본 머신러닝 강의와 관련된 질문은 강사의 [오픈 카카오톡 프로필](https://open.kakao.com/me/minsuksung)이나 `이메일`을 통해서 부탁드리겠습니다. 

<div align='center'><a href="https://open.kakao.com/me/minsuksung"><img src="https://i.imgur.com/l00WEY4.png" width=20%></a></div>

<br>

## 5. 라이센스

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright (c) 2022 Minsuk Sung

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
