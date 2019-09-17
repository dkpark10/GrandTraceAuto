# Perceptual_Losses 

> perceptual loss(판단손실)은 근처 픽셀과의 관계 등 좀더 현실적인 정보(?)</br>
per loss 는 대략 인풋이미지에 반고흐 스타일을 입힌다면 인풋이미지와 반고흐 스타일에서의 지각적 차이(?) </br>
어떤 semanic feature가 유지되기 위함임 </br>

## introduction

이미지 처리에서의 예로는 디노이즈, 초해상도(해상도 복원), colorzation 이있는데, </br>
여기서 인풋은 저하된 이미지(노이즈, 저해상도 또는 그레이스케일)이고 출력은 고품질 컬러 이미지다.</br>
컴퓨터 영상처리로부터의 예시는 semantic segmentation(의미있는 분류 즉 분류를 넘어서 그 장면을 완벽히 이해하는 것), </br>
depth estimation(?) 을 포함하는데 여기서 인풋은 컬러이미지고 아웃풋은 그 장면에 대한 의미 또는 기하학적 정보를 인코딩한다. </br>
</br>
이미지 스타일 변환을 해결하기위한 하나의 접근법은 **픽셀** 당 손실함수를 사용하여 출력이미지와</br>
groud-truth images(사물이 실제 위치한 정보) 간의 차이를 피드포워드 </br>
뉴럴신경망을 지도학습(?)안에서 학습시키는 것이다. </br>
</br>
~~예를 들어 이 접근방식은 초해상도[1]에 동 외 연구진,~~</br>
~~색채화를 위한 청 외 연구진[2], 분할을 위한 롱 외 연구진[3],~~ </br>
~~깊이와 표면 정상 예측을 위해 아이겐 외 연구진이 사용하였다[4,5].~~</br> 
</br>
이러한 접근법은 테스트 시간에 효율적이며 학습된 네트워크를 통과하는 전진패스(?)만 필요하다 </br>
그러나 이러한 방법에 사용되는 픽셀당 손실은 출력이미지와 </br>
ground-truth images간의 perceptual differences를 잘 잡지 못한다</br>
</br>2를 보자</br>

![perceptuallosses](https://user-images.githubusercontent.com/43857226/64936736-4ff82c80-d892-11e9-93b3-506cc33c1868.JPG) </br>

style transfer(top) 그리고 x4 초해상도(bottom)에 대한 결과 style transfer의 경우 Gatys et al과 유사한 결과를 얻지만 </br>
나머지 세 경우 더 빠른 순서를 가진다(?) </br>
초해상도를 위해 지각손실을 사용한 우리의 방법은 픽셀단위로 학습된 방법보다 디테일을 더 잘 구성할 수 있다. </br>
~~동일한 이미지가 픽셀당 손실 측정시 매우 다를 수 있는 지각적 유사성(?) 에도 불구하고 한 픽셀씩 서로 상쇄된다.~~ </br>
최근 연구는 고품질 이미지를 생성할 수 있다는 걸 보여준다. 픽셀간 차이가 아니라 미리 훈련된</br>
CNN에서 추출한 고차원 영상 특징 표현들 간의 차이에 기초한 perceptual losses를 사용하여서 </br>
</br>

이 논문에서 두가지 이점을 결합 </br>
style transfer를 위해 피드 포워드 네트워크를 학습하지만 낮은 픽셀 정보에만 의존하는 픽셀당 손실함수를 사용하기 보다</br>
사전학습된 loss 네트워크로부터 하이레벨기능에 의존하는 perceptual loss를 사용하여 트레이닝 한다.</br>
훈련중 perceptual loss는 픽셀당 losses 보다 이미지 유사성을 강력하게 측정하면서 실시간으로 네트워크가 수행한다. </br>
</br>
style transfer와 싱글이미지 초해상도</br>
둘다 본질적으로 잘못됨 style transfer는 정확한 단일 output이 없으며 초해상도는 동일한 저해상도인풋을 </br>
생성할 수 있는 고해상도 이미지가 많이 있다. </br>
style transfer 의 경우 output은 의미론적으로(매끄런 해석) 색변화와 텍스처의 급격한 변화에도 input과 </br>
유사해야 한다. </br>
percaptual loss 의 사용은 loss network에서 transformation network로 의미론적인 지식(?)의 전송을 허용한다 </br>
> 먼말이야 ㅅㅂ
style tansfer를 위해 우리의 피드포워드 네트워크는 최적화(optimizer) 문제를 해결하기 위해 훈련된다. </br>
대충 3배 빠름 ~~~ </br>

## related work

### feed-forward image transformation
최근 몇년동안 피드포워드 이미지 변환 작업은 다양하게 발전해왔다리 ~ </br>
segmantic segmentation(의미있는 분류)는 픽셀단위당 classfication loss(?)를 학습해 인풋에 </br>
풀리 컨볼루션 방법을 실행하여 dense scene labels를 생성한다. </br>
**CRF**추론을 통해 프레임화함으로서 픽셀당 손실을 move(?) 한다 </br>
우리의 transformation network의 구조는 **3** 과 **14** 에서 영감을 얻었는데 이 구조는 </br>
다운샘플링을 사용하여 피쳐맵의 spatial extent를 줄이기위해 사용 </br>
업생플링을 사용해 output 생성한다. </br>
</br>
depth 와 surface normal estimation을 위한 최근 방법들은 perpixel 회귀분석 또는 분류 losses로 </br>
학습된 피드포워드 컨볼루션 네트워크를 사용함으로서 3채널 컬러이미지를 의미있는(정보가있는??) 출력영상으로
변환하는 점이 유사하다. </br> 
일부 방법은 이미지 그라디언트 또는 CRF loss 레이어를 사용하여 아웃풋에서 local 일관성(?)을</br>
적용함으로 픽셀당 손실을 넘어선다 </br> 
> 알아듣게좀 ㅅㅂ 
피드포워드 모델은 그레이를 컬러로 변환하는 방법을 픽셀당 loss를 사용하여 학습된다. </br>

### perceptual optimization
**continue**

### style transfer

![styleransfer](https://user-images.githubusercontent.com/43857226/65004091-1da00b00-d936-11e9-8ece-62a4bbffc350.JPG)</br>
> sf의 전반적인 개요도 인풋이미지를 변환하기 위해 img trans network을 학습한다. </br>
img 분류를 위해 사전학습된 loss network을 사용하여 이미지 사이의 차이를 측정하는 </br>
perceptual loss를 정의한다. 

Gatys et al은 미술적인(반고흐같은)style transfer를 수행한다.  </br>
미리 학습된 컨볼루선 네트워크로부터 추출된 피쳐에 기반한 style reconstruction loss와 </br>
feature reconstruction loss를 공동으로 최소화함으로서 하나의 이미지를 다른 이미지스타일과 결합</br>
(코랩에 있는 예제 생각하면 댐)</br>
유사한 방법이 있는데 오래 걸려서 우리는 피드포워드 네트워크를 훈련시켜 신속하게 한다~~ 이말임 </br>

### image super-resolution

이미지 초해상도란 고질적으로 문제다 ~ </br>
초해상도 기법을 여러여러 방법으로 분류 </br>
최근 픽셀당 유클리드 loss로 훈련된 3층 컨볼루션 신경망을 사용하여 이미지 초해상도에 </br>
우수한 성능 달성 </br>

## method

