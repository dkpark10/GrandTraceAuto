# Perceptual_Losses 

> perceptual loss(판단손실)은 근처 픽셀과의 관계 등 좀더 현실적인 정보(?)</br>
per loss 는 대략 인풋이미지에 반고흐 스타일을 입힌다면 인풋이미지와 반고흐 스타일에서의 지각적 차이(?) </br>
어떤 semanic feature가 유지되기 위함임 </br>

## 1.Introduction

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
> 먼말이야 ㅅㅂ</br>
</br>
style tansfer를 위해 우리의 피드포워드 네트워크는 최적화(optimizer) 문제를 해결하기 위해 훈련된다. </br>
대충 3배 빠름 ~~~ </br>

## 2.Related work

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

> 알아듣게좀 ㅅㅂ </br>
</br>

피드포워드 모델은 그레이를 컬러로 변환하는 방법을 픽셀당 loss를 사용하여 학습된다. </br>

#### Perceptual Optimization
**continue**

#### Style Transfer

![styleransfer](https://user-images.githubusercontent.com/43857226/65004091-1da00b00-d936-11e9-8ece-62a4bbffc350.JPG)</br>
> sf의 전반적인 개요도 인풋이미지를 변환하기 위해 img trans network을 학습한다. </br>
img 분류를 위해 사전학습된 loss network을 사용하여 이미지 사이의 차이를 측정하는 </br>
perceptual loss를 정의한다. 

Gatys et al은 미술적인(반고흐같은)style transfer를 수행한다.  </br>
미리 학습된 컨볼루선 네트워크로부터 추출된 피쳐에 기반한 style reconstruction loss와 </br>
feature reconstruction loss를 공동으로 최소화함으로서 하나의 이미지를 다른 이미지스타일과 결합</br>
(코랩에 있는 예제 생각하면 댐)</br>
유사한 방법이 있는데 오래 걸려서 우리는 피드포워드 네트워크를 훈련시켜 신속하게 한다~~ 이말임 </br>

#### Image Super-Resolution

이미지 초해상도란 고질적으로 문제다 ~ </br>
초해상도 기법을 여러여러 방법으로 분류 </br>
최근 픽셀당 유클리드 loss로 훈련된 3층 컨볼루션 신경망을 사용하여 이미지 초해상도에 </br>
우수한 성능 달성 </br>

## 3.Method

그림 2와 같이 두가지 요소로 구성된다. </br>
여러 손실함수를 정의하는데 쓰이는 loss network 랑 img trnasfer network 이 두개로~~ </br>
img transfer network는 가중치 W에 의해 파라미터된 깊은 CNN이다. </br>
it transforms input images x into output images ^y via the mapping ^y = fW(x) </br>
> 매핑 y = fW(x)를 통해 입력이미지 x를 출력이미지 ^y로 변환 </br>
Each loss func computes a scalar value ``i(^y,yi) measuring the diffrence between </br>
the output image ^y and a target image yi. </br>
> 각 손실함수는 output ^y와 타켓 yi사이 차이를 계산하는 스칼라값 i(^y, yi)를 계산한다. </br>
img trans network는 loss func의 weighted combination을 최소화하기 위해 </br>
stochastic 경사하강법을 사용하여 학습합니다. </br>

![캡처](https://user-images.githubusercontent.com/43857226/65006652-b470c580-d93e-11e9-8e10-835bd46b2bdf.JPG) </br>

픽셀바이 픽셀을 해결하고 지각적 차이를 더 잘 하기 위해 영감얻는다 ~~ </br>
이러한 방법은 img cls(분류) 를 위해 학습된 CNN이 이미 loss func에서 측정하고자하는 perceptual 및 정보를 </br>
인코딩 하는 방법(????) </br>
따라서 loss func를 정의하기 위해 미리학습된 네트워크를 사용 </br>
deep conv transformation은 loss func을 사용함으로 학습되는데 이 loss func또한 </br>
deep conv network이다. </br>
</br>
The loss network φ is used to deﬁne a feature reconstruction loss φfeat and a style reconstruction </br>
loss φstyle that measure diﬀerences in content and style between images.  </br>

> 손실함수 네트워크는 reconstruction loss와 인풋(content)과 style 의 차이를 측정하는</br>
style reconstruction loss의 특징을 정의하는데 사용된다. </br>
</br>
for each input image 'x' we have a content target yc and a style target ys.</br>

> 각 입력 인풋 'x'에 대해 content target(?) **yc**와 style target **ys** 가 있다. </br>

for style transfer, the content target **yc** is the input image x and the output image **^y** shoud </br>
combine the content of x = yc with the style of ys; we train one network per style target </br>

> style transfer의 경우 content target **yc**는 입력 'x'이며 output은 **^y**는 x = yc의 내용을 </br>
**ys**의 스타일과 결합해야 합니다. </br>
스타일마다 하나의 network을 학습시킨다. </br> 

단일 이미지 고해상도의 경우 인풋 'x'는 low - resolution이며 content target **yc** 는 </br>
ground-truth high resolution(실제 라운딩 박스 고해상도 이미지) 이며 </br>
style reconstruction loss는 사용하지 않는다. </br>
하나의 네.트.워.크만 훈련한다. super resol당 (????) </br>

## 3.1 Image Transformation Network 

~~img trans networn는 radford등 에서 정한 아기텍처 지침을 대략 따른다 ~~ ~~ </br>

어떤 풀링 레이어를 사용하지 않고 대신 네트워크내 다운샘플 업샘플에 스트라이드 된 컨볼루션을 사용한다 </br>
네트워크는 아키텍쳐를 사용하는 다섯개의 residual blocks로 구성된다. **먼말?????**</br>

All non-residual convolutional layers are followed by spatial batch normalization [45] and ReLU </br>
nonlinearities with the exception of the output layer, which instead uses a scaled tanh to ensure that </br>
the output image has pixels in the range [0,255] </br>

> 모든 **non res-net**(공부)는 output layer를 제외하고 spatial batch normalization 및 relu 비선형 </br>
이 있으며 대신 tanh함수를 사용하여 output 이미지가 0 ~ 255 사이에 있는지 확인한다.</br>

**9x9 커널을 사용하는 처음 마지막 레이어를 제외하고 모든 컨볼루션 레이어는 3x3 커널을 사용한다. **</br>

#### Input and Output
style transfer 의 경우 인풋 아웃풋 모두 3 * 256 * 256 컬러영상 </br>
업샘플링 계수가 f인 초고해상도인 경우(????) **f는 대체 무엇이란 말인가........... **</br> 

출력은 3 * 288 * 288 초고해상도 이고 인풋은 저 해상도 3 * 288/f * 288/f이다 </br>
img transfer network는 풀리 컨볼루션이므로 어떤 해상도의 이미지에 적용될 수 있다. </br>

#### Downsampling and Upsampling

업샘플링 계수가 f인 초고해상도인 경우(????) **f는 대체 무엇이란 말인가........... **</br> 

여러 residual blocks를 사용한다. stride 1/2 컨볼루션 레이어를 log2 f를 </br>
이건 네트워크에 통과하기 전 저해상도 이미지를 업샘플링 하기위한 바이큐빅 보간법을 사용하는것과 다르다. </br>
고정된 업샘플링 func에 의존하지 않고 fractinally-strided 컨볼루션을 통해 업샘플링 기능은 나머지 </br>
네트워크와 공동으로 학습된다. </br>
</br>
style transfer의 경우 인풋에 다운샘플하기 위해 2-스트라이드 컨볼루션을 사용한다. </br>
input과 output은 동일 사이즈지만 업,다운 샘플링에 네트워크에 여러 이점이 많다. </br>
</br>
첫번째는 계산이다
C * H * W 크기의 인풋에 c필터를 포함한 3 * 3컨볼루션은 9HWC^2 multiple-add 연산이 필요한데. </br>
DC * H/D * W/D 사이즈 인풋에서 DC필터와 3 * 3컨볼루션과 동일한 연산시간이다. </br>
다운샘플링후 더 효과적으로 네트워크를 사용할 수 있다. </br>
</br>
두번째 이점은 효과적인 사이즈와 관련있다. </br>
고품질 style transfer는 일관성 있는 방식으로 이미지의 큰 부분을 바꾸는걸 요구한다. </br>
따라서 output의 각 픽셀이 input에서 효과적인 receptive field를 가지는것이 이점이다. </br>
다운샘플링 없이 각각 추가적인 3 * 3 컨볼루션레이어는 효과적인 receptive field 사이즈를 </br>
2만큼 증가한다. </br>
D의 요소로 다운샘플링 후 각 3 * 3 컨볼루션은 효과적인 **receptive field** 사이즈를 2D만큼 증가 시킨 후</br>
동일한 레이어 수로 더 큰 **receptive fields**를 제공한다. </br>

![캡처](https://user-images.githubusercontent.com/43857226/65095860-d1b89900-d9fd-11e9-85dd-20d2c130960c.JPG)
> 사전학습된 vgg-16 loss network로부터 여러개의 레이어 j에 대해 feature reconstruction loss를 최소화 하기 위한 </br>
^y을 찾기위해 최적화를 사용한다. 높은 레이어에서 재구성함으로 영상내용과 전체적인 **spatial structure**는 </br>
보존되지만 색상,텍스쳐,정확한 모양은 보존 되지 않음 </br> 

#### Residual Connections

이미지 분류를 위한 깊은 network를 학습하기 위해 residual connections를 사용한다. </br>
residual network는 identify func를 위한 네트워크를 쉽게 해준다. </br>
대부분의 경우 아웃풋과 인풋은 구조를 공유하기 때문에 img transfer에 좋은 특성이다. </br>
따라서 네트워크 구조는 여려 residual blocks로 구성되어 있으며 각 블록은 두개의 3 * 3컨볼루션을</br>
포함한다. </br>

## 3.2 Perceptual Loss Function

이미지들 사이의 하이레벨 perceptual과 semantic differ를 측정하는 구가지 perceptual loss를 정의한다. </br>
이미지 분류를 위해 사전학습된 네트워크를 이용 즉 perceptual loss는 그 자체로 deep conv neural network이다. </br>
실험에 쓰인건 imgaeNet 데이터 셋에서 학습된 16 - layer vgg network임 </br>

![캡처](https://user-images.githubusercontent.com/43857226/65102547-60311880-da06-11e9-8247-f1a75c3d7f7d.JPG) </br>
> 학습된 VGG - 16 loss network에서 스타일 재구성 loss func를 최소화하는 이미지 ^y을 찾기위해 최적화 사용 </br>
이미지는 스타일의 특징을 보존하지만 공간구조는 보존하지 않음... 

#### Feature Reconstruction Loss

output ^y = fW(x)의 픽셀을 타겟이미지 y의 픽셀과 정확히 매치하는 것보단 loss network에 의해 </br>
계산된 것과 유사한 특징묘사를 갖는것이 좋다. </br> 
이미지 x를 처리할 때 φj(x)가 네트워크 φ의 j 번째 레이어의 활성화가 되도록 한다. </br>
j가 컨볼루션 레이어일 때  φj(x)는 형상 Cj × Hj × Wj의 형상 맵이 된다. 

![캡처](https://user-images.githubusercontent.com/43857226/65114170-bb700480-da20-11e9-9149-1f5d12725ac2.JPG) </br>
feature reconstruction loss는 feature representations사이의 유클리드 거리(?????) 먼말임 ㅅㅂ </br>

그림 3에서 재현한 것처럼 초기(?)레이어에 대해 feature reconstruction loss를 최소화 하는 ^y 이미지를 찾는 것은 </br>
y와 시각적으로 구분되지 않는 이미지를 생산하는 경향이 있다. </br>

상위 레이어에서 재구성 할 때 이미지와 전체적인 spatial structure는 보존되지만 색상,텍스쳐,정확하 모양은 보존 ㄴㄴ </br>
img trnasfer networks를 학습시키기 위해 **feature reconstruction loss** 를 사용하는 것은 </br>
출력이미지 ^y이 **대상이미지 y**와 지각적으로 유사하도록 권장되지만 </br>
정확히 일치하도록 강제는 ㄴㄴ </br> 

#### Style Reconstruction Loss

feature reconstruction loss는 출력이미지 ^y가 타겟 y의 내용에서 벗어날 때(??) 불이익 안좋다 ~~ 이말 </br>
또 다른 목표는 색,질감,패턴등 같은 스타일의 차이의 변화가 있어야 한다. </br>
위같이 Cj * Hj * Wj의 피쳐맵인 인풋 x에 대해 네트워크의 j번째 레이어의 활성화가 되도록 한다.(?????)</br>
gram matrix Gφ j(x)를 다음 공식과 같이 Cj * Cj 매트릭스로 정의한다.</br>

![캡처](https://user-images.githubusercontent.com/43857226/65115446-cc217a00-da22-11e9-9eab-4879bac26982.JPG)</br>
</br>




## 용어정리
**^y : 출력이미지**



