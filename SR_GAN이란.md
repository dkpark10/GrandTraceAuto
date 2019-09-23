# SR_GAN에 대해서..

## Abstract
초해상도 문제에서 복잡하고 섬세한 텍스쳐 복구에 대한 문제가 남아있다. </br>
기존 방식은 MSE를 사용함으로서 초해상도 문제 해결 </br>
SRGAN의 목적을 간단하게 이해한다면, VDSR, ESPCN과 같이 기존의 </br>
Super Resolution의 경우 Loss를 평가할 때 psnr 방식을 사용하였다.  </br>
하지만 PSNR이 높은 사진일 수록 사람 눈에 선명하게 보이는 것은 아니다. </br>
즉, 사람 눈으로 보기에 선명하게 보이게 만들려는 목적과 기존의 평가 방식은 맞지 않는다. </br>
SRGAN을 사용하게 되면 psnr은 조금 낮더라도, 사람 눈에 보다 정확하게 보이게 학습을 하게 된다. </br>
이 문제를 해결하기 위해 **content loss**와 adversarial loss를 구성하는 perceptual loss </br>
를 사용한다. </br>
adversial loss는 오리지날과 고해상도 이미지 사이 차이점을 구별하도록 훈련된 </br>
discriminator network를 사용하여 해결 </br>
게다가 픽셀의 유사성 대신 perceptual의 유사성을 사용한다. </br>

## Introduction
저해상도이미지로부터 고해상도 이미지를 추정하는 것을 초해상도라 한다. </br>
학습된 SR알고리즘의 최적화 target은 공통적으로 복구된 고해상도 이미지와 </br>
ground - truth 사이의 평균제곱오차(MSE)를 최소화 하는것이다. </br>
MSE를 사용하면 PSNR(주로 영상 또는 동영상 손실 압축에서 화질 손실 정보를 평가할 때 사용되는것.) </br>
이 높아지므로 편리하지만 높은 질감 디테일같은 지각적으로 관련된 차이를 잡아내는</br>
MSE의 성능은 픽셀차이의 기반으로 정의되므로 떨어진다 ~~ </br>
</br>
![캡처](https://user-images.githubusercontent.com/43857226/65397278-b1b41b80-dde9-11e9-816a-bd16382a7b6c.JPG)
그림과 같이 높은 psnr이 반드시 좋은결과를 야기하지는 못한다. </br>
skip - connection을 사용하는 deep residual network(ResNet)와  </br>
단독 **optimization target으로 MSE에서 분기하는** SRGAN을 제시 </br>
</br>
이전 작업과 다르게 HR reference images로 지각적으로 구별하기 어려운 해결책을 권장하는(?????) </br>
discriminator와 결합된 VGG network의 high-level feature map을 사용하여 </br>
perceptual loss를 정의한다. </br>

## Related work 
## pass ~~~~ continue

## Method
SISR의 목표는 인풋으로 들어온 저해상도 이미지로부터 </br>
super-resolved img와 high-resolution을 추정하는 것. </br>
여기서 img L-R은 high-resolution이미지를 저해상도 처리화 한것 </br>
고해상도 이미지는 훈련중에만 사용가능 ????? </br>
트레이닝 중 저해상도 이미지는 고해상도이미지에 다운샘플링 후 가우시안 필터를 적용해서 얻은것 </br>
목표는 주어진 저해상도 입력에 대응하는 고해상도 이미지를 추정하는 Generator를 학습하는것. </br>
이를 달성하기 위해 θG의해 파라미터화된 CNN GθG 피드포워드로서 generator network를학습한다.</br>
여기서 θG = {W1:L; b1:L}은 L-layer deep network의 가중치와 편향을 나타내며 </br>
SR고유 손실함수를 최적화 하여 구한다. </br>
n = 1,2,3,... N >> imgHR[n] 에 대응하는 imgLR[n]에 대해서 </br>

![캡처](https://user-images.githubusercontent.com/43857226/65398003-6ef54200-ddef-11e9-9153-732b774cd885.JPG)
> 대충 1 ~ n 까지 저해상도와 고해상도이미지의 지각적차이를 1/N로 나눈 loss func인듯... </br>
</br>
이 작업에서 여려 loss 함수로 조합된 가중치 조합으로 perceptual loss를 설계한다.  </br>
여기서 loss 함수들은 복구된 고해상도 이미지의 특성을 확실하게 구별할 loss func이다. </br>

#### 2.1 Adversial network architecture
이안 굿펠로우의 제안된 적대적 네트워크 손실함수는 minmax game이며 다음과 같다.

![캡처](https://user-images.githubusercontent.com/43857226/65398175-bb8d4d00-ddf0-11e9-9656-3d7be67c87cb.JPG)
</br>
이 공식은 가짜와 진짜를 구별하기 위해 훈련된 discriminator 'D'를 속이는것을 목표로 </br>
생성모델 generator 'G'를 훈련시킬 수 있다는 것이다. </br>
이 방법을 통해 G는 실제와 유사하게 D가 구별하지 못하도록 할 수 있다. </br>
이건 MSE와 같이 픽셀단위로 loss값을 최소화시켜 얻는 SR과 대조적이다. </br>
generator G의 핵심은 그림 4에 동일한 레이아웃으로 B residual blocks에 예시되어있다. </br>

![캡처](https://user-images.githubusercontent.com/43857226/65398422-1a06fb00-ddf2-11e9-9c27-85c77927db97.JPG)
</br>
batch normalization 레이어들과 파라메트릭 렐루를 활성화 함수로 사용 후 3 * 3 커널과 </br>
64 featuremap(*CNN에서 filter를 거치고 나온 이미지 맵*)을 사용하는 두개의 컨볼루션 </br>
레이어를 사용한다. </br>
Shi에 제안된 두개의 학습된 서브픽셀 컨볼루션 레이어를 사용하여 이미지의 해상도를 높인다????</br>
생성된 SR 샘플로부터 실제 고해상도 이미지를 구분하기 위해 D를 학습한다. </br>
구조는 그림 4에 나와있다. </br>
리키렐루를 (alpha = 0.2)사용하고 전체네트워크에 맥스풀링을 방지(?)피한다(?).. </br>
D network는 방정식2의 maxmin문제를 해결하기 위해 학습된다. </br>
</br>
3 * 3필터 커널의 수가 증가하는 8개의 컨볼루션레이어를 포함하며 VGG network에서 64에서 </br>
512로 커널로 2배증가(????) **좀 봐야겠다** </br> 
피쳐가 2배 될 때마다 이미지 해상도를 줄이기 위해 스트라이드 컨볼루션을 사용한다. </br>
512개의 형상 맵은 후에 두 개의 밀집된 층과  sample classfication을 위한 확률을 </br>
얻기 위해 마지막 시그모이드 활성화 함수가 뒤따른다(????) </br> 

#### 2.2 Perceptual loss fucntion
ISR은 네트워크 발전에 중요함. </br>
일반적으로 ISR은 MSE에 기반하지만 지각적으로 관련있는 특성을 평가하는 손실함수를 </br>
평가한다. </br>
지각 함수는 content loss(ISRx)의 가중치함과 adversial loss func로 구성한다.</br>

![캡처](https://user-images.githubusercontent.com/43857226/65402458-745f8600-de09-11e9-997e-e3367b63741b.JPG)
</br>

##### 2.2.1 Content loss
픽셀기반 MSE loss는 다음과 같이 계산한다.</br>
![캡처](https://user-images.githubusercontent.com/43857226/65402514-d1f3d280-de09-11e9-8d4b-ba32231019c2.JPG)
</br>
MSE는 SR에 많이 적용되는 방법이기는 하나 </br>
높은 PSNR(주로 영상 또는 동영상 손실 압축에서 화질 손실 정보를 평가할 때 사용하는 것.) </br>
에도 불구하고 MSE의 최적화문제는 지각적으로 사람이 보기에 만족스럽지 못한 </br>
뭉개진 텍스처의 문제를 야기한다. </br>
MSE대신 perceptual 유사성에 가까운 손실함수를 사용한다. </br>
**미리 학습된 19계층 VGG network의 활성화 레이어 렐루를 기반으로 VGG loss를 정의한다.** </br>
**φi, j를 사용하여 VGG19 네트워크 내에 i번째 maxpooling 계층 이전의 j번째 컨볼루션**</br>
**(활성 후)으로 얻은 feature map을 나타냅니다.** </br>
VGG loss를 reconstructed img(ILR)과 reference img(IMR)의 feature representations을
유클리드 거리로 정의한다. </br>
</br>
![캡처](https://user-images.githubusercontent.com/43857226/65402905-4891cf80-de0c-11e9-8064-0bf268410f6d.JPG)
</br>
여기서 W,ij와 H,ij는 VGG network내의 각각의 피쳐맵의 차원을 나타낸다. </br>

##### 2.2.2 Adversial loss
네트워크가 discriminator를 속이기 위해 이미지의 다양한 특징에 있는 방법을 알도록 선호한다. </br>
generative loss(ISR)은 정의된다 모든 트레이닝 샘플에 discriminator가 구별하는 것으로 </br>

![캡처](https://user-images.githubusercontent.com/43857226/65403191-c4d8e280-de0d-11e9-8435-69c0a3c554f3.JPG)
</br>
여기서 D(G(ILR))은 생성한 이미지 G(ILR)가 자연스런 이미지일경우 (ex: 0.5에 수렴) </br>
더 나은 gradient 를 위해 **log[1 - D(G(ILR))]** 대신에 **-logD(G(ILR))** 을 최소화 한다. </br> 

**주요점**
1. loss func(perceptual loss)
2. residual blocks and skip-connection
3. maxpooling 대신 strides = 2 convnet 사용
4. LeakyReul(alpha = 0.2)사용
5. 학습된 VGGnet이용 featuremap 사이 유클리드 거리를 구함

## Experiments
#### 3.1 Data and similarity measures
dataset Set5, Set14, BSD100을 데이터 셋으로 사용 </br>
모든 실험은 저해상도와 고해상도사이에서 4배 스케일로 수행. </br>
이는 이미지를 16배 감소한것. *(ex: 2 * 2 = 4 , 8 * 8 = 64, 64/4 = 16)* </br>
그냥 데이터 이것저것 준비했다 ~~~ 이말임 </br>

#### 3.2 Training details and parameters
imageNet 데이터베이스에서 무작위로 350k 이미지를 사용하여 훈련 </br>
LR img를 다운샘플링 계수 r = 4인 바이큐빅 커널을 사용하여 얻음. </br>
미니 배치에 대해 96 * 96 HR 하위이미지 16개를 자른다. </br>
generator model을 임의 크기로 변환할 수 있다. </br>
LR은 0,1 사이 HR은 -1,1사이이다 ~ </br>
따라서 MSE loss는 -1, 1 사이로 계산된다. </br>
대충 러닝레이트 몇으로 하고 몇번 학습했다 ~~~ </br>
G와 D를 k = 1로 동등하게 맞춤 
G는 16개의 동일한 residual blocks를 사용 
학습하는동안 batch normalazation을 off하여 input에만 의존하는 출력을 얻는다. </br>

#### 3.3 Mean opinion score(MOS) testing
**일단 패쓰으으으**

#### 3.4 Investigation of content loss
GAN에 기반한 perceptual loss에 다양한 content loss 차이점의 영향을 조사 
특별히 





