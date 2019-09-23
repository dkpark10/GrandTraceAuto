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















