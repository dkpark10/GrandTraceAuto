# Conv2d에 대해 알아보자

```python
Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
```
필터가 탑재되어있는 conv2D의 인자를 알아보자 </br>
1. 첫번째 인자 : 컨볼루션 필터의 수</br>
2. 두번째 인자 : 컨볼루션 커널의 (행, 열)</br>
3. padding : 경계 처리 방법을 정의합니다.
  * valid : 유효한 영역만 출력. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작다.
  * same : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일
4. input_shape : 샘플 수를 제외한 입력 형태를 정의. 모델에서 첫 레이어일 때만 정의하면 된다. 
(행, 열, 채널 수)로 정의. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정
5. activation : 활성화 함수 설정
  * linear : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력
  * relu : rectifier 함수, 은익층에 주로 사용
  * sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 사용
  * softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 사용
</br>
인풋 이미지는 채널 수가 1, 가로 3 픽셀, 세로가 3 픽셀이고, </br>
크기가 2 x 2인 필터가 하나인 경우를 레이어로 표시하면 다음과 같다. </br>

```python
Conv2D(1, (2, 2), padding = 'valid', input_shape = (3, 3, 1))
```
</br>

![캡처](https://user-images.githubusercontent.com/43857226/67059659-e9b64080-f194-11e9-9553-efb48ada5e37.JPG)

</br>

필터는 가중치를 의미한다. 하나의 필터가 입력 이미지를 돌면서 적용된 값을 모으면 출력 이미지가 생성된다.</br> 
여기에는 두 가지 특성이 있습니다.</br>
하나의 필터로 입력 이미지를 순회하기 때문에 순회할 때 적용되는 가중치는 모두 동일하다. </br>
이를 파라미터 공유라고 부른다. </br>
이는 학습해야할 가중치 수를 현저하게 줄여준다.</br>
출력층의 각 값은 필터로 이미지를 추출해낸 특징을 가지고 있다.</br>
먼말이냐면 y0은 x0,x1,x3,x4의 특징을 가지고 있고 </br>
y1은 x1, x2, x,4, x5의 특징을 가지고 있다는 뜻이다. </br>

