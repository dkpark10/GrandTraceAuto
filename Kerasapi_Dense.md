# 인공신경망 구조
![캡처](https://user-images.githubusercontent.com/43857226/65927884-80b19780-e436-11e9-9676-71006fc7dd2e.PNG)
</br>

x0, x1, x2 : 입력되는 뉴런의 축삭돌기로부터 전달되는 신호의 양 입력데이터</br>
w0, w1, w2 : 시냅스의 강도, 즉 입력되는 뉴런의 영향력을 나타냅니다. 가중치</br>
w0x0 + w1x1 + w2*x2 : 입력되는 신호의 양과 해당 신호의 시냅스 강도가 곱해진 값의 합계</br>
f : 최종 합계가 다른 뉴런에게 전달되는 신호의 양을 결정짓는 규칙, 활성화 함수</br>

# Dense layer

Dense layer는 입출력을 모두 연결 입력이 4 출력이 8이라면 가중치는 32개 </br>
> 가중치가 높을수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고 낮을수록 적다. </br>
</br>

예를 들어 성별을 판단하는 문제있어서, 출력 뉴런의 값이 성별을 의미하고, 입력 뉴런에 머리카락길이, </br>
키, 혈핵형 등이 있다고 가정했을 때, 머리카락길이의 가중치가 가장 높고, 키의 가중치가 중간이고, </br>
혈핵형의 가중치가 가장 낮다. 딥러닝 학습과정에서 이러한 가중치들이 조정된다.</br>
</br>
`Dense(8, input_shape =(100,), init = 'uniform', activation = 'relu')`
</br>
* 첫번째 인자 = 출력의 수</br>
* input_shape = 입력의 수</br>
* init =  가중치 초기화 방법 설정</br>
  * uniform = 균일분포
  * normal = 가우시안 분포
* activation = 활성화함수
  * linear = 기본값
  * relu, sigmoid, softmax
  
</br>
덴스레이어는 입력수에 상관없이 출력을 자유롭게 설정할 수 있다. </br>
`Dense(1, input_shape = 3, acti = 'sigmoid)`
</br>
이를 그림으로 표현해보자 </br>

![캡처](https://user-images.githubusercontent.com/43857226/65928380-b2c3f900-e438-11e9-83e9-bf6414abc042.PNG)
</br> 

다중 클래스 분류문제에서 클래스 수만큼 출력이 필요하다. 만약 3가지로 분류된다면 아래코드처럼</br>
출력이 3개이고 계산값을 각 클래스의 확률개념으로 표현할 수 있는 softmax를 사용 </br>
`Dense(3, input_shape = 3, acti = 'softmax')`

</br>
![캡처](https://user-images.githubusercontent.com/43857226/65934098-11946d00-e44f-11e9-83bb-dbf084819ba7.PNG)
</br>

입력은 4개이며 출력은 3개이므로 시냅스 개수(곱한것)은 12개이다. </br>
Dense layer는 히든레이어 및 인풋레이어로 많이 쓰인다. </br>
이럴떄 보통 **relu**를 사용 </br>
`Dense(4, input_shape = 6, acti = 'relu')`
</br>
이를 표시하면 다음과 같다. </br>

![캡처](https://user-images.githubusercontent.com/43857226/65934380-1b6aa000-e450-11e9-84f1-0f315a6a3237.PNG)
</br>

인풋레이어가 아닐시 이전층의 출력수를 알 수 있기 때문에 input_shape를 지정하지 않아도 된다. </br>
다음 코드는 인풋에만 input_shape를 지정하고 이후 지정하지 않는다. </br>

</br>
```python
model.add(Dense(8, input_dim=4, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
```

이 코드를 표시하면 다음 아래 그림과 같다. </br>

![캡처](https://user-images.githubusercontent.com/43857226/65934467-8d42e980-e450-11e9-8474-f270704f3309.PNG)
</br>

다음구조는 입력이 4이고 출력이 0 ~ 1 사이를 나오는 모델임을 알 수 있다.</br>
쌓았던 블록을 실제로 구현해보장 </br> 

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_shape = 4, init = 'uniform', activation = 'relu'))
model.add(Dense(6, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
```
