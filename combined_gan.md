# gan 구조
![캡처](https://user-images.githubusercontent.com/43857226/65736274-6796a700-e115-11e9-818e-c8c4d74ab6b8.PNG)
</br>

딥러닝의 기본 프로세스는 가설정의 -> 손실정의 -> 최적화정의로 이루어진다. </br>
판별기는 입력 이미지가 진짜 이미지인지를 분류해 낸다. 이 분류는 적대 훈련 중에 일어날 것이다. </br>
본질적으로, 판별기는 신경망의 순방향 전파가 이뤄지는 동안에 입력을 분류한다.</br>
생성기는 임의의 벡터공간(잠재공간)에서 가져와 데이터를 생성. </br>
</br>
G와 D를 만들면 이를 합치는 적대모델도 만들어야 한다.</br>


```python
import sys
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU


class Trainer:
    def __init__(self, width=28, height=28, channels=1, latent_size=100, epochs=50000, batch=32, checkpoint=50,
                 model_type=-1):

        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type
        self.LATENT_SPACE_SIZE = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

        self.load_MNIST()

    def load_MNIST(self, model_type=3):
        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        (self.X_train, self.Y_train), (_, _) = mnist.load_data()          # X_train은 mnist 한장의 ndarray (60000,28,28)
                                                                          # Y_train은 mnist 한장의 숫자임 길이는 60000 (60000,)
        
        if self.model_type != -1:                                         # -1 아닐시 model_type숫자만 뽑는다 ~
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        # Rescale -1 to 1
        # Find Normalize Function from CV Class
        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5         # 0 ~ 255값을 0 ~ 1로 스케일링
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def train(self):
    
        for e in range(self.EPOCHS):                                                                # EPOCHS 반복횟수
           
            # Train Discriminator
            # Make the training batch for this model be half real, half noise
            # Grab Real Images for this training batch

            # 훈련데이터셋에서 랜덤이미지로 구성된 배치 한개를 가져와 x_real_img, y_real_label 생성
            count_real_images = int(self.BATCH / 2)                                                 # 16
            starting_index = randint(0, (len(self.X_train) - count_real_images))                    # 0 ~ (60000-16)
            real_images_raw = self.X_train[starting_index: (starting_index + count_real_images)]    # 16단위로 슬라이스
            x_real_images = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)      # (16,28,28,1)
            y_real_labels = np.ones([count_real_images, 1])                                         # 1로 이루어진 (16,1)행렬

            # Grab Generated Images for this training batch                                         # 임의공간에서 G 추출
            latent_space_samples = self.sample_latent_space(count_real_images)                      # 평균 0 표준편차1인 (16,100)행렬
            x_generated_images = self.generator.Generator.predict(latent_space_samples)             # 모델의 사용 (??)
            y_generated_labels = np.zeros([self.BATCH - count_real_images, 1])                      # 0으로 이루어진 (16,1)행렬

            # Discriminator에서 훈련용으로 결합한다. 
            x_batch = np.concatenate([x_real_images, x_generated_images])                           # x_real과 x_gene 합침
            y_batch = np.concatenate([y_real_labels, y_generated_labels])                           # y_rabel과 y_gene_label 합침
                
            # 판별기를 훈련 하고 있다. 훈련될 때 이미지가 가짜라는걸 알고 있으므로 판별기는
            # 생성된 이미지와 진짜사이 결함을 찾으려 한다.


            # Now, train the discriminator with this batch                                          # discriminator 학습
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0] 
            


            # Generate Noise
            x_latent_space_samples = self.sample_latent_space(self.BATCH)                           # (32,100)행렬 무작위 표본 추출
            y_generated_labels = np.ones([self.BATCH, 1])                                           # 1로이루어진 (16,1)행렬
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels) 

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')

            if e % self.CHECKPOINT == 0:                                                            # 50번마다 모델 저장
                self.plot_checkpoint(e)
        return

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))                 # 0에 가깝고 표준편차가 1인 (16,100)행렬

    def plot_checkpoint(self, e):
        filename = "data/sample_" + str(e) + ".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return


class GAN(object):                                                               # GAN은 g,d,loss로 구성된다.
    
    def __init__(self, discriminator, generator):
       
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.Generator = generator

        self.Discriminator = discriminator                                         
        self.Discriminator.trainable = False

        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self):                                                            # model은 심층신경망을 만든다.
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='data/GAN_Model.png')


class Generator(object):                                                        # 'G'는 간단한 순차모델
                                                                                # 순차모델이란 신경망에서 계층들을 순서대로 구성하고
                                                                                # 연결하는 것.
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.W = width
        self.H = height
        self.C = channels
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.LATENT_SPACE_SIZE = latent_size                                    # 100
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)  # GAN의 손실함수는 이진크로스엔트로피 사용
                                                                                      # 최적화 기법은 아담옵티마이저
        self.save_model()
        self.summary()

    def model(self, block_starting_size=128, num_blocks=4):                     
        model = Sequential()

        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        for i in range(num_blocks - 1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.W * self.H * self.C, activation='tanh'))                   # 인풋과 아웃풋을 동일크기로 재구성후 모델반환
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='data/Generator_Model.png')            # png이미지로 모델구조 저장


class Discriminator(object):                                                        # 'D'는 간단한 이진선형분류기.
   
   def __init__(self, width = 28, height= 28, channels = 1, latent_size=100):
   
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
                                                                                      # gan의 기본예제이기 때문에 내장된 손실함수 사용
                                                                                      # 여러 간모델에 따라 다양한 손실함수를 정의해야한다.
        self.save_model()
        self.summary()

    def model(self):                                                                  # 모델함수는 심층신경망을 만든다.
    
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))                                    # 이 계층은 데이터를 단일 스트림으로 전개
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))                       # dense 계층은 이전 계층과 완전히 연결된 계층
                                                                                      # 입력이 각 뉴런에 도달할 수 있게 한다.
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2)))                                        # 용량을 줄여 중요특징들을 학습하게 한다.
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='data/Discriminator_Model.png')    # png이미지로 모델구조 저장


if __name__ == '__main__':

    HEIGHT = 28
    WIDTH = 28
    CHANNEL = 1
    LATENT_SPACE_SIZE = 100
    EPOCHS = 50001
    BATCH = 32
    CHECKPOINT = 500
    MODEL_TYPE = -1

    trainer = Trainer(height=HEIGHT, \
                      width=WIDTH, \
                      channels=CHANNEL, \
                      latent_size=LATENT_SPACE_SIZE, \
                      epochs=EPOCHS, \
                      batch=BATCH, \
                      checkpoint=CHECKPOINT,
                      model_type=MODEL_TYPE)

    trainer.train()
```
