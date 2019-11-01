# Axis를 알아보자 
머리 깨진다. 어려운건 아닌데 정말 헷갈린다. 정신 차리자</br>

```python
data = np.array([[[1,11,111],[2,22,222]],
                [[3,33,333],[4,44,444]],
                [[5,55,555],[6,66,666]],
                [[7,77,777],[8,88,888]]])
```

다음은 (4,2,3) 형태의 numpy배열이다. </br>
이걸 그림으로 표현하면 다음과 같다. </br>

![깃허브악시스](https://user-images.githubusercontent.com/43857226/67074410-028a1a80-f1c4-11e9-8a42-c26f47f77d78.png)

누가봐도 미대를 다녔던 필자의 그림실력이다 </br>
axis는 기본적으로 none이다 </br>
axis = 0은 무엇일까 알아보자 </br>

# axis = 0
axis = 0은 기본적으로 row의 제거이다. </br>
이를 반환하면 (2,3)의 형태가 된다. </br>
합산방향은 다음 그림과 같다 </br>

![깃허브악시스0](https://user-images.githubusercontent.com/43857226/67075187-a32d0a00-f1c5-11e9-9145-1e6aef582db9.png)

```python
data0 = data.sum(axis = 0)
array([[  16,  176, 1776],
       [  20,  220, 2220]])
```
이렇게 나온다. </br>
axis = 0 은 가장 바깥 대괄호를 지우는 것과 같다. </br>

# axis= 1 
axis = 1은 기본적으로 col의 제거이다. </br>
이를 반환하면 (4,3)의 형태가 된다. </br>
합산방향은 다음과 같다 </br> 

![깃허브악시스1](https://user-images.githubusercontent.com/43857226/67077456-44b65a80-f1ca-11e9-84ed-9a5d78bed44b.png)


```python
data1 = data.sum(axis = 1)
array([[   3,   33,  333],
       [   7,   77,  777],
       [  11,  121, 1221],
       [  15,  165, 1665]])
```
이랗게 나온다. </br>
axis = 1 은 중간괄호를 지우는 것 같다. </br>

# axis = 2
axis = 2는 깊이의 제거이다. </br>
이를 반환하면 (4,2)의 형태가 된다. </br>
합산방향은 다음과 같다. </br> 

![깃허브악시스2](https://user-images.githubusercontent.com/43857226/67078259-e5594a00-f1cb-11e9-8c36-c5a171c23077.png)

```python
data2 = data.sum(axis =  2)
array([[123, 246],
       [369, 492],
       [615, 738],
       [861, 984]])

```
이렇게 나온다. </br>
axis = 2 는 가장 가까운 괄호를 지우는 것 같다. </br>
