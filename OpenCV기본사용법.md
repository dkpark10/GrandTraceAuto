# OpenCV 기본 사용법 모음

## cv2.imread
이미지를 읽어 온다.
```python
cv2.imread(flagname,flag)
img = cv2.imread('testimg.jpg', cv2.IMREAD_COLOR)
```
flagname은 이미지 파일의 경로</br>
flag는 이미지를 읽을 때 옵션</br>
return값은 이미지 행렬</br>
return type은 numpy</br>

```python
cv2.IMREAD_COLOR
cv2.IMREAD_GRAYSCALE
cv2.IMREAD_UNCHANGED
```
cv2.IMREAD_COLOR : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며, Default값입니다. </br>
cv2.IMREAD_GRAYSCALE : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간단계로 많이 사용합니다. </br>
cv2.IMREAD_UNCHANGED : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.</br>

```python
img.shape() # 720,1280,3
```
720 = y축</br>
1280 = x축</br>
3 = 컬러 채널</br>
