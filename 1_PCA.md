# 1. PCA 실습

## Import numpy


```python
import numpy as np
```

## 데이터 생성


```python
mean = np.array([2, 3])
cov = np.array([[2, 1], [1, 2]])
```


```python
data = np.random.multivariate_normal(mean, cov, (1000))
x = data[..., 0]
y = data[..., 1]
```

## 주어진 공분산 행렬로부터 eigenvalue와 eigenvector 구하기


```python
eig, P = np.linalg.eig(cov)
print("Original Eigenvalues are :", eig)
```


```python
print(f"First eigenvector (Corresponding eigenvalue {eig[0]}) :", list(P[1]))
print(f"Second eigenvector (Corresponding eigenvalue {eig[1]}):", list(P[0]))
```

## 데이터 시각화


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(x, y, 'bo')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.show()
```

## PCA - 1. 평균을 원점으로 이동


```python
x_centralized = x - np.mean(x)
y_centralized = y - np.mean(y)
```


```python
plt.figure(figsize=(8,8))
plt.plot(x_centralized, y_centralized, 'bo')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
```


```python
print(x_centralized.shape)
print(y_centralized.shape)
```

## PCA - 2. 데이터로부터 공분산 행렬 추출


```python
xy_centralized = np.array([[x_centralized[i], y_centralized[i]] for i in range(len(x_centralized))])
```


```python
print(xy_centralized.shape)
```


```python
cov_data = np.matmul(xy_centralized.T, xy_centralized) / len(xy_centralized)
```


```python
print(cov_data)
```


```python
Sigma, Q = np.linalg.eig(cov_data)
```

## PCA - 3. 데이터로부터 획득한 공분산 행렬의 고유값 확인


```python
print(Sigma)
```


```python
print(Q)
```

## PCA - 4. 획득한 행렬 $Q$가 직교행렬(orthogonal matrix)인지 확인


```python
print(np.matmul(Q.T, Q))
print(np.matmul(Q, Q.T))
```

## PCA - 5. $Q$행렬을 축으로 분해. $Q$행렬의 각 column이 eigenvector이 됨


```python
first_axis = Q[1]
second_axis = Q[0]
```


```python
first_axis
```


```python
plt.figure(figsize=(8,8))
plt.plot(x_centralized, y_centralized, 'bo', alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.arrow(0, 0, Sigma[0] * first_axis[0], Sigma[0] * first_axis[1], width=0.1, color='r')
plt.arrow(0, 0, Sigma[1] * second_axis[0], Sigma[1] * second_axis[1], width=0.1, color='r')
plt.show()
```


```python

```
