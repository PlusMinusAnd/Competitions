import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit

# x 값: 실수 영역
x_vals = np.linspace(-10, 10, 500)
sigmoid_vals = expit(x_vals)

# p 값: (0, 1) 사이 확률
p_vals = np.linspace(0.001, 0.999, 500)
logit_vals = logit(p_vals)

# 시각화
plt.figure(figsize=(12, 5))

# 시그모이드 함수
plt.subplot(1, 2, 1)
plt.plot(x_vals, sigmoid_vals, color='blue')
plt.title("Sigmoid Function: σ(x) = 1 / (1 + e^{-x})")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)

# 로짓 함수
plt.subplot(1, 2, 2)
plt.plot(p_vals, logit_vals, color='red')
plt.title("Logit Function: log(p / (1 - p))")
plt.xlabel("p (Probability)")
plt.ylabel("logit(p)")
plt.grid(True)

plt.tight_layout()
plt.show()
