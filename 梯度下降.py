import numpy as np
import matplotlib.pyplot as plt


class Sin_regression:
    def __init__(self, learn_speed_rate, echo_times):
        self.speed = learn_speed_rate  # 学习率
        self.times = echo_times  # 迭代次数
        self.c = np.random.rand(6)  # 随机初始化参数

    # 梯度下降求最小损失
    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        c = self.c
        for idx in range(1, self.times + 1):
            # 预测序列
            predict_Y = c[0] + c[1] * X + c[2] * X ** 2 + c[3] * X ** 3 + c[4] * X ** 4 + c[5] * X ** 5
            # 打印损失函数值
            if idx % 100 == 0:
                print(f"time: {idx}, loss: {self.loss(predict_Y, Y):.4f}")
            # 计算偏导
            temp = np.zeros(6)
            for i in range(6):
                temp[i] = np.mean(2 * (predict_Y - Y) * (X ** i))
            # 梯度
            self.c -= self.speed * temp
        return self.c

    def loss(self, predict_Y: np.ndarray, Y: np.ndarray) -> float:
        return np.mean((predict_Y - Y) ** 2)


if __name__ == "__main__":
    # 生成数据
    X = np.linspace(-np.pi, np.pi, 100)
    Y = np.sin(X)

    model = Sin_regression(learn_speed_rate=0.0001, echo_times=10000)
    c = model.fit(X, Y)
    prediction_Y = c[0] + c[1] * X + c[2] * X ** 2 + c[3] * X ** 3 + c[4] * X ** 4 + c[5] * X ** 5

    # 设置字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.title('梯度下降拟合')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y, color='red', label='sin(x)')
    plt.plot(X, prediction_Y, linestyle='--', color='orange', label='预测')
    plt.show()