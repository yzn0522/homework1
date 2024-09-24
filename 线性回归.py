import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str) -> tuple:

    df = pd.read_csv(path)
    x_array = df["years"].to_numpy()[::-1]
    y_array = df["prices"].to_numpy()[::-1]
    return x_array, y_array


class mylinearregession:
    def __init__(self, learn_speed_rate: float, echo_times: int) -> None:
        self.speed = learn_speed_rate  # 学习率
        self.times = echo_times  # 迭代次数
        self.k = None  # 斜率
        self.y0 = None  # 常数项

    def fit(self, X_array: np.ndarray, Y_array: np.ndarray) -> tuple:
        #梯度下降拟合最优函数
        self.k = 1000;
        self.y0 = -2000000
        n = len(X_array)
        for idx in range(self.times):
            cur_predict = self.predict(X_array)
            # 对损失函数求偏导数
            k_ = (-2 / n) * np.sum((Y_array - cur_predict) * X_array)
            y0_ = (-2 / n) * np.sum(Y_array - cur_predict)

            # 按梯度的反方向更新参数，靠近最小损失点
            self.k -= k_ * self.speed * 0.00001
            self.y0 -= y0_ * self.speed

        # 返回两参数
        return self.k, self.y0

    def loss(self, predict: np.ndarray, X: np.ndarray) -> float:
        # 均方差作为损失函数
        return sum((x - y) ** 2 for x, y in zip(predict, X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        #函数返回预测序列
        return self.k * X + self.y0


if __name__ == "__main__":
    print("开始预测")
    # 加载训练集
    path = "data.csv"
    X, Y = load_data(path)
    # 加载线性回归模型
    model = mylinearregession(learn_speed_rate=0.01, echo_times=100)
    predict_k, predict_y0 = model.fit(X, Y)
    # 预测函数
    prediction_X = np.linspace(2000, 2028, 29)
    prediction_Y = predict_y0 + prediction_X * predict_k
    # 设置字体为支持中文的字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 可视化训练集和预测结果
    plt.figure(figsize=(10, 6))
    plt.title('线性回归预测')
    plt.xlabel('年份')
    plt.ylabel('元每平方米')
    plt.scatter(X, Y, color='red', label='训练集')
    plt.plot(prediction_X, prediction_Y, label='预测')
    plt.show()