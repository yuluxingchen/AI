import numpy as np


class EM:
    def __init__(self, thetas, epochs=50, eps=1e-3):
        """
        :param thetas: 初始化的估计参数值
        :param epochs: 最大迭代次数
        :param eps: 收敛阈值
        """
        self.thetas = thetas
        self.epochs = epochs
        self.eps = eps

    def train(self, X):
        # 保存上一个似然函数
        log_like_old = 0
        for i in range(self.epochs):
            # E步，求隐变量分布
            # 对数似然
            log_like = np.array([np.sum(X * np.log(theta), axis=1) for theta in self.thetas])
            # 似然
            like = np.exp(log_like)
            # 求隐变量分布
            ws = like / like.sum(0)
            # 概率加权
            vs = np.array([w[:, None] * X for w in ws])
            # M步，更新参数值
            self.thetas = np.array([v.sum(0) / v.sum() for v in vs])
            # 更新似然函数
            log_like_new = np.sum([w * like for w, like in zip(ws, log_like)])
            print("Iteration: %d" %(i+1))
            print("theta_B = % .2f, theta_C = %.2f, log_like = %.2f" % (thetas[0, 0], thetas[1, 0], log_like_new))
            # 如果到达误差运行范围内则停止迭代
            if np.abs(log_like_new - log_like_old) < self.eps:
                break
            log_like_old = log_like_new
        return self.thetas


if __name__ == '__main__':
    # 观测数据，5次独立试验，每次试验10次抛掷的正反面次数
    # 比如第一次试验为5次正面、5次反面
    observed_data = np.array([(5, 5), (9, 1), (8, 2), (4, 6), (7, 3)])
    # 初始化参数值，即硬币B出现正面的概率为0.6，硬币C出现正面的概率为0.5
    thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
    # EM算法寻优
    em = EM(thetas, epochs=30, eps=1e-3)
    thetas = em.train(observed_data)
    # 打印最优参数值
    print(thetas)
