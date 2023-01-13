import numpy as np


def euclidean_distance(x, y):
    """
    欧式距离
    :param x: 向量 x
    :param y: 向量 y
    :return: 欧式距离
    """
    # 初始化距离
    distance = 0
    # 计算距离平方和
    for i in range(len(x)):
        distance += pow((x[i] - y[i]), 2)
    return np.sqrt(distance)


def centroids_init(X, k):
    """
    定义质心初始化函数
    :param X: 训练样本，Numpy数组
    :param k: 质心个数
    :return: 质心矩阵
    """
    # 样本数和特征数
    m, n = X.shape
    # 初始化质心矩阵， 大小为质心个数 * 特征数
    centroids = np.zeros((k, n))
    # 遍历
    for i in range(k):
        # 每一次循环随机选择一个类中心作为质心向量
        centroid = X[np.random.choice(range(m))]
        centroids[i] = centroid
    return centroids


def closest_centroid(x, centroids):
    """
    定义样本所属最近质心的索引
    :param x: 单个样本实例
    :param centroids: 质心矩阵
    :return: 最近索引
    """
    closest_i, cloest_dist = 0, float('inf')
    for i, centroid in enumerate(centroids):
        distance = euclidean_distance(x, centroid)
        if distance < cloest_dist:
            closest_i = i
            cloest_dist = distance
    return closest_i


def build_clusters(centroids, k, X):
    """
    分配样本与构造簇
    :param centroids: 质心矩阵
    :param k: 质心个数
    :param X: 训练样本
    :return: 聚类簇
    """
    # 初始化簇列表
    clusters = [[] for _ in range(k)]
    # 遍历训练样本
    for x_i, x in enumerate(X):
        # 获取样本所属最近质心的索引
        centroid_i = closest_centroid(x, centroids)
        # 将当前样本添加到所属类簇中
        clusters[centroid_i].append(x_i)
    return clusters


def calculate_centroids(clusters, k, X):
    """
    计算当前质心
    :param clusters: 上一步的聚类簇
    :param k: 质心个数
    :param X: 训练样本
    :return: 更新后的质心矩阵
    """
    # 特征数
    n = X.shape[1]
    # 初始化质心矩阵, 大小为质心个数*特征数
    centroids = np.zeros((k, n))
    # 遍历当前簇
    for i, cluster in enumerate(clusters):
        # 计算每个簇的均值作为新的质心
        centroid = np.mean(X[cluster], axis=0)
        # 将质心向量分配给质心矩阵
        centroids[i] = centroid
    return centroids


def get_cluster_labels(clusters, X):
    """
    获取每个样本所属的聚类类别
    :param clusters: 当前的聚类簇
    :param X: 训练样本
    :return: 预测类别
    """
    # 预测结果初始化
    y_pred = np.zeros(X.shape[0])
    # 遍历聚类簇
    for cluster_i, cluster in enumerate(clusters):
        # 遍历当前簇
        for sample_i in cluster:
            # 为每个样本分配类别簇
            y_pred[sample_i] = cluster_i
    return y_pred


def kmeans(X, k, max_iterations):
    """
    k 均值聚类算法流程封装
    :param X: 训练样本
    :param k: 质心个数
    :param max_iterations: 最大迭代次数
    :return: 预测类别列表
    """
    # 初始化质心
    clusters = None
    centroids = centroids_init(X, k)
    # 遍历迭代求解
    for _ in range(max_iterations):
        # 根据当前质心进行聚类
        clusters = build_clusters(centroids, k, X)
        # 保存当前质心
        cur_centroids = centroids
        # 根据聚类结果计算新的质心
        centroids = calculate_centroids(clusters, k, X)
        # 设定收敛条件为质心是否变化
        diff = centroids - cur_centroids
        if not diff.any():
            break
    # 返回最终的聚类标签
    return get_cluster_labels(clusters, X)


if __name__ == '__main__':
    X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
    label = kmeans(X, 2, 10)
    print(label)
