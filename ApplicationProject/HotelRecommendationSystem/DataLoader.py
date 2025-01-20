import numpy as np
import pandas as pd

from ApplicationProject.HotelRecommendationSystem.DataPreProcessing import DataPreProcessing


class DataLoader:
    def __init__(self, processing):
        self.processing = processing

    def dataset_split(self, test_size=0.25, random_state=None):
        dataset, label = self.processing.pre_processing()
        X = pd.concat(dataset, axis=1)
        y = label

        if random_state is not None:
            np.random.seed(random_state)

        # 计算测试集大小
        n_samples = len(X)
        test_size = int(n_samples * test_size)

        # 生成随机索引
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # 划分训练集和测试集
        train_indices = indices[test_size:]
        test_indices = indices[:test_size]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    processing = DataPreProcessing('data/hotel_bookings.csv')
    loader = DataLoader(processing)
    X_train, X_test, y_train, y_test = loader.dataset_split(0.25, 42)
    print("训练集特征形状：", X_train.shape)
    print("测试集特征形状：", X_test.shape)
    print("训练集标签形状：", y_train.shape)
    print("测试集标签形状：", y_test.shape)

    if X_train.isna().any().any():
        print("数据中包含无效值")
        raise ValueError
    if X_train.applymap(np.isinf).any().any():
        print("数据中包含无穷大值")
        raise ValueError
