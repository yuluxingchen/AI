import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    X, labels = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=2
    )
    # 设置随机数种子
    rng = np.random.RandomState(2)
    # 对生成的特征数据添加一组均匀分布噪声
    X += 2 * rng.uniform(size=X.shape)
    # 标签类别数
    unique_labels = set(labels)
    # 根据标签类别数设置颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        x_k = X[labels == k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o',
                 markerfacecolor=col,
                 markeredgecolor='k',
                 markersize=14)
    plt.title('Simulated binary data set')
    plt.show()

    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], labels[:offset]
    X_test, y_test = X[offset:], labels[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train = ', X_train.shape)
    print('X_test = ', X_test.shape)
    print('y_train = ', y_train.shape)
    print('y_test = ', y_test.shape)

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)