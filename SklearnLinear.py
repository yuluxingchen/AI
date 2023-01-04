from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

if __name__ == '__main__':
    diabetes = load_diabetes()
    data, target = diabetes.data, diabetes.target
    X, y = shuffle(data, target, random_state=13)
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print("X_train's shape: ", X_train.shape)
    print("X_test's shape: ", X_test.shape)
    print("y_train's shape: ", y_train.shape)
    print("Y_test's shape: ", y_test.shape)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # 输出模型剧方误差
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # 计算R2系数
    print("R Square score: %.2f" % r2_score(y_test, y_pred))