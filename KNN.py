from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算已知类别数据集中的点与当前点之间的欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 距离的平方
    sqDiffMat = diffMat ** 2
    # 矩阵的每一行相加求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 距离和的开方
    distances = sqDistances ** 0.5
    # 根据距离递增次序排序
    sortedDistIndicies = distances.argsort()
    # 根据前K个标签找到对应的目标值
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 返回频率最高的类
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print(result)
