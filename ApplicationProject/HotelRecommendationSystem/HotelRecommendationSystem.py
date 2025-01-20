import numpy as np

from DataLoader import DataLoader
from DataPreProcessing import DataPreProcessing
from Model import TextNet
from Trainer import Trainer

if __name__ == '__main__':
    processing = DataPreProcessing('data/hotel_bookings.csv')
    loader = DataLoader(processing)
    X_train, X_test, y_train, y_test = loader.dataset_split(0.25, 42)
    print("训练集特征形状：", X_train.shape)
    print("测试集特征形状：", X_test.shape)
    print("训练集标签形状：", y_train.shape)
    print("测试集标签形状：", y_test.shape)

    model = TextNet()
    trainer = Trainer(model)
    trainer.compile()
    trainer.fit(X_train, y_train, epochs=30)
    trainer.show()

    model_path = 'save_model/best_model.pth'
    predict = trainer.test(X_test, y_test, model_path)
    print(predict.shape)
    test_acc = np.sum(y_test == predict) / X_test.shape[0]
    print("测试集准确率为：{}".format(test_acc))

    result = trainer.predict(X_test, model_path)
    print(result.shape)
