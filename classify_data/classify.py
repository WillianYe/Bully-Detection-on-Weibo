from prepro_data.prepro import *
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from classify_data.network import *
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MyDataset(Dataset):
    '''
    用于把数据变成tensorflow数据集形式输入神经网络
    '''

    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.datas)


def usingNetwork(filename, modelname, embedding_type, model_type):
    '''
    使用神经网络进行分类
    :param filename: 要分类的数据集路径
    :param modelname: 使用的词嵌入模型路径
    :param embedding_type: 使用的词嵌入方法
    :param model_type: 使用的机器学习分类模型
    :return:
    classifier: 机器学习分类器
    True: 表示用神经网络分类
    '''
    global classifier
    X_train, X_test, y_train, y_test = processData(filename, modelname, embedding_type)
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    np.random.seed(0)
    torch.manual_seed(0)
    if model_type == 'FCN':
        classifier = nnNet()
    elif model_type == 'CNN':
        classifier = cnnNet()
    elif model_type == 'LSTM':
        classifier = LSTM()
    criterion = nn.CrossEntropyLoss()
    # criterion=nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
    for epoch in range(EPOCH_TIME):
        train_correct, train_total, test_correct, test_total = 0, 0, 0, 0
        loss, accuracy = 0, 0
        for datas, labels in train_loader:
            optimizer.zero_grad()
            if model_type == 'CNN' :
                datas = datas.numpy()
                datas = np.array([data.reshape(3, 10, 10) for data in datas])
                datas = torch.from_numpy(datas)
            outputs = classifier(datas, is_training=True)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accuracy = 100 * torch.true_divide(train_correct, train_total)
        print("epoch:", epoch, " train loss:", loss, " train accuracy:", accuracy)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        loss, accuracy = 0, 0
        with torch.no_grad():
            for datas, labels in test_loader:
                if model_type == 'CNN':
                    datas = datas.numpy()
                    datas = np.array([data.reshape(3, 10, 10) for data in datas])
                    datas = torch.from_numpy(datas)
                outputs = classifier(datas, is_training=False)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                accuracy = 100 * torch.true_divide(test_correct, test_total)
            print("test loss:", loss, " test accuracy:", accuracy)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
    # torch.save(classifier,"../model/"+embedding_type+"_"+classifiername)
    plt.figure()
    p1, = plt.plot(train_loss, 'r')
    p2, = plt.plot(train_accuracy, 'b')
    p3, = plt.plot(test_loss, 'g')
    p4, = plt.plot(test_accuracy, 'y')
    plt.legend([p1, p2, p3, p4], ["train loss", "train accuracy", "test loss", "test accuracy"], loc='center right')
    plt.show()
    return classifier, True


def usingTraditionalClassifier(filename, modelname, embedding_type, model_type):
    '''
    使用传统机器学习算法分类
    :param filename: 要分类的数据集路径
    :param modelname: 使用的词嵌入模型路径
    :param embedding_type: 使用的词嵌入方法
    :param model_type: 使用的机器学习分类模型
    :return:
    classifier: 机器学习分类器
    False: 表示用传统方法分类
    '''
    global classifier
    X_train, X_test, y_train, y_test = processData(filename, modelname, embedding_type)
    if model_type == 'svm':
        classifier = svm.SVC(C=1, kernel='rbf', decision_function_shape='ovo')
    elif model_type == 'decision tree':
        classifier = DecisionTreeClassifier(max_depth=11)
    elif model_type == 'naive bayes':
        classifier = GaussianNB()
    elif model_type == 'random forest':
        classifier = RandomForestClassifier(random_state=0)
    elif model_type == 'logistic regression':
        classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))
    return classifier, False


def demo(classifier, filename, classifiername, flag):
    '''
    用于输入某个评论演示分类的效果
    :param classifier: 机器学习分类器
    :param filename: 分类的数据集
    :param classifiername: 使用的机器学习分类模型
    :param flag: 为1表示用神经网络分类，为0表示用传统方法分类
    :return:
    '''
    active = True
    a = loadWeiboModel("../model/sgns.weibo.bigram-char", 1)
    while active:
        sentense = input('what is the sentence?')
        sentense = [sentense]
        data_vector = []
        for i in range(len(sentense)):
            data_vector.append(compute_vector_for_phrase(Sent2Word(sentense[i]), a, filename, True))
        if flag:
            data_vector = torch.Tensor(data_vector)
            if classifiername == 'CNN' :
                data_vector = data_vector.numpy()
                data_vector = np.array([data.reshape(3, 10, 10) for data in data_vector])
                data_vector = torch.from_numpy(data_vector)
            output = classifier(data_vector, is_training=False)
        else:
            output = classifier.predict(data_vector)
        output=output.detach().numpy()
        if output[0][0]>=output[0][1]:
            print("Non-bully!")
        else:
            print("bully!")
        repeat = input('repeat?(y/n)')
        if repeat == 'n':
            active = False


if __name__ == '__main__':
    filename = '../data/weibo_comment.csv'
    modelname = '../model/sgns.weibo.bigram-char'
    embeddingtype = 'word2vec'
    classifiername = 'FCN'
    setup_seed(0)
    classifier, flag = usingNetwork(filename, modelname, embeddingtype, classifiername)
    #classifier, flag = usingTraditionalClassifier(filename, modelname, embeddingtype, classifiername)
    demo(classifier, filename, classifiername, flag)  # 以使用sgns.weibo.word模型为例
