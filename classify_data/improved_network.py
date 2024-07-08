import sys
from keras.layers import LSTM, Masking, Dense, GRU, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from matplotlib import pyplot as plt
sys.path.append('D:\大三上\AU332\AI_PROJECT1')
from prepro_data.prepro import *
from sklearn.model_selection import train_test_split



def buildSimpleModel(input_dim, learning_rate, dc, output_dim):
    '''
    bully language分类网络，全连接网络
    :param input_dim: 输入维度
    :param learning_rate: 初始学习率
    :param dc：学习率递减率
    :param output_dim: 输出维度
    :return model: 一个FCN网络模型
    '''
    model = Sequential()
    model.add(
        Dense(input_dim, activation='selu', input_shape=(input_dim,), use_bias=False, kernel_initializer='he_normal'))
    model.add(Dense(200, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.add(Dense(100, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.add(Dropout(0.05))
    model.add(Dense(30, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.add(Dense(output_dim, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.compile(optimizer=Adam(lr=learning_rate, decay=dc), loss='mse', metrics=['accuracy'])
    model.summary()
    return model


def buildGRUModel(len1, len2, learning_rate, dc, output_dim=2):
    '''
    bully language分类网络，GRU网络
    :param len1: 时间步长度（句子最大长度）
    :param len2: 词向量长度
    :param learning_rate: 初始学习率
    :param dc：学习率递减率
    :param output_dim: 输出维度
    :return model: 一个GRU网络模型
    '''
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(len1, len2)))
    model.add(
        GRU(output_dim * 30, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='he_normal',
            return_sequences=False))
    model.add(Dropout(0.01))
    model.add(Dense(output_dim * 10, activation='tanh', use_bias=True))
    model.add(Dense(output_dim, activation='tanh', use_bias=True))
    model.compile(optimizer=Adam(lr=learning_rate, decay=dc), loss='mse', metrics=['accuracy'])
    model.summary()
    return model


def buildLSTMModel(len1, len2, learning_rate, dc, output_dim=2):
    '''
    bully language分类网络，LSTM网络
    :param len1: 时间步长度（句子最大长度）
    :param len2: 词向量长度
    :param learning_rate: 初始学习率
    :param dc：学习率递减率
    :param output_dim: 输出维度
    :return model: 一个LSTM网络模型
    '''
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(len1, len2)))
    model.add(
        LSTM(output_dim * 30, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='he_normal',
             return_sequences=False))
    model.add(Dense(output_dim * 15, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.add(Dense(output_dim, activation='tanh', use_bias=True, kernel_initializer='he_normal'))
    model.compile(optimizer=Adam(lr=learning_rate, decay=dc), loss='mse', metrics=['accuracy'])
    model.summary()
    return model


def modelTrain(model, train_data, train_label, test_data, test_label, ep, bs, model_name):
    '''
    模型训练子函数
    :param model: 需要训练的模型
    :param train_data: 训练数据
    :param train_label: 训练标签
    :param test_data: 测试数据
    :param test_label: 测试样本
    :param ep: 训练轮数
    :param bs：单次训练样本量batch_size
    :param model_name: 模型名称
    :return 无
    '''
    history = model.fit(train_data, train_label, epochs=ep, batch_size=bs, verbose=1,
                        validation_data=(test_data, test_label))
    dense1_output = model.predict(train_data)
    for i in range(10):
        print(dense1_output[i],train_label[i])
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ls, ac = model.evaluate(test_data, test_label)
    print("Accuracy=", ac)

    # plot of Accuracy and Loss
    plt.clf()
    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='accuracy of the train set')
    plt.plot(epochs, loss, 'blue', label='loss of the train set')
    plt.plot(epochs, val_acc, 'burlywood', label='accuracy of the test set')
    plt.plot(epochs, val_loss, 'green', label='loss of the test set')
    plt.legend()
    name0 = "../result/" + model_name + ".jpg"
    plt.savefig(name0)


def getDataWithOneShotLabel(filename):
    '''
    对数据样本标签进行单热向量转换
    :param filename: 需要操作的文件名
    :return: comment: 整理好的评论数据
    :return: label: 整理好的标签
    '''
    comment, label = [], []
    with open(filename, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.reader(f)
        for line in reader:
            comment.append(line[1])
            if int(line[0]) == 0:
                label.append([0,1])
            elif int(line[0]) == 1:
                label.append([1,0])
    entry = list(zip(comment, label))
    np.random.shuffle(entry)
    comment[:], label[:] = zip(*entry)
    for i in range(len(comment)):
        comment[i] = Sent2Word(Traditional2Simplified(comment[i]))

    return comment, label


def getDataWithNegativeSampling(filename):
    '''
    对不平衡的样本进行补偿（默认非Bully样本更多，补偿Bully样本）
    :param filename: 需要平衡样本的文件名
    :return: comment: 整理好的评论数据
    :return: label: 整理好的标签
    '''
    comment, label = getDataWithOneShotLabel(filename)
    num_pos = 0
    num_neg = 0
    for index in range(len(label)):
        if label[index] == [0,1]:
            num_pos += 1  # 非bully
        elif label[index] == [1,0]:
            num_neg += 1  # bully
    for index in range(num_pos - num_neg):
        fg = True
        while fg:
            no = np.random.randint(0, num_pos + num_neg)
            if label[no] == [1,0]:
                fg = False
                a = np.random.randint(0, len(comment))
                comment.insert(a, comment[no])
                label.insert(a, [1,0])

    return comment, label


def computeVectorForPhrase(data0, w_dict, mode, save_it=False):
    '''
    对一组分词后的中文句子（lists of str），利用预训练W2V词典，输出一组它们对应的词向量
    :param data0: 一组分词后的中文句子（lists of str）
    :param w_dict: 用于转换词向量的预训练W2V词典
    :param mode: 转换模式，对应使用的是普通的定长模型，变长模型(限制最长50词)
    :return: data: 一组data0对应的拼音编码（300维向量，lists of list），也会存一份.npy文件副本
    '''
    key = w_dict.keys()
    data = []
    print("Total number of data0:", len(data0))
    a = len(data0)
    empty_null = [0 for abc in range(300)]
    if mode == "simple":
        for index in range(len(data0)):
            print("0正在编码1，序号：", index, '/', a, sep='')
            for i in range(len(data0[index])):
                if data0[index][i] in key:
                    tmp = w_dict[data0[index][i]]
                else:
                    tmp = empty_null
                if i == 0:
                    all_code = pd.Series(tmp)
                else:
                    code = pd.Series(tmp)
                    all_code += code
            all_code /= max(len(data0[index]), 1)
            data.append(all_code.tolist())
    elif mode == 'variable_length':
        for index in range(len(data0)):
            print("0正在编码2，序号：", index, '/', a, sep='')
            all_code = []
            for i in range(min(len(data0[index]), 50)):
                if data0[index][i] in key:
                    tmp = w_dict[data0[index][i]]
                else:
                    tmp = empty_null
                all_code.append(tmp)
            empty = [-1 for y in range(300)]
            for i in range(max(0, 50 - len(data0[index]))):
                all_code.append(empty)
            data.append(all_code)

    data = np.array(data)
    if save_it:
        np.save(file="../data/pinyin_data_vector.npy", arr=data)

    return data


def getDataVector(comment, label, filename, modelname, embedding_type, mode):
    global data_vector
    if embedding_type == 'pinyin2vec':
        try:
            data_vector = np.load(file='../data/pinyin_data_vector.npy', allow_pickle=True)
        except FileNotFoundError:
            last_model = load_model(modelname)
            model = Model(inputs=last_model.input, outputs=last_model.get_layer('dense_2').output)
            model.summary()
            data_vector = pinyin_word2vec(comment, mode, model, True)
    elif embedding_type == 'word2vec':
        if modelname == '../model/word2vec':
            model = Word2Vec.load(modelname)
            vocabulary = processtfidf(filename)
            for i in range(len(comment)):
                tmp = compute_vector_for_phrase(comment[i], model, vocabulary, False)
                data_vector.append(tmp)
        else:
            model = loadWeiboModel(modelname, 30)
            data_vector = computeVectorForPhrase(comment, model, mode)
    data_vector = np.asarray(data_vector, dtype='float32')
    label = np.asarray(label)
    return data_vector, label


def confuse(model, data, label):
    '''
    绘制混淆矩阵
    :param model: 使用的深度学习模型
    :param data: 测试数据
    :param model: 测试标签
    :return: 无
    '''
    pred = np.argmax(model.predict(data),axis=1)
    c1 = [0, 0]  # Real-Bully（Bully/Non_bully）
    c0 = [0, 0]  # Real-Non_bully（Bully/Non_bully）
    for index in range(len(label)):
        if label[index][0] == 1:
            c1[pred[index]] += 1
        elif label[index][1] == 1:
            c0[pred[index]] += 1
    print("class1(Bully)", c1)
    print("class0(Non-bully)", c0)



if __name__ == "__main__":
    # parameter
    modelname = '../model/pinyin2vec.h5'
    embedding_type = 'pinyin2vec'
    classifiername = 'LSTM'  # Optional，和mode有对应关系
    filename = "../data/weibo_comment.csv"
    epoch0 = 5  # 最终训练轮数（仅限LSTM）
    comment, label = getDataWithNegativeSampling(filename)
    print(len(comment))
    print(len(label))
    if classifiername=='FCN':# 'simple'-普通定长，对应FCN//'variable_length'-变长，对应GRU、LSTM
        mode='simple'
    else:
        mode='variable_length'
    data_vector, label = getDataVector(comment, label, filename, modelname, embedding_type, mode)
    X_train, X_test, y_train, y_test = train_test_split(data_vector, label, train_size=0.8, test_size=0.2,
                                                        random_state=2, shuffle=True, stratify=label)

    pos=0#正样本量（测试样本）
    neg=0#负样本量（测试样本）
    for index in range(len(y_test)):
        if y_test[index][0]==1:
            pos+=1
        elif y_test[index][1]==1:
            neg+=1
    #print("测试样本：正样本量：",pos,"负样本量",neg,"测试样本总量：",len(y_test))
    # 深度学习分类模型
    if classifiername == 'FCN':
        model1 = buildSimpleModel(EMBEDDING_SIZE, 0.01, 0.2, 2)
        modelTrain(model1, X_train, y_train, X_test, y_test, 150, 100, classifiername)
    elif classifiername == 'GRU':
        model1 = buildGRUModel(50, EMBEDDING_SIZE, 0.03, 0.01, 2)  # 注意：Mask_value=-1
        modelTrain(model1, X_train, y_train, X_test, y_test, 20, 100, classifiername)
    elif classifiername == 'LSTM':
        model1 = buildLSTMModel(50, EMBEDDING_SIZE, 0.02, 0.018, 2)  # 注意：Mask_value=-1
        modelTrain(model1, X_train, y_train, X_test, y_test, epoch0, 80, classifiername)
    confuse(model1, X_test, y_test)
    model1.save("../model/" + classifiername + ".h5")  # 保存模型
