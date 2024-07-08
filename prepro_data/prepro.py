from prepro_data.langconv import *
import csv
import numpy as np
import jieba
import torch
from sklearn.model_selection import train_test_split
from classify_data.config import EMBEDDING_SIZE
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec, Word2Vec, FastText
from tensorflow.keras.models import load_model, Model
from pypinyin import pinyin, Style
import pandas as pd
import re


def Traditional2Simplified(sentence):
    '''
    繁体转简体
    :param sentence: 一个中文字符串
    :return: sentence: 转为简体后的中文字符串
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def Sent2Word(sentence):
    '''
    分词并去除停用词
    :param sentence: 一个中文字符串
    :return: words: 一个列表，包含多个由句子分词后得到的字符串
    '''
    stop_words = [w.strip() for w in open('../prepro_data/hit_stopwords.txt', 'r', encoding='UTF-8').readlines()]
    stop_words.append(' ')
    words = jieba.cut(sentence)
    words = [w for w in words if w not in stop_words]
    return words


def compute_vector_for_phrase(phrase, model, vocabulary, flag):
    '''
    将一个词语转变为词向量
    :param phrase: 一个中文词语字符串
    :param model: word2vec模型
    :param vocabulary: 中文词汇表，用于tfidf计算词频
    :param flag: 为1表示使用sgns.weibo.word模型，为0时表示使用word2vec模型
    :return: 词语对应的词向量，维度是EMBEDDING_SIZE
    '''
    result = np.zeros(EMBEDDING_SIZE, dtype='float32')
    for word in phrase:
        if flag:
            if word in model.keys():  # and word in vocabulary.keys():
                # result += model[word] * vocabulary[word]    # 使用tfidf加权
                result += model[word]  # 不使用tfidf加权
        else:
            if word in model.wv and word in vocabulary.keys():
                result += model[word] * vocabulary[word]
    return result


def sheng_yun_mu(st):
    '''
    对一个中文单字的拼音字符串，输出其对应的拼音编码（非单热向量）
    :param st: 一个中文单字的拼音字符串
    :return: code_1: 一个整型，对应声母的编号（一维，未转换）
    :return: code_2: 一个整型，对应韵母或（介母+韵母）的编号（一维，未转换）
    '''
    length = len(st)
    shengmu_num = 1
    code_1 = 0
    code_2 = 0
    # 拼音各部分对应的词典
    dict1 = {'z': 1, 'c': 5, 's': 9}  # Xh
    dict2 = {'ang': 5, 'ing': 10, 'ian': 12, 'iao': 14, 'eng': 20, \
             'ong': 23, 'uai': 28, 'uan': 29, 'van': 34}  # XXX
    dict3 = {'a': 1, 'i': 6, 'e': 16, 'o': 21, 'u': 24, 'v': 33}  # X
    dict4 = {'ai': 2, 'ao': 3, 'an': 4, 'ie': 7, 'iu': 8, 'in': 9, \
             'ia': 11, 'ei': 17, 'er': 18, 'en': 19, 'ou': 22, 'ua': 25, \
             'ui': 26, 'un': 27, 'uo': 31, 'ue': 32, 've': 35, 'vn': 36}  # XY
    dict5 = {'iang': 13, 'iong': 15, 'uang': 30}  # XXXX
    # 声母部分
    if st[0] in ['z', 'c', 's', 'a', 'e', 'i', 'o', 'u', 'v']:
        if st[0] in ['a', 'e', 'i', 'o', 'u', 'v']:
            # 无声母
            shengmu_num = 0
        # 两位声母
        elif st[1] == 'h':
            shengmu_num = 2
            code_1 = dict1[st[0]]
        else:
            code_1 = ord(st[0]) - 96

    else:  # 一位声母
        code_1 = ord(st[0]) - 96

    # 韵母部分
    length -= shengmu_num
    if length == 3:
        code_2 = dict2[st[shengmu_num:(shengmu_num + 3)]]
    elif length == 1:
        code_2 = dict3[st[shengmu_num]]
    elif length == 4:
        code_2 = dict5[st[shengmu_num:(shengmu_num + 4)]]
    elif length == 2:
        code_2 = dict4[st[shengmu_num:(shengmu_num + 2)]]
    else:
        code_2 = 0

    return code_1, code_2


def pinyin_code_single(word):
    '''
    对一个中文单词（字符串），输出其对应的拼音编码（252维向量，pd.Series形式）
    :param word: 一个中文单词字符串
    :return: all_code: 一个pd.Series形式的252维原始词向量
    '''
    # remain_punc={'！','？','…'}
    global all_code
    for i in range(1):
        all_code = pd.Series([0 for abc in range(252)])  # 4*(23+36+4)

        py = pinyin(word, style=Style.TONE3, strict=False, neutral_tone_with_five=True, errors='ignore')
        num = len(py)
        for x in range(min(num, 4)):  # 限制最多编码四字词语，过长部分舍弃
            if (ord(py[x][0][0]) < 123) and (ord(py[x][0][0]) > 96):
                tmp = int(re.findall(r'\d+', py[x][0])[0])  # 编码音调
                if tmp == 1:
                    all_code[15 + 63 * x] = 1
                elif tmp == 2:
                    all_code[21 + 63 * x] = 1
                elif tmp == 3:
                    all_code[22 + 63 * x] = 1
                elif tmp == 4:
                    all_code[0 + 63 * x] = 1

                [tmp1, tmp2] = sheng_yun_mu(py[x][0][:-1])
                all_code[63 * x + tmp1] = 1  # 编码声母
                all_code[63 * x + tmp2 + 26] = 1  # 编码韵母
    return all_code


def pinyin_word2vec(data0, mode, model, saveit=False):
    '''
    对一组分词后的中文句子（lists of str），利用拼音W2V模型，输出一组data0对应的词向量（300维向量，arrays of list）
    :param data0: 一组分词后的中文句子（lists of str）
    :param mode: 转换模式，对应使用的是普通的定长模型，变长模型(限制最长50词)
    :param model: 用于转换词向量的预训练W2V深度学习模型
    :return: data: 一组data0对应的词向量（300维向量，arrays of list），也会存一份.npy文件副本
    '''
    data = []
    print("Total number of data0:", len(data0))
    a = len(data0)
    if mode == "simple":
        for index in range(len(data0)):
            print("正在编码1，序号：", index, '/', a, sep='')
            for i in range(len(data0[index])):
                tmp = pinyin_code_single(data0[index][i])
                tp = np.array(tmp)
                tp = tp.reshape(1, 252)
                if i == 0:
                    code = model.predict(tp)
                    code = list(code)[0]
                    all_code = pd.Series(code)
                else:
                    code = model.predict(tp)
                    code = list(code)[0]
                    code = pd.Series(code)
                    all_code += code
            all_code *= 50  # 加速训练过程
            all_code /= max(len(data0[index]), 1)
            data.append(all_code.tolist())
        data = np.array(data)
        if saveit:
            np.save(file="../data/pinyin_data_vector_simple.npy", arr=data)
    elif mode == 'variable_length':
        for index in range(len(data0)):
            print("正在编码2，序号：", index, '/', a, sep='')
            all_code = []
            for i in range(min(len(data0[index]), 50)):
                tmp = pinyin_code_single(data0[index][i])
                tp = np.array(tmp)
                tp = tp.reshape(1, 252)
                code = model.predict(tp)
                code = list(code)[0]
                for x in range(len(code)):
                    code[x] *= 30
                all_code.append(code)
            empty = [-1 for y in range(300)]
            for i in range(max(0, 50 - len(data0[index]))):
                all_code.append(empty)
            data.append(all_code)
        data = np.array(data)
        if saveit:
            np.save(file="../data/pinyin_data_vector.npy", arr=data)
    return data


def is_all_chinese(str):
    '''
    判断一个句子是否全是中文
    :param str: 一个中文句子字符串
    :return: 判断结果，True表示全是中文
    '''
    for char in str:
        if not '\u4e00' <= char <= '\u9fa5':
            return False
    return True


def loadWeiboModel(modelname, k):
    '''
    用于加载sgns.weibo.word模型
    :return:
    '''
    stop_words = [w.strip() for w in open('../prepro_data/hit_stopwords.txt',
                                          'r', encoding='UTF-8').readlines()]
    stop_words.append(' ')
    with open(modelname, 'r', encoding='utf-8') as f:
        model = {}
        f.readline()
        for line in f:
            values = line.split()
            word = values[0]
            if word not in stop_words and is_all_chinese(word):
                model[word] = k * np.asarray(values[1:], dtype='float32')
    return model


def identity_tokenizer(text):
    '''
    用于设置tfidf分词器
    '''
    return text


def processtfidf(filename):
    '''
    用于输出tfidf得到的权值
    :param filename: 数据集路径
    :return: 一个字典，键为某个中文词语，值为其tfidf权值
    '''
    comment, _ = getData(filename, False)
    for i in range(len(comment)):
        comment[i] = Sent2Word(Traditional2Simplified(comment[i]))
    vector = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    vector.fit_transform(comment)
    return vector.vocabulary_


def getData(filename, flag):
    '''
    采集并打乱数据
    :param filename: 数据集路径
    :param flag: 为1表示收集语料库数据（只有评论），为0表示收集分类数据（评论加评分）
    :return:
    comment: 商品评论，一个列表，元素为中文句子字符串
    label: 评论对应的评分值（0、1、2、3、4）
    '''
    comment, label = [], []
    with open(filename, 'r', encoding='gbk', errors='ignore') as f:
        reader = csv.reader(f)
        for line in reader:
            if flag:
                comment.append(line)
            else:
                comment.append(line[1])
                label.append(int(line[0]))
    if flag:
        return comment
    entry = list(zip(comment, label))
    np.random.shuffle(entry)
    comment[:], label[:] = zip(*entry)

    return comment, label


def get_data_vector(comment,label,filename,modelname, embedding_type):
    global data_vector
    data_vector = []
    if embedding_type == 'doc2vec':
        model = Doc2Vec.load(modelname)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        for i in range(len(comment)):
            tmp = model.infer_vector(comment[i])
            data_vector.append(tmp)
    elif embedding_type == 'fasttext':
        model = FastText.load(modelname)
        vocabulary = processtfidf(filename)
        print(len(model.wv.vocab))
        for i in range(len(comment)):
            tmp = compute_vector_for_phrase(comment[i], model, vocabulary,False)
            data_vector.append(tmp)
    elif embedding_type == 'pinyin2vec':
        try:
            data_vector = np.load(file='../data/pinyin_data_vector_simple.npy', allow_pickle=True)
        except FileNotFoundError:
            last_model = load_model(modelname)
            model = Model(inputs=last_model.input, outputs=last_model.get_layer('dense_2').output)
            model.summary()
            data_vector = pinyin_word2vec(comment, "simple", model, True)
    elif embedding_type == 'word2vec':
        if modelname == '../model/word2vec':
            model = Word2Vec.load(modelname)
            flag = False
        else:
            model = loadWeiboModel(modelname,30)
            flag = True
        vocabulary = processtfidf(filename)
        for i in range(len(comment)):
            tmp = compute_vector_for_phrase(comment[i], model, vocabulary, flag)
            data_vector.append(tmp)
    return data_vector,label


def processData(filename, modelname, embedding_type):
    '''
    初步处理数据，将数据转变为词向量，并划分数据集
    :param filename: 数据集路径
    :param modelname: 使用的词嵌入模型路径
    :param embedding_type: 词嵌入方法
    :return:
    X_train: 训练样本评论
    X_test: 测试样本评论
    y_train: 训练样本标签
    y_test: 测试样本标签
    '''
    comment, label = getData(filename, False)
    for i in range(len(comment)):
        comment[i] = Sent2Word(Traditional2Simplified(comment[i]))
    data_vector,label=get_data_vector(comment,label,filename,modelname, embedding_type)
    X_train, X_test, y_train, y_test = train_test_split(data_vector, label, train_size=0.8, test_size=0.2,
                                                        random_state=2,
                                                        shuffle=True, stratify=label)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_test = torch.Tensor(y_test).long()

    return X_train, X_test, y_train, y_test
