from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import jieba as jb
from prepro_data.prepro import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from classify_data.config import *


def build_w2v_model(input_dim,dim):
    '''
    对以拼音编码形式存储的语料信息，用连续词袋CBOW模型进行训练，期望获得词向量编码网络
    :param input_dim: 输入向量维度
    :param dim: 期望词向量维度
    :return model: 一个神经网络模型，其中Dense2层的输出应为我们想要的词向量
    '''
    model = Sequential()
    model.add(Dense(int(0.5*dim+0.5*input_dim),activation='tanh', input_shape=(input_dim,),use_bias= True))
    model.add(Dense(dim,activation='tanh',use_bias= True))
    model.add(Dense(int(0.5*dim+0.5*input_dim),activation='tanh', input_shape=(input_dim,),use_bias= True))
    model.add(Dense(input_dim, activation='tanh', use_bias= True))
    model.compile(optimizer=Adam(lr=0.012,decay=0.2),loss='mse',metrics=['accuracy'])
    model.summary()
    return model

def wmodel_pre(data,windows,div_punc,stop_word):
    '''
    用于训练Word2Vec模型前的数据预处理工作
    :param data: 用于训练的数据，数据的每一行应为一个字符串（一句话）
    :param windows: CBOW模型的窗大小（单边大小，即上/下文需要考虑的词语数量）
    :param div_punc: 用于分句的标点符号集
    :param stop_word: 停用词
    :return： 无，将直接生成文件（存储可以被上面函数所处理的训练集和对应的期望输出（.npy））
    '''
    #分词分句
    data0=[]
    for i in range(len(data)):
        seg=jb.lcut(str(data[i]))#分词

        cn_punc=stop_word-div_punc
        cnt=0
        for index in range(len(seg)):
            if seg[index-cnt] in cn_punc:#去除停用词
                del seg[index-cnt]
                cnt+=1

        tmp=[]
        for index in range(len(seg)):#用标点符号分句
            if (seg[index] not in div_punc) and ( index<(len(seg)-1) ):
                tmp.append(seg[index])
            elif tmp:
                if (index==len(seg)-1) and (seg[index] not in div_punc):
                    tmp.append(seg[index])
                data0.append(tmp)
                tmp=[]

        #print("正在分句，序号：",i)
    #len_disturibution_plot(data0,30)
    a=len(data0)

    #构建训练集
    train_data=[]
    train_label=[]
    less_cnt=0#不足长度的样本量
    for index in range(len(data0)):
        print("正在编码，序号：",index,"/",a,sep='')
        if len(data0[index])<2*windows+1:#去除不足长度的样本
            less_cnt+=1
            continue
        else:
            for i in range(len(data0[index])-2*windows):
                tp=pinyin_code_single(data0[index][i+windows])#单句拼音编码
                train_label.append(tp)
                for j in range(windows):
                    tmp0=pinyin_code_single(data0[index][i+j])
                    tmp0*=max(0.6,1-0.2*(windows-1-j))#位置加权
                    if j==0:
                        tmp=tmp0
                    else:
                        tmp=tmp+tmp0
                for j in range(windows):
                    tmp0=pinyin_code_single(data0[index][i+windows+1+j])
                    tmp0*=max(0.6,1-0.2*j)#位置加权
                    tmp=tmp+tmp0
                train_data.append(tmp)

    train_data=np.array(train_data)
    train_label=np.array(train_label)
    print(less_cnt,train_label.shape,train_data.shape)
    np.save(file="../data/t_label.npy", arr=train_label)
    np.save(file="../data/t_data.npy", arr=train_data)

def wmodel_train(model,epoch,label_file,data_file):
    '''
    用于训练已搭建好的Word2Vec模型
    :param model: 需要训练的W2V模型
    :param epoch: 训练轮数
    :param label_file: 标签文件名（中心单词/.npy）
    :param data_file: 数据文件名（上下文/.npy）
    :return: 无,将直接保存模型文件.h5
    '''
    print("Begin W2V Training")
    train_label=np.load(file=label_file)
    train_data=np.load(file=data_file)

    #训练
    history=model.fit(train_data,train_label,epochs=epoch,batch_size=int(train_label.shape[0]/60),verbose=1)

    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    #plot of Accuracy and Loss
    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='accuracy of the train set')
    plt.plot(epochs, loss, 'blue', label='loss of the train set')
    plt.legend()
    name0='../result/w2v_fig.jpg'
    plt.savefig(name0)

    model.save("../model/pinyin2vec.h5")


if __name__ == "__main__":
    filename = "../data/weibo_comment.csv"
    excel_data = pd.read_csv(filename, encoding='gb18030')
    d_p={'，','。','；','\n','！','？','…'}#分句用的标点符号
    with open('../prepro_data/hit_stopwords.txt', 'r', encoding='utf-8') as rfile:#停用词
        stop_word = set(rfile.read().split("\n"))
    stop_word.add(' ')
    stop_word.add('.')
    w_data=excel_data.iloc[:,1]
    wmodel=build_w2v_model(252,EMBEDDING_SIZE)
    wmodel_pre(w_data,2,d_p,stop_word)
    wmodel_train(wmodel,100,"../data/t_label.npy","../data/t_data.npy")

