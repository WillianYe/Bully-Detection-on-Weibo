# Bully-Detection-on-Weibo
微博网络暴力言论检测（SJTU AU332)

本项目通过自然语言处理技术，检测微博上的网络暴力言论。首先，利用[结巴](https://github.com/fxsjy/jieba)进行中文分词，由于中文和英文相比缺少空格这样的明显单词界限标记，需要切割成有意义的词组合；其次，构建四种词嵌入模型，包括Word2Vec with TF-IDF, Fasttext, Doc2Vec, Pinyin2Vec using CBOW model，将文本通过一个向量来表示，便于提取特征；最后，采用传统机器学习方法（svm，决策树，贝叶斯，随机森林，线性回归）和神经网络方法（FCN，CNN，LSTM）训练，构建分类器。综合比较多种方案，采用Pinyin2Vec+LSTM时达到了最优的86%正确率，具体见报告ai_project_report.pdf。

##### 文件内容
+ data文件夹包含用到的数据
  
weibo_comment.csv：初步整理的微博语料库

t_data.npy：程序生成的用于拼音编码模型的训练样本（上下文）

t_label.npy：程序生成的用于拼音编码模型的训练标签（中心词）

+ build_word2vec_model文件夹包含训练词嵌入模型的代码
  
TF-IDF.py：计算TF-IDF权值

train_fasttext.py：训练fastText模型

train_doc2vec.py：训练doc2vec模型

train_pinyin2vec.py：训练拼音编码模型

train_word2vec.py：训练word2vec模型
+ model文件夹包含训练后的词嵌入模型（由于模型较大，没有上传）

doc2vec和word2vec：用weibo_comment.csv训练出的doc2vec和word2vec模型

fasttext：用weibo_comment.csv训练的fasttext模型

pinyin2vec.h5：用weibo_comment.csv训练的拼音编码模型

sgns.weibo.bigram-char.txt：网上预训练的word2vec词向量（建议与FCN合用）

LSTM.h5：用weibo_comment.csv训练完的LSTM模型
+ prepro_data文件夹将数据变成分类器的输出

hit_stopwords.txt：停用词表，用于去除停用词

langconv.py和zh_wiki.py：用于繁体转简体

prepro.py：实现数据繁体转简体、分词、去除停用词，计算word2vec、拼音、doc2vec词向量，计算tfidf权重，划分数据集等等
+ classify_data文件夹使用机器学习分类器对数据分类
  
config.py：设置参数

network.py：建立神经网络模型

classify.py：对数据进行分类并演示结果

improved_network.py：可选择运用拼音编码模型（亦可选择网上W2V预训练模型），然后可选择在FCN\GRU\LSTM上进行训练

textcnn.py：使用textCNN模型进行分类

demo.py：运用训练完的分类模型，自行输入中文语句，判定网络暴力言论
+ get_data文件夹存放对微博网页版源代码抓取后进行处理的程序

微博数据转换.py对存入excel的源代码进行分析（一级、二级评论）
##### 运行环境
+ 运行demo.py和improved_network.py：
  
torch1.7.1/tensorflow2.0.0/scikit-learn0.23.2/scipy1.4.1/Keras2.3.1
+ 运行classify.py:

tensorflow2.3.0/scikit-learn0.23.2/scipy1.5.2/keras2.4.3
##### 运行参数
+ classify.py和improved_network.py中：

filename：要分类的数据的路径

modelname：词嵌入模型路径，可在model文件夹中选择，需与modelname对应

embeddingtype：词嵌入方法，需和词嵌入模型对应

classifiername：使用的机器学习分类器。可选择usingNetwork()函数或usingTraditionalClassifier()函数中的一种进行分类，分别对应神经网络和传统机器学习算法。
+ demo.py：

w_mode：使用哪种模型，0-拼音模型，1-他人预训练模型

len_mode: 'simple'-普通定长，对应FCN/'variable_length'-变长，对应GRU、LSTM

filename = 'model0.h5' :需要使用的模型文件名
+ 微博数据转换.py里根据需要选择一级评论（默认）还是二级评论
+ prepro.py中的compute_word2vec_for_phrase函数选择是否使用tfidf加权
