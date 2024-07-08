from ai_project import*
import keras
import numpy

txt = input("请输入语句:")

if True:
    #parameter
    w_mode = 0 #Optional使用哪种模型，0-拼音模型，1-他人预训练模型
    len_mode = 'variable_length' #'simple'-普通定长，对应FCN//'variable_length'-变长，对应GRU、LSTM
    filename = 'model0.h5' #需要使用的模型文件名

    with open('hit_stopwords.txt', 'r', encoding='utf-8') as rfile:#停用词
        stop_word = set(rfile.read().split("\n"))
    stop_word.add(' ')
    stop_word.add('.')
    cn_punc=stop_word
    
    #词向量嵌入
    if w_mode==1:
        #使用他人预训练的W2V模型
        w_dict=load_o_pre_w2v() 
    elif w_mode==0:
        #训练Word2Vec
        word_model=load_pretrain_wmodel("w2v.h5")
    
    seg=jb.lcut(txt)#分词
    
    cnt=0
    for index in range(len(seg)):
        if seg[index-cnt] in cn_punc:#去除停用词
            del seg[index-cnt]
            cnt+=1
    
    data0=[]
    data0.append(seg)
        
    #编码模式选择
    if w_mode==1:
        train_code=word2vec0(data0,w_dict,len_mode)
    elif w_mode==0:
        train_code=pinyin_word2vec(data0,len_mode,word_model)
    
    train_code=np.array(train_code)
    
    #深度学习分类模型
    model = load_model(filename)
    #model.summary()
    pred=model.predict_classes(train_code)
    result=pred[0]   
    if result==0:
        print(txt,":Bully!")
    else:
        print(txt,":Non-bully.")
    
