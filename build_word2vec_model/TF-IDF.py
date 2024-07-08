#TF-IDF
import math
import numpy

def TFIDF(data,word_bank):
    file_len=len(data)
    wb_len=len(word_bank)

    #计算TF/IDF
    word0_freq=[0 for i in range(wb_len)]
    iword0_freq=[0 for i in range(wb_len)]
    tfidf0=[0 for i in range(wb_len)]
    word_freq=dict(zip(word_bank, word0_freq))
    iword_freq=dict(zip(word_bank, iword0_freq))
    tfidf=dict(zip(word_bank, tfidf0))
    for index in range(file_len):
        tmp=data[index]
        for w in tmp:
            if w in word_bank:
                word_freq[w]+=1
        tmp0=set(tmp)#去重
        tmp=list(tmp0)
        for w in tmp:
            if w in word_bank:
                iword_freq[w]+=1
    for index in range(wb_len):
        w=word_bank[index]
        tfidf[w]=100*(word_freq[w]/wb_len)*math.log(file_len/(iword_freq[w]+1))

    return tfidf
    #return word_freq,iword_freq