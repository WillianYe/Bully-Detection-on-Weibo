# encoding=utf-8
import numpy as np
import pandas as pd

def check_chinese(str0):
    '''
    检查字符是否是中文（但保留分隔的字符）
    :param str: 需要检查的字符
    :return FLAG: True or False
    '''
    punc={'，','。','：','；','、',' ','/','“','”','‘','’','.','（','）','【','】','\n','！','？','…','.','?','!'}
    if u'\u4e00' <= str0 <= u'\u9fff':
        return True
    if str0 in punc:
        return True
    return False

def data_cleaning(data,data_index=0):
    '''
    数据清洗，清除无效和重复的评论，输入的应为pd.dataframe
    :param data: 待清洗的数据，pd.dataframe形式，每一行为一条数据
    :param data_index: 文字评论所在列索引
    :return data: 清洗完的数据，还是pd.dataframe
    '''
    print("原始数据量：",data.shape[0])
    #过滤无效数据
    A=["图片评论","转发微博"]
    tmp=data.shape[0]
    for index in range(tmp):
        if data.iloc[tmp-index-1,data_index][0:4] in A:
            data.drop([tmp-index-1],inplace=True)
    print("清洗后的数据量：",data.shape[0])
    data=data.reset_index(drop=True)
    return data

#一级评论

filename='weibo.xlsx' #根据需要请修改此处的文件名！包括下面是否存在表头
weibo_data=pd.read_excel(filename,header=None)
cnt=0
for index in range(len(weibo_data)):
    if weibo_data.iloc[index-cnt,0][14] is not '<':
        weibo_data.drop(index=[index],inplace=True)
        cnt+=1

A="d52="
len0=len(weibo_data)
wdata=[]
for index in range(len0):#len(weibo_data)):
    fg=False
    for index0 in range(1000,len(weibo_data.iloc[index,0])-7):
        if weibo_data.iloc[index,0][index0:index0+4] == A :
            #print(index0,weibo_data.iloc[index,0][index0+8:index0+10])
            for index1 in range(len(weibo_data.iloc[index,0])-8-index0):
                str0=weibo_data.iloc[index,0][index0+7+index1]
                str1=weibo_data.iloc[index,0][index0+8+index1]
                if not check_chinese(str0) and not check_chinese(str0) :
                    if index1 > 2:
                        wdata.append(weibo_data.iloc[index,0][index0+7:index0+7+index1])
                        #print(weibo_data.iloc[index,0][index0+7:index0+7+index1])
                        fg=True
                    break
        if fg :
            break

df=pd.DataFrame(wdata, columns=['微博评论'])
#df=data_cleaning(df,1)
df.to_excel('weibo-new.xlsx')

'''
#二级评论

filename='weibo_second.xlsx'
weibo_data=pd.read_excel(filename,header=None)
cnt=0
for index in range(len(weibo_data)):
    if weibo_data.iloc[index-cnt,0][18] is not 'd':
        weibo_data.drop(index=[index],inplace=True)
        cnt+=1

A="<h3>"
len0=len(weibo_data)
wdata=[]
B="回复"
C="</a>:"
D=["<s",""]
for index in range(len0):#len(weibo_data)):
    fg=False
    for index0 in range(500,len(weibo_data.iloc[index,0])-4):
        if weibo_data.iloc[index,0][index0:index0+4] == A :
            bias=0
            str01=weibo_data.iloc[index,0][index0+4:index0+6]
            if str01 == B:
                for index1 in range(len(weibo_data.iloc[index,0])-11-index0):
                    str_n=weibo_data.iloc[index,0][index0+6+index1:index0+11+index1]
                    if str_n == C:
                        bias=7+index1
                        break

            for index1 in range(bias,len(weibo_data.iloc[index,0])-5-index0):
                str0=weibo_data.iloc[index,0][index0+4+index1]
                str1=weibo_data.iloc[index,0][index0+5+index1]
                if not check_chinese(str0) and not check_chinese(str0) :
                    if index1 > 1:
                        if not (weibo_data.iloc[index,0][index0+4+bias:index0+4+index1] in D):
                            wdata.append(weibo_data.iloc[index,0][index0+4+bias:index0+4+index1])
                            print(weibo_data.iloc[index,0][index0+4+bias:index0+4+index1])
                            fg=True
                    break
        if fg :
            break

df=pd.DataFrame(wdata, columns=['微博二级评论'])
df.to_excel('weibo__second_new.xlsx')

#print(wdata)
'''