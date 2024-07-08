from prepro_data.prepro import getData,Sent2Word,Traditional2Simplified
from gensim.models import FastText
import time
import logging
from classify_data.config import *

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SentenceIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        comment = getData(filename, True)
        for i in range(len(comment)):
            comment[i] = Traditional2Simplified(comment[i])
            yield Sent2Word(comment[i])

def train_fasttext(filename, vec_dim=EMBEDDING_SIZE):
    '''
    训练fasttext词嵌入模型
    :param filename: 用于训练模型的语料库
    :param vec_dim: 输出的词向量维度
    :return:
    '''
    t1 = time.time()
    model = FastText(
        SentenceIterator(filename),
        workers=FASTTEXT_WORKERS,
        size=vec_dim,
        min_count=MIN_WORD_COUNT,
        window=FASTTEXT_CONTEXT,
    )
    model.init_sims(replace=True)  # 不需要继续训练可以用这个函数更高效
    model.save('../model/fasttext')
    print('-------------------------------------------')
    print("Training fasttext model cost %.3f seconds...\n" % (time.time() - t1))


if __name__ == '__main__':
    filename = '../data/weibo_comment.csv'
    train_fasttext(filename)
