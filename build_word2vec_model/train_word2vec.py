from prepro_data.prepro import getData, Sent2Word, Traditional2Simplified
from gensim.models import Word2Vec
import time
import logging
from classify_data.config import *
import numpy as np

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


def train_word2vec(filename, vec_dim=EMBEDDING_SIZE):
    '''
    训练word2vec词嵌入模型
    :param filename: 用于训练模型的语料库
    :param vec_dim: 输出的词向量维度
    :return:
    '''
    t1 = time.time()
    model = Word2Vec(
        SentenceIterator(filename),
        workers=WORD2VEC_WORKERS,
        size=vec_dim,
        min_count=MIN_WORD_COUNT,
        window=WORD2VEC_CONTEXT,
    )
    model.init_sims(replace=True)  # 不需要继续训练可以用这个函数更高效
    model.save('../model/word2vec')
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

def get_embedding_weights(model,vocabulary):
    global embedding_weights
    if vocabulary!=None:
        embedding_weights = {key: model[word] if word in model else
                              np.random.uniform(-0.25, 0.25, EMBEDDING_SIZE)
                         for key, word in vocabulary.items()}
    return embedding_weights


if __name__ == '__main__':
    filename = '../data/weibo_comment.csv'
    train_word2vec(filename)
