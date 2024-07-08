import logging
import random
from gensim.models import doc2vec
from prepro_data.prepro import *
from classify_data.config import *

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class doc2VecModel():

    def __init__(self):
        super().__init__()

    def initialize_model(self,vec_dim=EMBEDDING_SIZE):
        '''
        初始化doc2vec模型
        :param vec_dim: 输出的词向量维度
        :return:
        '''
        logging.info("Building Doc2Vec vocabulary")
        self.model = doc2vec.Doc2Vec(min_count=MIN_WORD_COUNT,
                                     window=DOC2VEC_CONTEXT,
                                     vector_size=vec_dim,
                                     workers=DOC2VEC_WORKERS,
                                     alpha=0.025,  # The initial learning rate
                                     min_alpha=0.00025, # Learning rate will linearly drop
                                     # to min_alpha as training progresses
                                     dm=1)
                                     # dm defines the training algorithm.
                                     #  If dm=1 means 'distributed memory' (PV-DM)
                                     # and dm =0 means 'distributed bag of words' (PV-DBOW)
        self.model.build_vocab(self.corpus)
        self.vector_size=300

    def train_model(self):
        '''
        训练doc2vec模型并保存
        :return:
        '''
        logging.info("Training Doc2Vec model")
        for epoch in range(10):
            logging.info('Training iteration #{0}'.format(epoch))
            self.model.train(
                self.corpus, total_examples=self.model.corpus_count,
                epochs=self.model.epochs)
            # shuffle the corpus
            random.shuffle(self.corpus)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha
        self.model.save('../model/doc2vec')

    def build_corpora(self,filename):
        '''
        处理数据集，建立语料库
        :param filename: 使用的语料库路径
        :return:
        '''
        corpus = []
        comment=getData(filename, True)
        for i, item in enumerate(comment):
            word_list=[]
            item = Sent2Word(Traditional2Simplified(item))
            for j in item:
                word_list.append(j)
            corpus.append(doc2vec.TaggedDocument(words=word_list, tags=[i]))
        self.corpus=corpus


def train_doc2vec(filename):
    d2v = doc2VecModel()
    d2v.build_corpora(filename)
    d2v.initialize_model()
    d2v.train_model()


if __name__ == '__main__':
    filename = '../data/weibo_comment.csv'
    train_doc2vec(filename)
