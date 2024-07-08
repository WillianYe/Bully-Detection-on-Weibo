import itertools
from collections import Counter
from prepro_data.prepro import *
from build_word2vec_model.train_word2vec import get_embedding_weights
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers


def build_vocab(sentences):
    """
    建立词汇表
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    让输入的句子长度一样
    """
    sequence_length = 50
    padded_sentences = []
    for sentence in sentences:
        if len(sentence) <= 50:
            new_sentence = sentence + [padding_word] * (sequence_length - len(sentence))
        else:
            new_sentence = sentence[:50]
        padded_sentences.append(new_sentence)
    return padded_sentences


# 参数设置
filter_sizes = (3, 5)
num_filters = 3
hidden_dims = 50
batch_size = 64
num_epochs = 10
sequence_length = 50
filename = '../data/weibo_comment.csv'
modelname = '../model/sgns.weibo.word'

# 数据预处理
comment, label = getData(filename, False)
for i in range(len(comment)):
    comment[i] = Sent2Word(Traditional2Simplified(comment[i]))
comment = pad_sentences(comment)
label = np.array(label)
vocabulary, vocabulary_inv = build_vocab(comment)
comment = np.array([[vocabulary[word] for word in sentence] for sentence in comment], dtype='float32')
if modelname == '../model/word2vec':
    model = Word2Vec.load(modelname)
else:
    model = loadWeiboModel(modelname,1)
embedding_weights = get_embedding_weights(model, vocabulary)
X_train, X_test, y_train, y_test = train_test_split(comment, label, train_size=0.01, test_size=0.99,
                                                    random_state=2,
                                                    shuffle=True, stratify=label)

# cnn模型
input_shape = (sequence_length,)
input = Input(shape=input_shape)
output1 = Embedding(len(vocabulary_inv), EMBEDDING_SIZE, input_length=sequence_length, name="embedding")(input)
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1,
                         )(output1)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
output2 = Concatenate(name='conv')(conv_blocks)
output3 = BatchNormalization(name='nor')(output2)
output4 = Dense(hidden_dims, activation="relu", name='fc1')(output3)
output = Dense(2, activation="relu", name='fc2')(output4)
model = Model(input, output)
adam = optimizers.Adam(lr=0.01, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), verbose=2)
# 查看某一层的输出
# layer_model=Model(inputs=model.input,outputs=model.get_layer("fc2").output)
# print(layer_model.predict(X_train))
