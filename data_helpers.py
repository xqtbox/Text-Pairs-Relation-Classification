import numpy as np
import gensim
import randolph
import multiprocessing
from collections import Counter
from gensim import corpora, models, similarities
from gensim.models import word2vec
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from tflearn.data_utils import to_categorical, pad_sequences

BASE_DIR = randolph.cur_file_dir()
TEXT_DIR = BASE_DIR + '/content.txt'
VOCABULARY_DICT_DIR = BASE_DIR + '/math.dict'
WORD2VEC_DIR = BASE_DIR + '/math.model'

def create_vocab(text_file):
    texts = []
    with open(text_file, 'r') as fin:
        for eachline in fin:
            line = eachline.strip().split(' ')
            texts.append(line)
    vocab_dict = corpora.Dictionary(texts)
    vocab_dict.save(VOCABULARY_DICT_DIR)
    return vocab_dict

def word2vec_train(embedding_size, inputFile=TEXT_DIR, outputFile=WORD2VEC_DIR):
    sentences = word2vec.LineSentence(inputFile)

    # sg=0 -> CBOW model; sg=1 -> skip-gram model.
    # 生成 embedding_size 的词向量 model
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(outputFile)

def data_word2vec(inputFile, dictionary):
    def token_to_index(content, dictionary):
        list = []
        for item in content:
            if item != '<end>':
                list.append(dictionary.token2id[item])
        return list

    with open(inputFile) as fin:
        labels = []
        front_content_indexlist = []
        behind_content_indexlist = []
        for index, eachline in enumerate(fin):
            front_content = []
            behind_content = []
            line = eachline.strip().split('\t')
            label = line[2]
            content = line[3].strip().split(' ')

            end_tag = False
            for item in content:
                if item == '<end>':
                    end_tag = True
                if end_tag == False:
                    front_content.append(item)
                if end_tag == True:
                    behind_content.append(item)

            labels.append(label)

            front_content_indexlist.append(token_to_index(front_content, dictionary))
            behind_content_indexlist.append(token_to_index(behind_content[1:], dictionary))
        total_line = index + 1

    class Data:
        def __init__(self, total_line, labels, front_content_indexlist, behind_content_indexlist):
            self.number = total_line
            self.labels = labels
            self.front_tokenindex = front_content_indexlist
            self.behind_tokenindex = behind_content_indexlist

    return Data(total_line, labels, front_content_indexlist, behind_content_indexlist)


def load_word2vec_matrix(vocab_size, embedding_size):
    model = gensim.models.Word2Vec.load(WORD2VEC_DIR)
    vocab_dict = Dictionary.load(VOCABULARY_DICT_DIR)

    Vector = np.zeros([vocab_size, embedding_size])
    for value, key in vocab_dict.items():
        if len(key) > 0:
            Vector[value] = model[key]

    return Vector

def max_seq_len_cal(content_indexlist):
    result = 0
    for item in content_indexlist:
        if len(item) > result:
            result = len(item)
    return result


def load_data_and_labels(data_file, MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE):
    """
    Loads research data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load vocabulary dict file
    vocab_dict = create_vocab(TEXT_DIR)
    #Load word2vec file
    Vector = word2vec_train(EMBEDDING_SIZE, TEXT_DIR, WORD2VEC_DIR)

    # Load data from files and split by words
    data = data_word2vec(inputFile=data_file, dictionary=vocab_dict)
    max_seq_len = max(max_seq_len_cal(data.front_tokenindex), max_seq_len_cal(data.behind_tokenindex))
    print('Found %s texts.' % data.number)
    print('Max sequence length is:', max_seq_len)
    data_front = pad_sequences(data.front_tokenindex, maxlen=MAX_SEQUENCE_LENGTH, value=0.)
    data_behind = pad_sequences(data.behind_tokenindex, maxlen=MAX_SEQUENCE_LENGTH, value=0.)
    labels = to_categorical(data.labels, nb_classes=2)
    print('Shape of data front tensor:', data_front.shape)
    print('Shape of data behind tensor:', data_behind.shape)
    print('Shape of label tensor:', labels.shape)
    return data_front, data_behind, labels, max_seq_len

def load_vocab_size(vocab_data_file=VOCABULARY_DICT_DIR):
    vocab_dict = Dictionary.load(vocab_data_file)
    return len(vocab_dict.items())

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
	Generates a batch iterator for a dataset.
	"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
