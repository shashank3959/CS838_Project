import pickle
import numpy as np
import os

# Make sure these files are present at the expected location
vocab_file = 'vocab.pkl'
embed_file = '../data/glove.6B.300d.txt'


def random_glove_generator(emb_mean, emb_stddev):
    x = np.random.normal(loc=0.0, scale=1.0, size=len(emb_mean))
    x_rand = np.multiply(x, emb_stddev) + emb_mean
    return x_rand


def vocab_glove_list(vocab_file, embed_file):
    embedding_file = open(embed_file, encoding='utf8', mode='r')
    f = open(vocab_file, "rb")
    vocab = pickle.load(f)
    word2idx = vocab.word2idx
    # idx2word = vocab.idx2word
    print("Vocabulary successfully loaded from vocab.pkl file!")

    all_words_vocab = list(word2idx.keys())

    glove = {}
    for line in embedding_file:
        splitted = line.split()
        word = splitted[0]
        emb = np.array([float(val) for val in splitted[1:]])
        glove[word] = emb

    print("Glove Dictionary successfully built!")

    return all_words_vocab, glove


def create_glove_pickle(vocab_file, embed_file):
    vocab_list, glove_dict = vocab_glove_list(vocab_file, embed_file)

    all_embeddings = np.stack(glove_dict.values())
    emb_mean = all_embeddings.mean(axis=0)
    emb_stddev = all_embeddings.std(axis=0)

    vocab2glove = {}
    for word in vocab_list:
        if word not in glove_dict.keys():
            vocab2glove[word] = random_glove_generator(emb_mean, emb_stddev)
        else:
            vocab2glove[word] = glove_dict[word]
    return vocab2glove


if __name__ == "__main__":
    vocab_embedding = create_glove_pickle(vocab_file, embed_file)
    file_name = 'vocab_glove.pkl'
    print('generating pickle file!')
    with open(file_name, "wb") as f:
        pickle.dump(vocab_embedding, f)
    print('Generated pickle file!')

