import numpy as np
import random
import nltk
import collections
import operator
import json
from data_loader import *

ann_file = '../data/results_20130124.token'
pad_limit = 20


def random_glove_generator(emb_mean, emb_stddev):
    x = np.random.normal(loc=0.0, scale=1.0, size=len(emb_mean))
    x_rand = np.multiply(x, emb_stddev) + emb_mean
    return x_rand


def gen_annotations(ann_file):
    print("Loading Annotations")

    annotations = collections.defaultdict(list)
    with open(ann_file, encoding='utf-8') as f:
        for line in f:
            image_id, caption = line.strip().split('\t')
            if len(caption.split(" ")) <= pad_limit:
                annotations[image_id[:-2]].append(caption)

    return annotations

def gen_vocabulary(annotations):

    all_words = []
    for item in annotations.keys():
        for caption in annotations[item]:
            all_words.extend(caption.split(' '))

    all_words = [word.lower() for word in all_words]
    vocabulary = list(set(all_words))
    vocabulary.extend(["<start>","<end>","<unk>"])
    print('Total Words found: ', len(all_words))
    print("Vocabulary length: ", len(vocabulary))

    return vocabulary


def vocab_glove_list(embed_file):
    embedding_file = open(embed_file, encoding='utf8', mode='r')

    glove = {}
    for line in embedding_file:
        splitted = line.split()
        word = splitted[0]
        emb = np.array([float(val) for val in splitted[1:]])
        glove[word] = emb

    print("Glove Dictionary successfully built!")
    return glove


def create_glove_for_vocab(vocabulary, embed_file):
    glove_dict = vocab_glove_list(embed_file)

    all_embeddings = np.stack(glove_dict.values())
    emb_mean = all_embeddings.mean(axis=0)
    emb_stddev = all_embeddings.std(axis=0)

    vocab2glove = {}
    for word in vocabulary:
        if word not in glove_dict.keys():
            vocab2glove[word] = random_glove_generator(emb_mean, emb_stddev).tolist()
        else:
            vocab2glove[word] = glove_dict[word].tolist()
    return vocab2glove





if __name__ == "__main__":

    annotations = gen_annotations(ann_file)
    vocabulary = gen_vocabulary(annotations)
    embed_file = '../data/glove.6B.300d.txt'
    vocab_glove_file = '../data/vocab_glove_flickr.json'


    vocab_embedding = create_glove_for_vocab(vocabulary, embed_file)

    print('Generating JSON file!')
    with open(vocab_glove_file, "w") as fp:
        json.dump(vocab_embedding, fp)
    print('Generated JSON file!')


