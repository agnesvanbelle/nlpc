import nltk
import pickle
import re
import numpy as np
from gensim.models import KeyedVectors
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': '../week3/data/starspace/testModel_w2vf.txt',
    'CHATTERBOT_LISTTRAINDATA': 'movie_conversations_20movies.txt'
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # this will load it in np.float32 type for the numbers in the embedding vectors
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    
    dim = embeddings[embeddings.index2word[0]].size
    
    return embeddings, dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    start_vector = np.zeros(dim)
    counter = 0
    for word in question.split():
        try:
            word_embedding = embeddings[word]
            counter += 1
            start_vector = start_vector + word_embedding
        except KeyError as e:
            if 'not in vocabulary' in str(e):
                pass
    mean_vector = start_vector / counter if counter > 0 else start_vector
    return mean_vector


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
