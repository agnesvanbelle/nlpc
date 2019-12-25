import os
from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from chatterbot.response_selection import get_random_response
from chatterbot.comparisons import levenshtein_distance

from utils import *

# set to any value to limit the number of threads loaded per programming language
# (to save memory)
MAX_TRDS_TO_LOAD = None #150000

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        
        if MAX_TRDS_TO_LOAD is not None: # sample a predefined number of tags
            indices = range(0, thread_ids.shape[0])
            random_indices_choice = np.random.choice(indices, size=min(len(indices), MAX_TRDS_TO_LOAD), 
                                                  replace=False)
            thread_ids = thread_ids[random_indices_choice,]
            thread_embeddings = thread_embeddings[random_indices_choice,]

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        
        min_dist = pairwise_distances_argmin(question_vec.reshape(1, -1), thread_embeddings, axis=1, metric='cosine')
        
        best_thread = min_dist[0]
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")
        
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])
        
        self.ANSWER_TEMPLATE = "I think its about {:s}.\n" + \
                    "For this problem this thread might help you: https://stackoverflow.com/questions/{:d} "

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        
        # Bot part:
        self.create_chitchat_bot(paths['CHATTERBOT_LISTTRAINDATA'])
       

    def create_chitchat_bot(self, list_train_data_path):
        """Initializes self.chitchat_bot with some conversational model."""

        
        self.bot = ChatBot('My ChatBot', logic_adapters=[         
                                {"import_path": "chatterbot.logic.BestMatch",
                                 'maximum_similarity_threshold': 0.80,
                                  "statement_comparison_function": levenshtein_distance                                                   }
                             ], response_selection_method=get_random_response)
        
        trainer = ChatterBotCorpusTrainer(self.bot)
        # Don't train on the whole corpus, only relevant parts, to be faster when replying
        trainer.train(  'chatterbot.corpus.english.greetings',
                        'chatterbot.corpus.english.conversations',
                        'chatterbot.corpus.english.emotion',
                        'chatterbot.corpus.english.psychology',
                        'chatterbot.corpus.english.science',
                        'chatterbot.corpus.english.trivia',
                        'chatterbot.corpus.english.botprofile')
        
        # Train also on the extra conversations we generated        
        data = open(list_train_data_path, encoding='utf-8').read()
        conversations = data.strip().split('\n')
        trainer2 = ListTrainer(self.bot)
        trainer2.train(conversations)
        
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        
        # Intent recognition
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]
      
        # Chit-chat part:   
        if intent == 'dialogue':            
            # Pass question to chitchat_bot to generate a response.       
            response = self.bot.get_response(question)
            response_text = response.text
            return response_text
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
              
            return self.ANSWER_TEMPLATE.format(tag, thread_id)

