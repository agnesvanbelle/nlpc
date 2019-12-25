import os
from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from chatterbot.response_selection import get_random_response


from utils import *


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
        
        indices = np.random.randint(0, thread_ids.shape[0], 100000)
        thread_ids_sample = thread_ids[indices,]
        thread_embeddings_sample = thread_embeddings[indices,]
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        
        min_dist = pairwise_distances_argmin(question_vec.reshape(1, -1), thread_embeddings_sample, axis=1, metric='cosine')
        
        best_thread = min_dist[0]
        
        #sims = cosine_similarity(question_vec.reshape(1, -1), thread_embeddings)
        #sims = sims.squeeze().tolist()
        #print('sims len:', len(sims))
        #result = [c[0] for c in 
        #          sorted(enumerate(sims), key = lambda x: x[1], reverse=True)
        #     ][0]
         
        #print('result:', result)
        #best_thread = result
        
        return thread_ids_sample[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")
        
        self.paths = paths
        
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
                                 'maximum_similarity_threshold': 0.70
                                },    
                                # also support logical operations but with higher threshold
                                {"import_path": 'chatterbot.logic.MathematicalEvaluation',
                                 'maximum_similarity_threshold': 0.95
                                }    
                             ], response_selection_method=get_random_response)
        
        trainer = ChatterBotCorpusTrainer(self.bot)
        trainer.train('chatterbot.corpus.english')
        
        # Train also on the extra conversations we generated        
        data = open(list_train_data_path, encoding='utf-8').read()
        conversations = data.strip().split('\n')
        trainer2 = ListTrainer(self.bot)
        trainer2.train(conversations)
        
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        
        # Intent recognition:
        self.intent_recognizer = unpickle_file(self.paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(self.paths['TFIDF_VECTORIZER'])
        
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

