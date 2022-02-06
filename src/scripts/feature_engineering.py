import pandas as pd
import numpy as np
import nltk
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle
import os
#nltk.download('stopwords')
#nltk.download('wordnet')

from data_preperation import *

class FeatureEngineering:

    def __init__(self, wd = './data/training_sets/', lst_stopwords = nltk.corpus.stopwords.words("english")):

        self.wd = wd
        self.data = self.load_data()
        self.encode_labels()
        self.clean_text(lst_stopwords = lst_stopwords)
        self.train, self.test = self.split_data()
        self.vectorizer = self.dump_vectorizer()
    

    def load_data(self):
        
        lst = sorted(os.listdir(self.wd))
        if len(lst) == 0:
            df = ProcessData().data
        else:
            print('Reading preprocessed data from directory...')
            file_name = lst[-1]  # get the latest dataset name
            try:
                df = pd.read_pickle(self.wd + file_name)
            except:
                raise RuntimeError('Error while reading ' + file_name + '!')

        return df

    
    def encode_labels(self):

        print('Encoding labels...')
        label_encoder = LabelEncoder()
        self.data["label_encoded"] = label_encoder.fit_transform(self.data["label"])
    
    
    def clean_text(self, lst_stopwords):
        
        def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
            '''
            Preprocess a string.
            :parameter
                :param text: string - name of column containing text
                :param lst_stopwords: list - list of stopwords to remove
                :param flg_stemm: bool - whether stemming is to be applied
                :param flg_lemm: bool - whether lemmitisation is to be applied
            :return
                cleaned text
            '''
        
            ## clean (convert to lowercase and remove punctuations and characters and then strip)
            text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
                    
            ## Tokenize (convert from string to list)
            lst_text = text.split()
            ## remove Stopwords
            if lst_stopwords is not None:
                lst_text = [word for word in lst_text if word not in 
                            lst_stopwords]
                        
            ## Stemming (remove -ing, -ly, ...)
            if flg_stemm == True:
                ps = nltk.stem.porter.PorterStemmer()
                lst_text = [ps.stem(word) for word in lst_text]
                        
            ## Lemmatisation (convert the word into root word)
            if flg_lemm == True:
                lem = nltk.stem.wordnet.WordNetLemmatizer()
                lst_text = [lem.lemmatize(word) for word in lst_text]
                    
            ## back to string from list
            text = " ".join(lst_text)
            return text
        
        print('Extracting clean text from the text...')
        self.data['text_clean']=self.data['text'].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
    

    def split_data(self, size = config.test_size):
        '''
        Split dataframe into train and test sets.
        :parameter
            :param df: dataframe to be splitted
            :param size: float - test data proportion
        :return
            train-test dataframe
        '''

        print('Splitting data into train and test sets...')
        df_train, df_test = train_test_split(self.data, test_size = size)
        
        with open('./data/interim/train.pkl', 'wb') as fin:
            pickle.dump(df_train, fin)
        
        with open('./data/interim/test.pkl', 'wb') as fin:
            pickle.dump(df_test, fin)
        
        return df_train, df_test

    
    def dump_vectorizer(self, feature_selection = config.feature_select, max_features = config.vectorizer_max_features, ngram_range = config.ngram, p_value = config.feature_selection_p_value):
        '''
        Dumps a Tfidf vectorizer with vocabulary
        :parameter
            :param feature_selection: boolean for feature selection
            :param max_features: integer - max features for tfidf vectorizer
            :param ngram_range: ngram range for tfidf vectorizer
            :param p_value: float - p_value for feature selection 
        '''
        print('Initializing Tfidf vectorizer...')
        corpus = self.train['text_clean']
        vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range)
        vectorizer.fit(corpus)
        X_train = vectorizer.transform(corpus)
        y_train = self.train['label_encoded'].values
        X_names = vectorizer.get_feature_names()

        if feature_selection:
            print('Applying feature selection with chi2 metric...')
        
            df_features = pd.DataFrame()

            for cat in np.unique(y_train):
                chisqu, p = chi2(X_train, y_train==cat)
                df_features = df_features.append(pd.DataFrame({"feature":X_names, "score":1-p, "y":cat}))
                df_features = df_features.sort_values(["y","score"], ascending=[True,False])
                df_features = df_features[df_features["score"] > p_value]
            
            X_names = df_features["feature"].unique().tolist()
            vectorizer = TfidfVectorizer(vocabulary=X_names)
        
        with open('./data/interim/vectorizer.pkl', 'wb') as fin:
            pickle.dump(vectorizer, fin)
        
        return vectorizer
