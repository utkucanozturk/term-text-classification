import pandas as pd
import numpy as np
from sklearn import pipeline, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from feature_engineering import *

class Modeling:

    def __init__(self, classifier = None, param_grid = None):
        fe = FeatureEngineering()
        self.train = fe.train
        self.test = fe.test
        self.vectorizer = fe.vectorizer
        self.classifier = classifier
        self.param_grid = param_grid

    def construct_pipeline(self):
        self.pipeline = pipeline.Pipeline([("vectorizer", self.vectorizer), ("classifier", self.classifier)])
    
    def search_fit(self):
        print('Searching parameter space this')
        self.construct_pipeline()
        search = GridSearchCV(self.pipeline, self.param_grid, cv = config.inner_cv, refit = True)
        search.fit(self.train['text_clean'], self.train["label_encoded"].values)

        return search
    
    def classifier_fit(self):
        self.construct_pipeline()
        self.pipeline.fit(self.train['text_clean'], self.train["label_encoded"].values)
        
        return self.pipeline
    

class MultinomialNBModel(Modeling):

    def __init__(self):
        super().__init__()
        self.init_classifier()
    
    def init_classifier(self):

        if config.tune_param:
            self.classifier = MultinomialNB()
            self.param_grid = {'vectorizer__max_df': config.vectorizer__max_df,
                                'classifier__alpha': config.MultinomialNB__alpha,
                                'classifier__fit_prior': config.MultinomialNB__fit_prior,
                                }
            self.model = self.search_fit()
        else:
            self.classifier = MultinomialNB(alpha = config.MultinomialNB__alpha_best, fit_prior = config.MultinomialNB__fit_prior_best)
            self.vectorizer.max_df = config.vectorizer__max_df_best
            self.model = self.classifier_fit()
        
        with open('./model/MultinomialNBModel.pkl', 'wb') as fin:
            pickle.dump(self.model, fin)


class SGDModel(Modeling):

    def __init__(self):
        super().__init__()
        self.init_classifier()
    
    def init_classifier(self):

        if config.tune_param:
            self.classifier = SGDClassifier()
            self.param_grid = {'vectorizer__max_df': config.vectorizer__max_df,
                                'classifier__penalty': config.SGDClassifier__penalty,
                                'classifier__max_iter': config.SGDClassifier__max_iter,
                                'classifier__tol': config.SGDClassifier__tol,
                                'classifier__loss': config.SGDClassifier__loss,
                                }
            self.model = self.search_fit()
        else:
            self.classifier = SGDClassifier(penalty = config.SGDClassifier__penalty_best,
                                            max_iter = config.SGDClassifier__max_iter_best,
                                            tol = config.SGDClassifier__tol,
                                            loss = config.SGDClassifier__loss_best,
                                            )
            self.vectorizer.max_df = config.vectorizer__max_df_best
            self.model = self.classifier_fit()
        
        with open('./model/SGDModel.pkl', 'wb') as fin:
            pickle.dump(self.model, fin)