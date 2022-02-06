import pandas as pd
import wikipedia
import os, stat
import pickle

import configs as config

class ProcessData:

    def __init__(self, base_wd = './data/', balance = True, fill_missing = False, missing_terms = ['No Results', 'DisambiguationError'], balance_label_proportion = 0.1):
        self.base_wd = base_wd
        self.missing_terms = missing_terms

        print('Reading raw data...')
        self.data = self.read_raw()

        print('Starting to process raw data!')
        label_sample_size = round(balance_label_proportion * len(self.data))
        self.process_data(balance = balance, fill_missing = fill_missing, label_sample_size=label_sample_size)

        if config.save_preprocessed_dataset:
            self.save_dataset()


    def read_raw(self):

        raw_dir = self.base_wd + 'raw/'

        if os.path.exists(raw_dir + 'dataset.pkl'):
            df = pd.read_pickle(raw_dir + 'dataset.pkl')
        else:
            try:
                labels = pd.read_pickle(raw_dir + 'labels.pkl')
                terms = pd.read_pickle(raw_dir + 'terms.pkl')
                texts = pd.read_pickle(raw_dir + 'texts.pkl')
            except:
                raise RuntimeError('Make sure that the data is placed under raw folder!')
            
            try:
                df = pd.DataFrame({'term': terms, 'text': texts, 'label': labels})
            except:
                raise RuntimeError('Make sure that labels, terms and texts are in the same length!')

            df.text = df.text.map(lambda x: x[0])
            
            with open(raw_dir + 'dataset.pkl', 'wb') as out:
                pickle.dump(df, out)

        return df
    

    def process_data(self, balance, fill_missing, label_sample_size):

        print('Dropping duplicate terms...')
        self.data.drop_duplicates(subset='term', inplace=True)
        
        if fill_missing is True:
            try:
                print('Filling missing text from wiki...')
                self.wiki_fill()
            except:
                print('Filling from wiki failed; dropping rows with missing text...')
                self.drop_missing()
        else:
            print('Dropping rows with missing text...')
            self.drop_missing()
        
        if balance is True:
            print('Creating balanced dataset...')
            self.data = self.create_balanced(label_sample_size)
        
        self.data.reset_index(drop=True, inplace=True)
    

    def create_balanced(self, label_sample_size):

        df_balanced ={}
        
        for i in self.data['label'].unique():

            df_i = self.data[self.data['label']==i]

            if len(df_i) >= label_sample_size:
                df_balanced[i] = df_i.sample(label_sample_size)
            else:
                df_balanced[i] = df_i.sample(label_sample_size, replace=True)

        return pd.concat(df_balanced)
    

    def wiki_fill(self):

        term_no_results = self.data.loc[self.data.text.isin(self.missing_terms), 'term']
        
        for i in term_no_results:
            if len(i) <= 300: # maximum allowed wiki api search length
                term_search = wikipedia.search(i, results=0)
            else:
                continue

            if len(term_search) !=0:
                self.data['text'] = wikipedia.page(term_search[0]).content
    
    def drop_missing(self):

        term_no_results = self.data.loc[self.data.text.isin(self.missing_terms), 'term']
        self.data.drop(index=term_no_results.index, inplace=True)
    
    
    def save_dataset(self):
        
        ts_dir = self.base_wd + 'training_sets/'

        lst = sorted(os.listdir(ts_dir))
        if len(lst) == 0:
            v_no = 1
        else:
            v_no = int(lst[-1][-7:-4]) +1

        v_no = str(v_no).zfill(3)

        fname = ts_dir + 'dataset_v' + v_no + '.pkl'
        with open(fname, 'wb') as out:
            pickle.dump(self.data, out)
            os.chmod(fname, stat.S_IRWXO)
        print('Dataset version ' + v_no + ' saved!')
        out.close()