from modeling import *

class RunPipeline:

    def process(self):

        ProcessData()
    
    def feature_engineer(self):

        FeatureEngineering()

    def modeling(self):

        SGDModel()
        MultinomialNBModel()


if __name__ == '__main__':
    RunPipeline().modeling()
