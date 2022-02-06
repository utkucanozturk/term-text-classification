"""DATA PREPERATION"""
save_preprocessed_dataset = True

"""FEATURE ENGINEERING"""
test_size = 0.3
vectorizer_max_features = 10000
ngram = (1,2)
feature_select = True
feature_selection_p_value = 0.95

"""TUNING"""
tune_param = False
inner_cv = 3
vectorizer__max_df = (.25, .5, .75, 1)
MultinomialNB__alpha = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
MultinomialNB__fit_prior = [True, False]
SGDClassifier__penalty = ('l2', 'elasticnet', 'l1')
SGDClassifier__max_iter = [100, 200]
SGDClassifier__tol = [0.0001]
SGDClassifier__loss = ['hinge', 'log', 'modified_huber']

"""BEST PARAM"""
vectorizer__max_df_best = 0.25
MultinomialNB__alpha_best = 0.5
MultinomialNB__fit_prior_best = False
SGDClassifier__penalty_best = 'elasticnet'
SGDClassifier__max_iter_best = 100
SGDClassifier__tol = 0.0001
SGDClassifier__loss_best = 'log'


