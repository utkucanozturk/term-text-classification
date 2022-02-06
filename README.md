# term-text-classification

Classification of scientific terms from Wikipedia search texts for terms. Applied and evaluated Stochastic Gradient Descent and Multinomial Naive Bayes Model. Comparison of results with pretrained BERT model.

Folder structure -->

- conf
	- configs.yaml

- data
	- interim
		- *train.pkl
		- *test.pkl
		- *vectorizer.pkl
	- meta
		- data_dict.xlsx
	- raw
		- *terms.pkl
		- *labels.pkl
		- *texts.pkl
		- *dataset.pkl
	- training_sets
		- *dataset_v001.pkl

- model
	- MultinomialNBModel.pkl
	- SGDModel.pkl

- notebooks
	- evaluation.ipynb

- src\scripts
	- configs.py
	- data_preperation.py
	- feature_engineering.py
	- init.py
	- main.py
	- modeling.py

Note: .pkl files marked with '*' is not included in this repository.