# standards imports
import pandas as pd

# extra imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from gensim import corpora, matutils, models
from gensim.similarities.docsim import MatrixSimilarity

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.sparse import save_npz, load_npz, csr_matrix

from words_clean_function import normalize_text, denoise_text, lemmatize_verbs
import seaborn as sns


def evaluate_matrix(matrix, labels, classifier, data_set_name):
	# Define the k-fold cross-validation
	kf = KFold(n_splits=10, shuffle=True, random_state=42)

	# Initialize the metrics list
	acc_list = []
	prec_list = []
	recall_list = []
	f1_list = []

	# Perform the k-fold cross-validation
	for train_index, test_index in kf.split(matrix):
		X_train, X_test = matrix[train_index], matrix[test_index]
		y_train, y_test = labels[train_index], labels[test_index]

		# Fit the classifier on the training set
		classifier.fit(X_train, y_train)

		# Make predictions on the test set
		y_pred = classifier.predict(X_test)

		# Calculate the evaluation metrics
		acc = accuracy_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred, average='weighted')
		recall = recall_score(y_test, y_pred, average='weighted')
		f1 = f1_score(y_test, y_pred, average='weighted')

		# Append the metrics to the lists
		acc_list.append(acc)
		prec_list.append(prec)
		recall_list.append(recall)
		f1_list.append(f1)

	# Calculate the mean metrics
	acc_mean = sum(acc_list) / len(acc_list)
	prec_mean = sum(prec_list) / len(prec_list)
	recall_mean = sum(recall_list) / len(recall_list)
	f1_mean = sum(f1_list) / len(f1_list)

	# Save the metrics to a file
	result_df = pd.DataFrame({'Classifier': [str(classifier)],
	                          'Data Set': [data_set_name],
	                          'Accuracy': [acc_mean],
	                          'Precision': [prec_mean],
	                          'Recall': [recall_mean],
	                          'F1': [f1_mean]})
	with open('result_metrics.csv', 'a') as f:
		result_df.to_csv(f, header=f.tell() == 0, index=False)


# load data
X_parse = load_npz('./data/tf_sparse.npz')
X_dense = np.load('./data/tf_idf.npy')
y = pd.read_csv('./data/news_labels.csv')['label']

# Random Forest
# clf = RandomForestClassifier()
clf = XGBClassifier()

# For tdm_matrix
evaluate_matrix(X_parse, y, clf, data_set_name='tdm')
evaluate_matrix(X_dense, y, clf, data_set_name='tf-idf')
