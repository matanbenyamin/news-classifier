## A script to run after training the model to analyze the errors
## possible commented way of loading th model for analysis on new data


import pandas as pd
from classifier import TextClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Load the data
#classifier = TextClassifier().load_model('text_classifier.pkl')



# show ROC curve
pred = classifier.Pipeline.predict_proba(X_test)[:, 1]


# look for avoerfit = predict on the training set
df = classifier.df
df['classification'] = classifier.predict(classifier.df)
# compare the scores
acc_train = accuracy_score(classifier.df[classifier.df['split'] == 'train']['class'], classifier.df[classifier.df['split'] == 'train']['classification'])
acc_test = accuracy_score(classifier.df[classifier.df['split'] == 'test']['class'], classifier.df[classifier.df['split'] == 'test']['classification'])
acc_total = accuracy_score(classifier.df['class'], classifier.df['classification'])
# roun
print(f'Train accuracy: {acc_train}')
print(f'Test accuracy: {acc_test}')
print(f'Total accuracy: {acc_total}')


# accuracy metrics
rep  = classification_report(df['class'], df['classification'])
print(rep)


# We will try to prune the model based on these results



# show some failures
fails = df[df['classification'] != df['class']]
fp = fails[fails['classification'] == 'news']
fn = fails[fails['classification'] == 'non-news']

# fp pie chart
fp['content_type'].value_counts().plot.pie()
plt.title('False Positives')




