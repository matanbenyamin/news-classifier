import pandas as pd
from classifier import NewsClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
# accuracy report
from sklearn.metrics import classification_report

# Load the data
classifier = NewsClassifier.load('model.pkl')

# Load the data
df = classifier.df
df = df[df['split'] == 'test']

# Predict
df['classification'] = classifier.predict(df)

# show ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
y_true = df['class']
y_true = y_true.apply(lambda x: 0 if x == 'news' else 1)
X = classifier.preprocess(df)
y_pred = classifier.model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
# add the f1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# add the text scores
plt.text(0.6, 0.2, f'AUC: {np.round(roc_auc_score(y_true, y_pred), 2)}', fontsize=12)
plt.text(0.6, 0.1, f'Accuracy: {np.round(accuracy_score(df["class"], df["classification"]), 2)}', fontsize=12)
plt.show()

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



#


# accuracy metrics
rep  = classification_report(df['class'], df['classification'])
print(rep)


# plot feaure importance with shap
# shap.initjs()
# explainer = shap.TreeExplainer(classifier.model)
X = classifier.preprocess(df)
# shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, classifier.X_test)

# Feature importance with sklearn
importances = classifier.model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature importances")
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.show()


# We will try to prune the model based on these results



# show some failures
fails = df[df['classification'] != df['class']]
fp = fails[fails['classification'] == 'news']
fn = fails[fails['classification'] == 'non-news']

# fp pie chart
fp['content_type'].value_counts().plot.pie()
plt.title('False Positives')




