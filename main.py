import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class NewsClassifier:
    def __init__(self, df):
        df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')
        self.df = df
        self.title_vectorizer = CountVectorizer()
        self.title_tfidf = TfidfTransformer()
        pass
    def fit(self, X, y):
        X_counts = self.title_vectorizer.fit_transform(X['scraped_title'])
        self.title_tfidf = self.title_tfidf.fit_transform(X_counts)
        url_counts = self.url_vectorizer.fit_transform(X['url'])
        url_tfidf = TfidfTransformer()
        url_tfidf = url_tfidf.fit_transform(url_counts)
        self.title_tfidf =  pd.DataFrame(self.title_tfidf.toarray())
        url_tfidf = pd.DataFrame(url_tfidf.toarray())

        return self
    def preprocess(self, X):
        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(X['scraped_title'])
        # tfidf
        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)
        # add tfidf of the url
        url_vectorizer = CountVectorizer()
        url_counts = url_vectorizer.fit_transform(X['url'])
        url_tfidf = TfidfTransformer()
        url_tfidf = url_tfidf.fit_transform(url_counts)
        X_tfidf =  pd.DataFrame(X_tfidf.toarray())

        return X_tfidf





data_path = r'/Users/matanb/Downloads/DV_NLP_assignment/assignment_data_en.csv'

df = pd.read_csv(data_path)


# preprocess
df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')



# train test split
# X is all numerical features

X = df[['scraped_title','url']]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# basic feature extraction of df['scraped_title']
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train['scraped_title'])
# tfidf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



# add tfidf of the url
url_vectorizer = CountVectorizer()
url_counts = url_vectorizer.fit_transform(X_train['url'])
url_tfidf = TfidfTransformer()
url_tfidf = url_tfidf.fit_transform(url_counts)
X_train_tfidf =  pd.DataFrame(X_train_tfidf.toarray())
url_tfidf = pd.DataFrame(url_tfidf.toarray())
X_train_tfidf = pd.concat([X_train_tfidf, url_tfidf], axis=1)




# train a model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# predict
X_test_counts = vectorizer.transform(X_test['scraped_title'])
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
url_counts = url_vectorizer.transform(X_test['url'])
url_tfidf = TfidfTransformer()
url_tfidf = url_tfidf.fit_transform(url_counts)
X_test_tfidf =  pd.DataFrame(X_test_tfidf.toarray())
url_tfidf = pd.DataFrame(url_tfidf.toarray())
X_test_tfidf = pd.concat([X_test_tfidf, url_tfidf], axis=1)
predicted = clf.predict(X_test_tfidf)


# evaluate
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predicted)

# try random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try SVM
from sklearn.svm import SVC
clf = SVC().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)

# try XGBoost
from xgboost import XGBClassifier
clf = XGBClassifier().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
accuracy_score(y_test, predicted)


