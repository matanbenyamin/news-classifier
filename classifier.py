import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from textblob import TextBlob
import joblib
from feature_extraction import clean_text, get_pos_tags, get_named_entities
# sklearn.ensemble.HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =======
# Based on EDA conclusions, we will try to train a model basde on:
# 1. TF-IDF transofrmed title and URL
# 2. Sentiment analysis of the title (subectivity and polarity)
# 3. num nouns and num verbs in the title
# 4. amount of named entities


def get_named_entities(text):
    doc = nlp(text)
    return len([ent.label_ for ent in doc.ents])


# Part-of-Speech tag counts
def get_pos_tags(text):
    doc = nlp(text)
    return len([token.pos_ for token in doc])


# Sentiment Analysis
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


class NewsClassifier:
    def __init__(self, df, model =RandomForestClassifier, columns = ['scraped_title', 'scraped_text'],
                 numerical_features = ['sentiment_polarity', 'sentiment_subjectivity']):
        df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')
        self.df = df
        self.fitted = False
        self.columns = columns
        self.model = model
        self.numerical_features = numerical_features
        pass
    def fit(self, kfold = True):

        columns = self.columns
        X = self.df[columns]
        y = self.df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # add a 'asplit column to the original df
        self.df['split'] = 'train'
        self.df.loc[X_test.index, 'split'] = 'test'


        self.vectorizers = []
        self.tfidf_transformers = []
        for idx, col in enumerate(columns):
            self.vectorizers.append(CountVectorizer(max_features=10))
            self.tfidf_transformers.append(TfidfTransformer())

        # now fit all the transformers
        for idx, col in enumerate(columns):
            X_counts = self.vectorizers[idx].fit_transform(X_train[col])
            self.tfidf_transformers[idx] = self.tfidf_transformers[idx].fit(X_counts)


        # now transform the data
        X_train_tfidf = []
        for idx, col in enumerate(columns):
            X_counts = self.vectorizers[idx].transform(X_train[col])
            X_train_tfidf.append(self.tfidf_transformers[idx].transform(X_counts))
        # all to a dataframe
        X_train_tfidf = pd.concat([pd.DataFrame(X_train_tfidf[i].toarray()) for i in range(len(columns))], axis=1)
        # add the original index
        X_train_tfidf.index = X_train.index

        # add sentiment analysis
        if 'sentiment_polarity' in self.numerical_features:
            X_train['sentiment_polarity'] = X_train['scraped_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        if 'sentiment_subjectivity' in self.numerical_features:
            X_train['sentiment_subjectivity'] = X_train['scraped_title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        if 'num_nouns' in self.numerical_features:
            X_train['num_nouns'] = X_train['scraped_title'].apply(lambda x: sum(1 if y == 'NOUN' else 0 for y in get_pos_tags(x)))
        if 'num_verbs' in self.numerical_features:
            X_train['num_verbs'] = X_train['scraped_title'].apply(lambda x: sum(1 if y == 'VERB' else 0 for y in get_pos_tags(x)))
        if 'num_adjectives' in self.numerical_features:
            X_train['num_adjectives'] = X_train['scraped_title'].apply(lambda x: sum(1 if y == 'ADJ' else 0 for y in get_pos_tags(x)))
        if 'num_named_entities' in self.numerical_features:
            X_train['num_named_entities'] = X_train['scraped_title'].apply(lambda x: len(get_named_entities(x)))


        # rename all columns to strings
        X_train_tfidf.columns = [str(i) for i in range(X_train_tfidf.shape[1])]
        # X_train_tfidf = pd.concat([X_train_tfidf, X_train[['sentiment', 'subjectivity', 'num_nouns', 'num_verbs', 'num_named_entities']]], axis=1)
        X_train_tfidf = pd.concat([X_train_tfidf, X_train[self.numerical_features]], axis=1)

        # drop all the zero columns
        X_train_tfidf = X_train_tfidf.loc[:, (X_train_tfidf != 0).any(axis=0)]


        # normalize the columns
        self.col_means = X_train_tfidf.mean()
        self.col_stds = X_train_tfidf.std()
        X_train_tfidf = (X_train_tfidf - self.col_means) / self.col_stds

        # now train the model
        # Create a pipeline with feature selection and classifier
        pipeline = Pipeline([
            ('select_kbest', SelectKBest(score_func=chi2)),
            ('clf', DecisionTreeClassifier(random_state=42))
        ])

        if kfold:
            # K-Fold Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
            cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, cv=kf, scoring='accuracy')
            print(f'Cross-validation accuracy: {cv_scores.mean()}')


        self.clf = self.model.fit(X_train_tfidf, y_train)

        self.fitted = True

        return X_train, y_train, X_test, y_test

    def preprocess(self, X):
        X_tfidf = []
        for idx, col in enumerate(self.columns):
            X_counts = self.vectorizers[idx].transform(X[col])
            X_tfidf.append(self.tfidf_transformers[idx].transform(X_counts))
        X_tfidf = pd.concat([pd.DataFrame(X_tfidf[i].toarray()) for i in range(len(self.columns))], axis=1)
        X_tfidf.index = X.index

        # add sentiment analysis
        if 'sentiment_polarity' in self.numerical_features:
            X['sentiment_polarity'] = X['scraped_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        if 'sentiment_subjectivity' in self.numerical_features:
            X['sentiment_subjectivity'] = X['scraped_title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        if 'num_nouns' in self.numerical_features:
            X['num_nouns'] = X['scraped_title'].apply(lambda x: sum(1 if y == 'NOUN' else 0 for y in get_pos_tags(x)))
        if 'num_verbs' in self.numerical_features:
            X['num_verbs'] = X['scraped_title'].apply(lambda x: sum(1 if y == 'VERB' else 0 for y in get_pos_tags(x)))
        if 'num_adjectives' in self.numerical_features:
            X['num_adjectives'] = X['scraped_title'].apply(lambda x: sum(1 if y == 'ADJ' else 0 for y in get_pos_tags(x)))
        if 'num_named_entities' in self.numerical_features:
            X['num_named_entities'] = X['scraped_title'].apply(lambda x: len(get_named_entities(x)))



        # rename all columns to strings
        X_tfidf.columns = [str(i) for i in range(X_tfidf.shape[1])]
        # X_tfidf = pd.concat([X_tfidf, X[['sentiment', 'subjectivity', 'num_nouns', 'num_verbs', 'num_named_entities']]], axis=1)
        X_tfidf = pd.concat([X_tfidf, X[self.numerical_features]], axis=1)

        # drop all the columsn that are not in the training set
        X_tfidf = X_tfidf.loc[:, (X_tfidf.columns.isin(self.col_means.index))]



        # normalize the columns
        X_tfidf = (X_tfidf - self.col_means) / self.col_stds

        return X_tfidf

    def predict(self, X):
        X_tfidf = self.preprocess(X)
        return self.clf.predict(X_tfidf)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

class pipelineClassifier:
    def __init__(self, df, model =RandomForestClassifier(), columns = ['scraped_title', 'scraped_text'],
                 numerical_features = ['sentiment_polarity', 'sentiment_subjectivity']):
        df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')
        self.fitted = False
        self.columns = columns
        self.model = model
        self.numerical_features = numerical_features
        self.df = df


        pass
    def create_prepocessor(self):
        # Preprocessor for text and numerical features
        # cerate the relelvant transofrmers as found in self.columns and self.numerical_features


        preprocessor = ColumnTransformer(
            transformers=[
                [('title_tfidf', TfidfVectorizer(), 'scraped_title') if 'scraped_title' in self.columns else None],
                [('url_tfidf', TfidfVectorizer(), 'url') if 'url' in self.columns else None],
                [('text_tfidf', TfidfVectorizer(), 'scraped_text') if 'scraped_text' in self.columns else None],
                [('title_ner', StandardScaler(), 'title_ner') if 'title_ner' in self.numerical_features else None],
                [('title_pos', StandardScaler(), 'title_pos') if 'title_pos' in self.numerical_features else None],
                [('title_sentiment', StandardScaler(), 'title_sentiment') if 'title_sentiment' in self.numerical_features else None],
                [('text_sentiment', StandardScaler(), 'text_sentiment') if 'text_sentiment' in self.numerical_features else None],
                [('text_ner', StandardScaler(), 'text_ner') if 'text_ner' in self.numerical_features else None],
                [('text_pos', StandardScaler(), 'text_pos') if 'text_pos' in self.numerical_features else None],
                ('scaler', StandardScaler(), self.numerical_features)
            ],
            remainder='passthrough'
        )
        self.preprocessor = preprocessor
    def fit(self, kfold = True):
        # Split the data
        X = self.df[self.columns]
        y = self.df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Create a pipeline with feature selection and classifier
        self.create_prepocessor()
        # Pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('clf', self.model)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)



    def predict(self, X):
        return self.pipeline.predict(X)
