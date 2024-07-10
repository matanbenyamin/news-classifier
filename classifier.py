import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
import spacy
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#bunch of classificaion models
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import LatentDirichletAllocation

class TextClassifier:
    def __init__(self, model=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.pipeline = None
        self.model = model

    def get_named_entities(self, text):
        doc = self.nlp(text)
        return len([ent.label_ for ent in doc.ents])

    def get_pos_tags(self, text, tag):
        doc = self.nlp(text)
        return len([token.pos_ for token in doc if token.pos_ == tag])

    def get_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def getg_subjectivity(self, text):
        blob = TextBlob(text)
        return blob.sentiment.subjectivity

    def preprocess_data(self, df):
        df['title_ner'] = df['scraped_title'].apply(self.get_named_entities)
        df['title_nouns'] = df['scraped_title'].apply(lambda x: self.get_pos_tags(x, 'NOUN'))
        df['title_verbs'] = df['scraped_title'].apply(lambda x: self.get_pos_tags(x, 'VERB'))
        df['title_sentiment'] = df['scraped_title'].apply(self.get_sentiment)
        df['title_subjectivity'] = df['scraped_title'].apply(self.getg_subjectivity)
        return df

    def build_pipeline(self):
        tfidf_preprocessor = ColumnTransformer(
            transformers=[
                ('title_tfidf', TfidfVectorizer(), 'scraped_title'),
                ('url_tfidf', TfidfVectorizer(), 'url'),
                ('text_tfidf', TfidfVectorizer(), 'scraped_text')
            ]
        )

        non_text_preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), ['title_ner', 'title_nouns', 'title_verbs', 'title_sentiment', 'title_subjectivity'])
            ]
        )

        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('tfidf_pipeline', Pipeline([
                    ('tfidf', tfidf_preprocessor),
                    ('selector', SelectKBest(chi2, k=2750))
                ])),
                ('non_text', non_text_preprocessor)
            ])),
            ('clf',self.model)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_feature_importances(self):
        clf = self.pipeline.named_steps['clf']
        tfidf_preprocessor = self.pipeline.named_steps['features'].transformer_list[0][1].named_steps['tfidf']
        tfidf_selector = self.pipeline.named_steps['features'].transformer_list[0][1].named_steps['selector']

        title_features = tfidf_preprocessor.named_transformers_['title_tfidf'].get_feature_names_out().tolist()
        url_features = tfidf_preprocessor.named_transformers_['url_tfidf'].get_feature_names_out().tolist()
        text_features = tfidf_preprocessor.named_transformers_['text_tfidf'].get_feature_names_out().tolist()

        selected_title_features = [title_features[i] for i in tfidf_selector.get_support(indices=True) if
                                   i < len(title_features)]
        selected_url_features = [url_features[i - len(title_features)] for i in tfidf_selector.get_support(indices=True)
                                 if len(title_features) <= i < len(title_features) + len(url_features)]
        selected_text_features = [text_features[i - len(title_features) - len(url_features)] for i in
                                  tfidf_selector.get_support(indices=True) if
                                  i >= len(title_features) + len(url_features)]

        other_features = ['title_ner', 'title_nouns', 'title_verbs', 'title_sentiment', 'title_subjectivity']

        feature_names = selected_title_features + selected_url_features + selected_text_features + other_features

        coefs = clf.coef_[0]
        feature_importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coefs
        }).sort_values(by='importance', ascending=False)
        return feature_importances_df

    def plot_feature_importances(self, n=10):
        feature_importances_df = self.get_feature_importances()
        plt.figure(figsize=(10, 10))
        temp_df = feature_importances_df.head(n)
        sns.barplot(x='importance', y='feature', data=temp_df)
        plt.title('Top Feature Importances')
        plt.show()

    def save_model(self, file_path):
        joblib.dump(self.pipeline, file_path)

    def load_model(self, file_path):
        self.pipeline = joblib.load(file_path)

