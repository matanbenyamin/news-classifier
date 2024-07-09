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
from classifier import NewsClassifier


# =======
# Based on EDA conclusions, we will try to train a model basde on:
# 1. TF-IDF transofrmed title and URL
# 2. Sentiment analysis of the title (subectivity and polarity)
# 3. num nouns and num verbs in the title
# 4. amount of named entities


def __main__():
    data_path = r'./assignment_data_en.csv'
    df = pd.read_csv(data_path)
    # preprocess
    df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')

    # Try several models and textual inputs
    # classifier = NewsClassifier(df,columns=['scraped_title'], model = SVC()) # 0.72
    # classifier = NewsClassifier(df, columns=['scraped_title', 'url'],
    #                             numerical_features=['sentiment_polarity', 'sentiment_subjectivity',
    #                                                 'num_nouns', 'num_verbs', 'num_adjectives', 'num_named_entities'],
    #                             model=RandomForestClassifier(max_depth = 5))  # 0.8
    # classifier = NewsClassifier(df, columns=['scraped_title', 'url'],
    #                             numerical_features=['sentiment_polarity', 'sentiment_subjectivity',
    #                                                 'num_nouns', 'num_verbs', 'num_adjectives', 'num_named_entities'],
    #                             model = LogisticRegression()) # 0.8
    # classifier = NewsClassifier(df,columns=['scraped_title','url'], model = HistGradientBoostingClassifier()) # 0.8
    # svm
    classifier = NewsClassifier(df, columns=['scraped_title', 'url'],
                                numerical_features=['sentiment_polarity', 'sentiment_subjectivity',
                                                    'num_nouns', 'num_verbs', 'num_adjectives', 'num_named_entities'],
                                model = SVC()) # 0.8
    classifier.fit()
    # save the model
    classifier.save('model.pkl')




    # Perform Eror Analysis

    # realizie we have overfitting

    # tr opne of the following
    # Pruining the tree
    # Regularization
    # Cross validation (K Fold)
    # Feature selection
    # Ensemble methods
    # Data Augmentation
    # Transfer Learning
    # Hyperparameter tuning




if __name__ == '__main__':
    __main__()
