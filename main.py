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
from classifier import TextClassifier


# =======
# Based on EDA conclusions, we will try to train a model basde on:
# 1. TF-IDF transofrmed title, text and URL
# 2. Sentiment analysis of the title (subectivity and polarity)
# 3. num nouns and num verbs in the title
# 4. amount of named entities


def __main__():

    # Load your DataFrame
    df = pd.read_csv('assignment_data_en.csv')
    df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')

    # Initialize the classifier
    classifier = TextClassifier(model=LogisticRegression())
    # === here I tried several classifiers and paranmeters

    df = classifier.preprocess_data(df)

    # Split the data
    X = df[['scraped_title', 'scraped_text', 'url', 'title_ner', 'title_nouns', 'title_verbs', 'title_sentiment', 'title_subjectivity']]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Build the pipeline and train
    classifier.build_pipeline()
    classifier.train(X_train, y_train)

    # Evaluate the model
    classifier.evaluate(X_test, y_test)

    classifier.save_model('text_classifier.pkl')

    # Get feature importances
    feature_importances = classifier.get_feature_importances()
    classifier.plot_feature_importances()

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


    # Perform Eror Analysis

    # realizie we have overfitting

    # tr opne of the following
    # Pruining the tree
    # Regularization
    # Cross validation (K Fold)
    # Feature selection
    # Hyperparameter tuning




    ## Error Analysis
    # We will try to prune the model based on these results
    # show some failures
    X_test['classification'] = classifier.predict(X_test)
    fails = X_test[X_test['classification'] != y_test]
    fp = fails[fails['classification'] == 'news']
    fn = fails[fails['classification'] == 'non-news']

    # fp pie chart
    fp['content_type'].value_counts().plot.pie()

    # show ROC curve






if __name__ == '__main__':
    __main__()
