import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import gensim.downloader as api
import spacy.cli



# download  necessary resources
nltk.download('punkt')
nltk.download('stopwords')
# if necesary download the spacy model
if not spacy.util.is_package('en_core_web_sm'):
    spacy.cli.download("en_core_web_sm")

def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

def extract_features(title):

    # clean the text
    title = clean_text(title)


    # Tokenize the title
    words = word_tokenize(title)
    word_count = len(words)
    character_count = len(title)
    average_word_length = sum(len(word) for word in words) / word_count if word_count else 0
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    capital_letters_count = sum(1 for char in title if char.isupper())
    digits_count = sum(1 for char in title if char.isdigit())
    special_characters_count = len(re.findall(r'[^A-Za-z0-9\s]', title))
    punctuation_count = len(re.findall(r'[^\w\s]', title))
    sentiment = TextBlob(title).sentiment
    sentiment_polarity = sentiment.polarity
    sentiment_subjectivity = sentiment.subjectivity

    features = {
        'word_count': word_count,
        'character_count': character_count,
        'average_word_length': average_word_length,
        'stop_word_count': stop_word_count,
        'capital_letters_count': capital_letters_count,
        'digits_count': digits_count,
        'special_characters_count': special_characters_count,
        'punctuation_count': punctuation_count,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity
    }

    return features


# Named Entity Recognition counts
nlp = spacy.load('en_core_web_sm')

def get_named_entities(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]
# Part-of-Speech tag counts
def get_pos_tags(text):
    doc = nlp(text)
    return [token.pos_ for token in doc]