import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from scraper import get_text_from_url
from feature_extraction import extract_title_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data_path = r'/Users/matanb/Downloads/DV_NLP_assignment/assignment_data_en.csv'

df = pd.read_csv(data_path)

# add some fetures
df['domain'] = df['url'].apply(lambda x: x.split('/')[2])
df['domain'] = df['domain'].apply(lambda x: x.split('.')[-2])
df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')
df['title_len'] = df['scraped_title'].apply(lambda x: len(x))
df['av_word_len'] = df['scraped_title'].apply(lambda x: sum([len(word) for word in x.split()]) / len(x.split()))
df['num_words'] = df['scraped_title'].apply(lambda x: len(x.split()))
df['sub'] = df['scraped_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['pol'] =  df['scraped_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['text_len'] = df['scraped_text'].apply(lambda x: len(x))

for idx, row in df.iterrows():
    features = extract_title_features(row['scraped_text'])
    for key, value in features.items():
        df.loc[idx, key] = value


# present some eda
df['domain'].value_counts()
df['content_type'].value_counts()
df['scraped_title'].str.len().hist()
# hist of title lenght colored by content type

plt.Figure()
sns.histplot(df, x='title_len', hue='content_type', kde=True)
# now try news vs non-news
plt.Figure()
sns.histplot(df, x='title_len', hue='class', kde=True)
# now try news vs non-news
plt.Figure()
sns.histplot(df, x='av_word_len', hue='class', kde=True)
# now try news vs non-news
plt.Figure()
sns.histplot(df, x='num_words', hue='class', kde=True)

# sentiment analysis
plt.Figure()
sns.histplot(df, x='polarity', hue='class', kde=True)

# 2d hist of polarity vs subjectivity
plt.Figure()
sns.histplot(df, x='polarity', y='subjectivity', hue='class', kde=True)

# capital letter count
plt.Figure()
sns.histplot(df, x='punctuation_count', hue='class', kde=True)

# stop word count
plt.Figure()
sns.histplot(df, x='stop_word_count', hue='class', kde=True)

sns.histplot(df,x = 'stop_word_count',y = 'pol', hue = 'content_type')


# feature correlation
plt.Figure()
numeric_features = ['title_len', 'av_word_len', 'num_words',  'stop_word_count', 'capital_letters_count', 'digits_count', 'special_characters_count', 'punctuation_count']
sns.heatmap(df[numeric_features].corr(), annot=True)


# tfidf title and url. than pca or tsne
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(df['scraped_title'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# add tfidf of the url
url_vectorizer = CountVectorizer()
url_counts = url_vectorizer.fit_transform(df['url'])
url_tfidf = TfidfTransformer()
url_tfidf = url_tfidf.fit_transform(url_counts)
X_train_tfidf =  pd.DataFrame(X_train_tfidf.toarray())
url_tfidf = pd.DataFrame(url_tfidf.toarray())
X_train_tfidf = pd.concat([X_train_tfidf, url_tfidf], axis=1)
X_train_tfidf['class'] = df['class']

# Step 1: Import the necessary library
from sklearn.manifold import TSNE

# Step 2: Fit the t-SNE model to your high-dimensional data
tsne = TSNE(n_components=3)
tsne_result = tsne.fit_transform(X_train_tfidf.drop('class', axis=1))

# Step 3: Plot the results using a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=df['class'].map({'news': 0, 'non-news': 1}), cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
plt.title('t-SNE projection of the dataset', fontsize=24)
plt.show()