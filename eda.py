import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from feature_extraction import extract_features, clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ===============================================================================
data_path = r'./assignment_data_en.csv'
df = pd.read_csv(data_path)

# save a copy without text and domain
df.drop(['content_type'], axis=1).to_excel('cleaned_data.xlsx', index=False)

# ====================  add some fetures
df['clean_text'] = df['scraped_text'].apply(clean_text)
df['clean_title'] = df['scraped_title'].apply(clean_text)
df['domain'] = df['url'].apply(lambda x: x.split('/')[2])
df['domain'] = df['domain'].apply(lambda x: x.split('.')[-2])
df['class'] = df['content_type'].apply(lambda x: 'news' if x == 'news' else 'non-news')

# add features
# look for file
file_found = False
filename = 'extracted_features.csv'
if os.path.exists(filename):
    df = pd.read_csv(filename)
    file_found = True
if not file_found:
    for idx, row in tqdm.tqdm(df.iterrows()):
        title_features = extract_features(row['scraped_text'])
        for key, value in title_features.items():
            key = 'title_' + key
            df.loc[idx, key] = value
        text_features = extract_features(row['scraped_text'])
        for key, value in text_features.items():
            key = 'text_' + key
            df.loc[idx, key] = value
    # save extracted features
    df.to_csv('extracted_features.csv', index=False)


# add sentiment features
df['title_sentiment_polarity'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['title_sentiment_subjectivity'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['text_sentiment_polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['text_sentiment_subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


#  over all EDA
# check for missing values
nulls = df.isnull().sum()
print('Missing values:' + '\n' + str(nulls))
# check for duplicates
print('Duplicates:' + '\n' + str(df.duplicated().sum()))
# check for class imbalance
df['class'].value_counts()

# =============           hist of class imbalance - 2 subplots. one new vs non-news and one for content type
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(df, x='class', hue='class', kde=True, ax=axs[0])
sns.histplot(df, x='content_type', hue='content_type', kde=True, ax=axs[1])
plt.title('Class Imbalance')
plt.show()

# ==============================    subplots with histograms of features
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, feature in enumerate(df.columns[7:]):
    sns.histplot(df, x=feature, hue='class', kde=True, ax=axs[i // 3, i % 3])

# feature correlation
plt.Figure()
dtypes = df.dtypes
numeric_features = dtypes[dtypes != 'object'].index
sns.heatmap(df[numeric_features].corr(), annot=True)
plt.title('Feature Correlation')
plt.show()
# print feature pairs with correlation > 0.5
correlation_matrix = df[numeric_features].corr().abs()
correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
correlation_pairs = correlation_pairs[correlation_pairs < 1]
correlation_pairs = correlation_pairs[correlation_pairs > 0.35]
print(correlation_pairs)
# ==============================
# Higher correlation receved for various word counts, so not very useful


# ======================================= N-grams analysis
# Function to extract n-grams
def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Top 20 bi-grams and tri-grams
top_bigrams = get_top_ngrams(df['clean_title'], n=20, ngram_range=(2,2))
top_trigrams = get_top_ngrams(df['clean_title'], n=20, ngram_range=(3,3))

# show top in each class
top_bigrams_news = get_top_ngrams(df[df['class'] == 'news']['clean_title'], n=20, ngram_range=(2,2))
top_bigrams_non_news = get_top_ngrams(df[df['class'] == 'non-news']['clean_title'], n=20, ngram_range=(2,2))
print(top_bigrams_news)
print(top_bigrams_non_news)



from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix_news = tfidf_vectorizer.fit_transform(df[df['class'] == 'news']['scraped_title'])
tfidf_matrix_non_news = tfidf_vectorizer.fit_transform(df[df['class'] == 'non-news']['scraped_title'])
feature_names = tfidf_vectorizer.get_feature_names_out()
news_feature_names = [feature_names[i] for i in tfidf_matrix_news.toarray().sum(axis=0).argsort()[::-1][:10]]
non_news_feature_names = [feature_names[i] for i in tfidf_matrix_non_news.toarray().sum(axis=0).argsort()[::-1][:10]]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(x=news_feature_names, y=tfidf_matrix_news.toarray().sum(axis=0).argsort()[::-1][:10], ax=axs[0])
axs[0].set_title('Top 10 Features for News')
sns.barplot(x=non_news_feature_names, y=tfidf_matrix_non_news.toarray().sum(axis=0).argsort()[::-1][:10], ax=axs[1])
axs[1].set_title('Top 10 Features for Non-News')
plt.show()



# ================ Vusialuzation of the top 50 features with a word2vec model
tfidf_df_news = pd.DataFrame(tfidf_matrix_news.toarray(), columns=feature_names)
tfidf_df_non_news = pd.DataFrame(tfidf_matrix_non_news.toarray(), columns=feature_names)

mean_tfidf_news = tfidf_df_news.mean()
mean_tfidf_non_news = tfidf_df_non_news.mean()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gensim.downloader as api

# Load pre-trained word2vec model
word2vec_model = api.load('word2vec-google-news-300')

# Get word embeddings for the most frequent words
words = mean_tfidf_news[:50].index
words_non_news = mean_tfidf_non_news[:50].index
word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
word_vectors_non_news = [word2vec_model[word] for word in words_non_news if word in word2vec_model]

# Perform PCA to reduce dimensions
pca = PCA(n_components=2)
pca_news = pca.fit_transform(word_vectors)
pca_non_news = pca.fit_transform(word_vectors_non_news)

word_vectors_pca = pca.fit_transform(word_vectors)
word_vectors_pca_non_news = pca.fit_transform(word_vectors_non_news)
# Create a DataFrame for plotting
pca_df = pd.DataFrame(word_vectors_pca,  columns=['x', 'y'])
pca_df_non_news = pd.DataFrame(word_vectors_pca_non_news,  columns=['x', 'y'])
# Plot word embeddings
plt.figure(figsize=(10, 10))
plt.scatter(pca_df['x'], pca_df['y'], c='blue', label='News')
plt.scatter(pca_df_non_news['x'], pca_df_non_news['y'], c='red')
plt.title('Word Embeddings Visualization')
plt.show()






for word, pos in pca_df.iterrows():
    plt.annotate(word, pos, fontsize=12)

plt.title('Word Embeddings Visualization')
plt.show()

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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gensim.downloader as api

mean_tfidf
# Load pre-trained word2vec model
word2vec_model = api.load('word2vec-google-news-300')

# Get word embeddings for the most frequent words
words = mean_tfidf[:50].index
word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]

# Perform PCA to reduce dimensions
pca = PCA(n_components=2)
word_vectors_pca = pca.fit_transform(word_vectors)

# Create a DataFrame for plotting
pca_df = pd.DataFrame(word_vectors_pca, index=words, columns=['x', 'y'])

# Plot word embeddings
plt.figure(figsize=(10, 10))
plt.scatter(pca_df['x'], pca_df['y'])

for word, pos in pca_df.iterrows():
    plt.annotate(word, pos, fontsize=12)

plt.title('Word Embeddings Visualization')
plt.show()

# ===============================================================================
# a bit more sophisticated - user BERT to embed
# ===============================================================================
# import the necessary libraries
from sklearn.preprocessing import LabelEncoder



LE = LabelEncoder()
df['label'] = LE.fit_transform(df['scraped_text'])
df.head()



import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

tokenized_train = tokenizer(df["scraped_text"].values.tolist(), padding = True, truncation = True, return_tensors="pt")
print(tokenized_train.keys())

#move on device (GPU)
tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}

with torch.no_grad():
  hidden_train = model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]
#get only the [CLS] hidden states
cls_train = hidden_train.last_hidden_state[:,0,:]

from transformers import AlbertTokenizer, AlbertModel

# Load pre-trained ALBERT model and tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')
# Function to extract ALBERT embeddings
def get_albert_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

# Extract ALBERT embeddings
albert_embeddings = get_albert_embeddings(df['scraped_text'].values)
albert_embeddings.shape


from transformers import DistilBertTokenizer, DistilBertModel

# Load pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to extract DistilBERT embeddings
def get_distilbert_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

# Extract DistilBERT embeddings
distilbert_embeddings = get_distilbert_embeddings(df['scraped_text'].values)
distilbert_embeddings_title = get_distilbert_embeddings(df['scraped_title'].values)
distilbert_embeddings_url = get_distilbert_embeddings(df['url'].values)

# save the embeddings
np.save('distilbert_embeddings.npy', distilbert_embeddings)

# try to train
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# xgboost
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(distilbert_embeddings_url, df['class'], test_size=0.33, random_state=42)

# xgboost
clf = XGBClassifier().fit(X_train, y_train)
predicted = clf.predict(X_test)
accuracy_score(y_test, predicted)



# ===== Text Cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Fit KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix_news)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Visualize clusters with PCA
pca = PCA(n_components=2)
tfidf_pca = pca.fit_transform(tfidf_matrix_news.toarray())

plt.figure(figsize=(10, 5))
sns.scatterplot(x=tfidf_pca[:, 0], y=tfidf_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Text Clustering with KMeans')
plt.show()


# show the features umap
import umap

    umap_embeddings = umap.UMAP(n_neighbors=5, n_components=3, metric='cosine').fit_transform(distilbert_embeddings_url)
    plt.figure(figsize=(10, 10))
    # scatter 3d
    ax = plt.axes(projection='3d')
    ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], c=df['class'].map({'news': 0, 'non-news': 1}), cmap='Spectral', s=5)
    plt.show()
