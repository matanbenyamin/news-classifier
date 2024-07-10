
# EDA
 A link to a Colab norebook with the EDA
https://colab.research.google.com/drive/1HUNt80vmZUOKx6ezoGfUmPV-dkHKhA0K?usp=sharing
for easier running and analysis. In order to run it, upload the data file to the colab environment.


### Basic EDA
- missing values
- duplicates, imbalance

### Advanced EDA
- Embeddings
- sentiment analysis
- multi class vs binary, N-grams
- Topic Modeling 
- Correlation Analysis

 ### The purpose in this phase is to identify candidate informative features, correlated features that will be redundant in training, general behaviour of the data.



##  EDA Conculsions
most are sumarized in the eda itself. In general I hae learned titles and even urls are very informative, 
and may suffcie for pretty good classification without the need to proceess large amounts of text.


# Training Session and model selection
I have tried several classifiers, and used k-fold ross valirfatin and hyper parameter grid search 
I Also played with more sophiticated feature extraction - word2vec and sentence embeddings using pre trained models (DistillBERT). There was a slight improvement in the model performance, but the model was too slow to train and evaluate. I decided to stick with the simpler model for now.


Results on the test set:

              precision    recall  f1-score   support

        news       0.82      0.83      0.83       590
    non-news       0.81      0.80      0.81       528

    accuracy                           0.82      1118


### After training I have found significiant overfitting, probably caused by the large amount of features extracted from the tf-idf.

### second attempt included reducing the maount of features (terms)
## Error Analysis
### feature importance
### content type of miss detections
### features of miss detections
