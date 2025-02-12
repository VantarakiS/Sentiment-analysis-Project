#THE SCRIPT IS READY JUST RUN IT. IT ASSUMES THAT THE FILES ARE NAMED train.csv and test.csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import  PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder

# Ensure NLTK resources are available
import nltk
nltk.download('stopwords')
def preprocess_text(text):
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english')) 
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)



# Load the training dataset
train_data = pd.read_csv('train.csv')
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
train_data['sentiment_encoded'] = label_encoder.fit_transform(train_data['sentiment'])

# Vectorize text data using TF-IDF (ngrams = (1, 2))
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(train_data['cleaned_text'])
y_train = train_data['sentiment_encoded']

# Train Logistic Regression model with specified parameters
logistic_regression = LogisticRegression(C=10, solver='liblinear', max_iter=1000)
logistic_regression.fit(X_train, y_train)

# Load the test dataset
test_data = pd.read_csv('test.csv')
test_data = test_data[test_data['text'].apply(lambda x: isinstance(x, str))]
test_data['cleaned_text'] = test_data['text'].apply(preprocess_text)

# Vectorize the test dataset
X_test = tfidf_vectorizer.transform(test_data['cleaned_text'])

# Predict sentiment for the test dataset
test_data['predicted_sentiment'] = logistic_regression.predict(X_test)
test_data['predicted_sentiment_label'] = label_encoder.inverse_transform(test_data['predicted_sentiment'])

# Export the predicted sentiments to a text file
test_data['predicted_sentiment_label'].to_csv('predictions.txt', index=False, header=False)

print("Sentiment classification complete. Results saved to 'predictions.txt'.")

