from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from collections import Counter
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from collections import Counter
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

# Ensure NLTK resources are available
import nltk
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('train.csv')

def preprocess_text(text):
        
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    #text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = text.lower()
    tokens = word_tokenize(text)

     # Remove all punctuation except "!" and "?"
    #text = re.sub(r'[^\w\s!?]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    #tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Encode the sentiment labels
label_encoder = LabelEncoder()
data['sentiment_encoded'] = label_encoder.fit_transform(data['sentiment'])

# Plot label distribution
def plot_label_distribution(data):
    plt.figure(figsize=(8, 5))
    data['sentiment'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Label Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

plot_label_distribution(data)

# Plot the most used words for each label
def plot_most_common_words(data, label, n=20):
    words = ' '.join(data[data['sentiment'] == label]['cleaned_text']).split()
    word_freq = Counter(words).most_common(n)
    words, counts = zip(*word_freq)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(f'Most Common Words for {label} Sentiment')
    plt.xticks(rotation=45)
    plt.show()

plot_most_common_words(data, 'positive')
plot_most_common_words(data, 'neutral')
plot_most_common_words(data, 'negative')


# Define parameter grids for vectorizers
tfidf_param_grid = {
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.8, 0.9, 1.0],
    "tfidf__max_features": [500, 1000, 2000, 'None'],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    
}

bow_param_grid = {
    "bow__min_df": [1, 2, 5],
    "bow__max_df": [0.8, 0.9, 1.0],
    "bow__max_features": [500, 1000, 2000,'None'],
    "bow__ngram_range": [(1, 1), (1, 2), (1, 3)]
}

# Combine vectorizer with classifier in a pipeline
pipelines = {
    "TF-IDF + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "BoW + Logistic Regression": Pipeline([
        ("bow", CountVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "TF-IDF + SVM": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", SVC())
    ]),
    "BoW + SVM": Pipeline([
        ("bow", CountVectorizer()),
        ("clf", SVC())
    ]),
    "TF-IDF + Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier())
    ]),
    "BoW + Random Forest": Pipeline([
        ("bow", CountVectorizer()),
        ("clf", RandomForestClassifier())
    ]),
    "TF-IDF + Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ]),
    "BoW + Naive Bayes": Pipeline([
        ("bow", CountVectorizer()),
        ("clf", MultinomialNB())
    ])
}

# Define parameter grids for classifiers
svm_param_grid = {
    "clf__C": [0.1, 1, 10],  # Regularization strengthn
    "clf__kernel": ["linear", "rbf"]  #  kernel type 
}

lr_param_grid = {
    "clf__C": [0.1, 1, 10],  # Inverse of regularization strength
    "clf__solver": ["liblinear", "rbf"] # Optimization algorithm
}

rf_param_grid = {
    "clf__n_estimators": [50, 100, 200],  # Number of trees in the forest
    "clf__max_depth": [None, 10, 20, 30]  # Maximum depth of the tree
}

nb_param_grid = {
    "clf__alpha": [0.1, 0.5, 1.0, 5.0, 10.0]  # Additive smoothing parameter 
}


### FEATURES 
def unique_word_ratio(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def average_word_length(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    return sum(len(word) for word in tokens) / len(tokens)

def punctuation_count(text, punctuation_mark):
    return text.count(punctuation_mark)

# Apply new features
data['unique_word_ratio'] = data['cleaned_text'].apply(unique_word_ratio)
data['average_word_length'] = data['cleaned_text'].apply(average_word_length)
data['exclamation_count'] = data['text'].apply(lambda x: punctuation_count(x, '!'))
#data['question_count'] = data['text'].apply(lambda x: punctuation_count(x, '?'))
#FEATURE

# Apply preprocessing
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

def count_pos_tags(text, pos_tags):
    """
    Counts occurrences of specific POS tags in a given text.
    Args:
        text (str): The text to analyze.
        pos_tags (set): A set of POS tags to count (e.g., {"NN", "JJ"}).
    Returns:
        int: The count of specified POS tags in the text.
    """
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    return sum(1 for word, tag in tagged_words if tag in pos_tags)



# Define POS groups
noun_tags = {"NN", "NNS", "NNP", "NNPS"}  # Nouns
adjective_tags = {"JJ", "JJR", "JJS"}  # Adjectives
adverb_tags = {"RB", "RBR", "RBS"}  # Adverbs

# Add POS features
data['noun_count'] = data['cleaned_text'].apply(lambda x: count_pos_tags(x, noun_tags))
data['adjective_count'] = data['cleaned_text'].apply(lambda x: count_pos_tags(x, adjective_tags))
data['adverb_count'] = data['cleaned_text'].apply(lambda x: count_pos_tags(x, adverb_tags))
#FEATURES

# Define positive and negative word lists
positive_words = {"good", "great", "excellent", "amazing", "wonderful", "love", "positive", "happy", "fantastic", "awesome"}
negative_words = {"bad", "worse", "worst", "terrible", "awful", "hate", "negative", "sad", "poor", "disappointing"}

# Define negation words
negation_words = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "isn't", "aren't", "wasn't", "weren't", "shouldn't", "mightn't", "mustn't"}

def count_sentiment_words(text, word_list):
    tokens = text.split()
    return sum(1 for word in tokens if word in word_list)

# Add custom features
data['positive_word_count'] = data['cleaned_text'].apply(lambda x: count_sentiment_words(x, positive_words))
data['negative_word_count'] = data['cleaned_text'].apply(lambda x: count_sentiment_words(x, negative_words))
data['negation_count'] = data['cleaned_text'].apply(lambda x: count_sentiment_words(x, negation_words))
data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))


y = data['sentiment_encoded']
custom_features = data[[
                        'noun_count', 'adjective_count', 'positive_word_count', 'negative_word_count', 'negation_count', 
                        'text_length', 'unique_word_ratio', 'average_word_length', 
                        'exclamation_count',  'adverb_count']].values
'''
# THIS IS WHERE I PUT THE FEATURES I DONT WANT TO TRY 
'''

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

method_results = {}
for pipeline_name, pipeline in pipelines.items():
    print(f"Processing {pipeline_name}")
    
    # Select the appropriate vectorizer
    if "TF-IDF" in pipeline_name:
        vectorizer = TfidfVectorizer()
        vec = vectorizer.fit_transform(data['cleaned_text']).toarray()
    else:  # For "BoW"
        vectorizer = CountVectorizer()
        vec = vectorizer.fit_transform(data['cleaned_text']).toarray()

    # Combine text vectorizer output with custom features
    X = np.hstack([vec, custom_features])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Define the parameter grid
    if "SVM" in pipeline_name:
        classifier_params = svm_param_grid
    elif "Logistic Regression" in pipeline_name:
        classifier_params = lr_param_grid
    elif "Random Forest" in pipeline_name:
        classifier_params = rf_param_grid
    elif "Naive Bayes" in pipeline_name:
        classifier_params = nb_param_grid
    else:
        classifier_params = {}

    vectorizer_params = tfidf_param_grid if "TF-IDF" in pipeline_name else bow_param_grid
    full_param_grid = {**vectorizer_params, **classifier_params}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, full_param_grid, cv= skf, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Collect results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    y_pred = best_model.predict(X_test)
    test_accuracy = best_model.score(X_test, y_test)

    print(f"Results for {pipeline_name}")
    print(f"Best Params: {best_params}")
    print(f"Best Cross-Val Accuracy: {best_score:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    method_results[pipeline_name] = {
        "Best Params": best_params,
        "Cross-Val Accuracy": best_score,
        "Test Accuracy": test_accuracy
    }


results = pd.DataFrame(method_results).T

# Display results
print("Final Results:")
print(results)