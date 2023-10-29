# # get training data
# def get_train_data(datapath=DATA_PATH):
#     train_data = pd.read_csv(datapath)
#     X = train_data['posts']
#     y = train_data['type']
#     return X, y
import numpy as np
# split data into train and test
# form dataframe
def split_data(dataframe, size=0.2):
    train_data = dataframe.sample(frac=1-size, random_state=0)
    test_data = dataframe.drop(train_data.index)
    return train_data, test_data

# # applying tf-idf vectorizer
# def tfidf_vectorizer(X_train, X_test):
#     vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words='english')
#     X_train = vectorizer.fit_transform(X_train)
#     X_test = vectorizer.transform(X_test)
#     return X_train, X_test

# encode labels
def encode_labels(y):
    ie_encoding = {"I":1 , "E":-1}
    ns_encoding = {"N":1 , "S":-1}
    tf_encoding = {"T":1 , "F":-1}
    pj_encoding = {"P":1 , "J":-1}
    y_encoded = np.zeros((len(y), 4))
    for i, _y in enumerate(y):
        y_encoded[i][0] = ie_encoding[_y[0]]
        y_encoded[i][1] = ns_encoding[_y[1]]
        y_encoded[i][2] = tf_encoding[_y[2]]
        y_encoded[i][3] = pj_encoding[_y[3]]
    
    return y_encoded 

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
# clean the data
# remove urls, and punctuations
# remove stop words
# remove numbers
# remove MBTI types
def clean_data(X_train):
    ps = PorterStemmer()
    X_train_cleaned = []
    for i in range(len(X_train)):
        # remove urls
        X_train[i] = re.sub(r'http\S+', '', X_train[i])
        # remove punctuations
        X_train[i] = X_train[i].translate(str.maketrans('', '', string.punctuation))
        # remove stop words
        # X_train[i] = ' '.join([word for word in X_train[i].split() if word not in stopwords.words('english')]) # remove numbers
        X_train[i] = ''.join([i for i in X_train[i] if not i.isdigit()])
        # stemming
        # X_train[i] = ' '.join([ps.stem(word) for word in X_train[i].split()])
        # remove MBTI types
        mbti_types = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp','isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
        for mbti_type in mbti_types:
            X_train[i] = X_train[i].replace(mbti_type, '')
        # also upper case
        for mbti_type in mbti_types:
            X_train[i] = X_train[i].replace(mbti_type.upper(), '')
        X_train_cleaned.append(X_train[i])
    return X_train_cleaned

def get_average_length_and_emotion(X_train):
    from textblob import TextBlob
    average_word_counts = []
    average_word_length = []
    average_emo_score = []
    for i in range(len(X_train)):
        posts = X_train[i].split('|||')
        total_word_counts = 0
        total_char_counts = 0
        emo_score = 0
        for post in posts:
            blob = TextBlob(post)
            emo_score += blob.sentiment.polarity
            words = post.split()
            total_word_counts += len(words)
            for word in words:
                total_char_counts += len(word)
        average_word_counts.append(total_word_counts / len(posts))
        average_word_length.append(total_char_counts / total_word_counts)
        average_emo_score.append(emo_score / len(posts))
    return average_word_counts, average_word_length, average_emo_score

# balance the data by its label using SMOTE
# make sure each label has the same number of samples
def balance_data(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    sm = SMOTE(random_state=42)
    print('Original dataset shape %s' % Counter(y_train))
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_train))
    return X_train, y_train

def train_model_balence(X_train, y_train, X_test, y_test, models):
    from sklearn.metrics import accuracy_score, classification_report
    for i,model in enumerate(models):
        X_train_balenced, y_train_balenced = balance_data(X_train, y_train[:,i])
        model.fit(X_train_balenced, y_train_balenced)
        y_pred = model.predict(X_test)
        print("Accuracy for dimension {} is {}".format(i, accuracy_score(y_test[:,i], y_pred)))
        print(classification_report(y_test[:,i], y_pred))
        print("\n")