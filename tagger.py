import re

import numpy as np
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from metrics import roc_auc

# nltk.download('reuters')
# nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
PUNCTUATION_RE = re.compile('[\W]')
SPECIAL_CHARACTERS = re.compile('[!@$%^&*()"<>]');
STOPWORDS = set(stopwords.words('english'))
MODEL_NAMES = ['Logit', 'LinearSVC', 'RandomForest']

labels = reuters.categories()
n_classes = len(labels)

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    # text = re.sub(PUNCTUATION_RE,'', text)
    # text = re.sub(SPECIAL_CHARACTERS, '', text)
    text = re.sub(REPLACE_BY_SPACE_RE, '\s', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    word_list = text.split()
    # delete stopwords from text
    filtered_words = [word for word in word_list if word not in STOPWORDS]
    text = ' '.join(filtered_words)
    return text


def load_data():
    """
    Load the Reuters dataset.

    Returns
    -------
    train_docs, train_labels, test_docs, test_labels.
    """
    documents = reuters.fileids()
    train = [d for d in documents if d.startswith('training/')]
    train_docs = [reuters.raw(doc_id) for doc_id in train]
    train_docs = [text_prepare(x) for x in train_docs]
    train_labels = [reuters.categories(doc_id) for doc_id in train]

    test = [d for d in documents if d.startswith('test/')]
    test_docs = [reuters.raw(doc_id) for doc_id in test]
    test_docs = [text_prepare(x) for x in test_docs]
    test_labels = [reuters.categories(doc_id) for doc_id in test]

    print("len(train_docs)={}, len(train_labels)={}".format(len(train_docs), len(train_labels)))
    print("len(test_docs)={}, len(test_labels)={}".format(len(test_docs), len(test_labels)))

    mlb = MultiLabelBinarizer(classes=sorted(labels))
    train_labels = mlb.fit_transform(train_labels)
    test_labels = mlb.fit_transform(test_labels)
    print("y_train.shape={}, y_test.shape={}".format(train_labels.shape, test_labels.shape))

    return (train_docs, train_labels, test_docs, test_labels, mlb.classes)


def tfidf_features(train_docs, test_docs):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    #tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=1, max_df=0.8, ngram_range=(1, 2))
    #tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', analyzer='word', min_df=5, max_df=0.9, ngram_range=(1, 3))
    #tfidf_vectorizer = TfidfVectorizer( min_df=1, max_df=0.8, ngram_range=(1, 3))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(train_docs)
    vectorised_train_documents = tfidf_vectorizer.transform(train_docs)
    vectorised_test_documents = tfidf_vectorizer.transform(test_docs)
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vectorizer.vocabulary_.items()}
    return vectorised_train_documents, vectorised_test_documents, tfidf_reversed_vocab

def train_classifier(model_name, X_train, y_train, penalty_param, C_param):
    """
      X_train, y_train — training data

      return: trained classifier
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    if model_name == 'Logit':
        print("\nLogisticRegression")
        model = OneVsRestClassifier(LogisticRegression(penalty=penalty_param, C=C_param)).fit(X_train, y_train)
    elif  model_name == 'LinearSVC':
        print("\nLinearSVC")
        model = OneVsRestClassifier(LinearSVC(penalty=penalty_param, C=C_param)).fit(X_train, y_train)
    elif model_name == 'RandomForest':
        print("\nRandomForest")
        model = RandomForestClassifier(n_estimators=n_classes).fit(X_train, y_train)
    return model


def print_evaluation_scores(y_val, predicted):
    acc_score = accuracy_score(y_val, predicted, normalize=False)
    print('accuracy_score= ', acc_score)

    f1_macro = f1_score(y_val, predicted, average='macro')
    print('f1_macro= ', f1_macro)
    f1_micro = f1_score(y_val, predicted, average='micro')
    print('f1_micro= ', f1_micro)
    f1_weighted = f1_score(y_val, predicted, average='weighted')
    print('f1_weighted= ', f1_weighted)

    precision_macro = average_precision_score(y_val, predicted, average='macro')
    print('precision_macro= ', precision_macro)
    precision_micro = average_precision_score(y_val, predicted, average='micro')
    print('precision_micro= ', precision_micro)
    precision_weighted = average_precision_score(y_val, predicted, average='weighted')
    print('precision_weighted= ', precision_weighted)

    recall_macro = recall_score(y_val, predicted, average='macro')
    print('recall_macro= ', recall_macro)
    recall_micro = recall_score(y_val, predicted, average='micro')
    print('recall_micro= ', recall_micro)
    recall_weighted = recall_score(y_val, predicted, average='weighted')
    print('recall_weighted= ', recall_weighted)

    auc_macro = roc_auc_score(y_val, predicted, average='macro')
    print('auc_macro= ', auc_macro)
    auc_micro = roc_auc_score(y_val, predicted, average='micro')
    print('auc_micro= ', auc_micro)
    auc_weighted = roc_auc_score(y_val, predicted, average='weighted')
    print('auc_weighted= ', auc_weighted)


def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
    used_cls = classifier.estimators_[tags_classes.index(tag)]
    coeff = np.squeeze(used_cls.coef_)
    sorted_ind = sorted(range(len(coeff)), key=lambda x: coeff[x])
    top_positive_words = [index_to_words[ind] for ind in sorted_ind[-5:]]
    top_negative_words = [index_to_words[ind] for ind in sorted_ind[:5]]

    print('\nTag:\t{}'.format(tag))
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


if __name__ == "__main__":
    train_docs, train_labels, test_docs, test_labels, mlb_classes =  load_data()
    vectorised_train_documents, vectorised_test_documents,tfidf_reversed_vocab = tfidf_features(train_docs, test_docs)
    predicted_scores = None
    best_model = None
    for model_name in MODEL_NAMES:
        model = train_classifier(model_name, vectorised_train_documents, train_labels, 'l2', 10)
        predicted_labels = model.predict(vectorised_test_documents)

        if model_name == 'LinearSVC':
            best_model = model
            predicted_scores = model.decision_function(vectorised_test_documents)

        print(model_name + ' scores')
        print_evaluation_scores(test_labels, predicted_labels)

    roc_auc(test_labels, predicted_scores, n_classes)
    print_words_for_tag(best_model, 'cotton-oil', mlb_classes, tfidf_reversed_vocab)



