import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__== "__main__":
    text = "feets cats wolves talked"
    print(text)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=5, max_df=0.9, ngram_range=(1, 2))
    tfidf_vectorizer.fit([text])
    tfidf_vectorizer.transform([text])
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    print(tfidf_vocab)
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
    print(tfidf_reversed_vocab)
    # tokenizer = nltk.tokenize.TreebankWordTokenizer()
    # tokens = tokenizer.tokenize(text)
    # stemmer = nltk.stem.PorterStemmer()
    # " ".join(stemmer.stem(token) for token in tokens)
    # stemmer = nltk.stem.WordNetLemmatizer()
    # l= " ".join(stemmer.lemmatize(token) for token in tokens)
    # print(l)


