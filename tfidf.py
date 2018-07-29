from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
texts = ["good movie", "not a good movie", "did not like", "i like it", "good one"]
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
features = tfidf.fit_transform(texts)
df = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
print(df.head())