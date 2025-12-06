import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import time
import re

data = '/uss/hdsi-prismdata/q1-ucsd-outflows.pqt' if os.path.exists('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt') \
    else 'data/outflows.pqt'

df = pd.read_parquet(data)[['memo', 'category']]
df = df[df['memo'] != df['category']]

def preprocess_memo(text):
    text = text.upper()
    text = re.sub(r'\d+', 'N', text)  # Mask numbers
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    return text

vectorizer = CountVectorizer(
    preprocessor=preprocess_memo,
    ngram_range=(2, 3),
    stop_words=None
)

print("Analyzing ngrams...")
start_time = time.time()
X = vectorizer.fit_transform(df['memo'])

counts = X.sum(axis=0).A1
feature_names = vectorizer.get_feature_names_out()

freq_distribution = pd.DataFrame({
    'pattern': feature_names,
    'count': counts
}).sort_values(by='count', ascending=False)

end_time = time.time()
print(f"Completed in {end_time - start_time:.2f} seconds.")

print(freq_distribution.head(20).reset_index(drop=True))
print("Saving to data/ngrams.csv...")
freq_distribution.to_csv('data/ngrams.csv', index=False)

