import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def top_ngrams(corpus: pd.Series, n_gram_range: tuple, top_n: int = 200):
    print(f"Analyzing {n_gram_range} n-grams...")
    vec = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words='english',
        max_features=None  # Count all n-grams first
    ).fit(corpus.astype(str))

    # Get the counts
    bag_of_words = vec.transform(corpus.astype(str))

    # Sum the counts for each n-gram
    sum_words = bag_of_words.sum(axis=0)

    # Map n-grams to their frequencies
    words_freq = [
        (word, sum_words[0, idx])
        for word, idx in vec.vocabulary_.items()
    ]

    # Sort by frequency (descending)
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]

