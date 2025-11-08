import sys
import json
import pandas as pd
from src.preprocessing.clean import clean_memo
from src.preprocessing.ngrams import top_ngrams

if __name__ == '__main__':
    args = sys.argv[1:]
    