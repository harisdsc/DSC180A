import regex as rules
import pandas as pd
import json

def _apply_regex_post(memo):
    for pattern in rules.REGEX_POST:
        match = pattern.match(memo)
        if match:
            groups = match.groups()
            if groups:
                return groups[-1].strip()
    return memo

def clean_memo(config):
    # Load Configuration
    print("Starting preprocessing...")
    with open(config) as f:
        config = json.load(f)

    input_file = config['input_memos']
    output_file = config['output_memos']

    # Load Data
    print(f"Loading data from {input_file.split('/')[-1]}...")
    df = pd.read_csv(input_file)

    # Pass 1
    print("Starting Pass 1...")
    memos = df['memo'].astype(str).fillna('').str.upper()
    memos = memos.str.replace(r"\\u00A0", " ", regex=True)
    memos = memos.str.replace(r"\\s{2,}", " ", regex=True)
    memos = memos.str.strip()

    for pattern, repl in rules.REGEX_PRE:
        memos = memos.str.replace(pattern, repl, regex=True)

    memos = memos.str.replace(rules.NOISE_WORDS_REGEX, " ", regex=True)
    memos = memos.str.replace(r"\\s{2,}", " ", regex=True)
    memos = memos.str.replace(r"^[\\s-]+|[\\s-]+$", "", regex=True)
    df['memo_pre'] = memos
    print("Pass 1 complete.")

    # Pass 2
    print("Starting Pass 2...")
    df['memo_post'] = df['memo_pre'].apply(_apply_regex_post)
    print("Pass 2 complete.")

    # Final Cleaning
    df['memo_post'] = df['memo_post'].str.replace('-', '')
    df['memo_post'] = df['memo_post'].str.replace("'", '')
    df['memo_post'] = df['memo_post'].str.replace(':', '')
    df['memo_post'] = df['memo_post'].str.replace('.', '')
    df['memo_post'] = df['memo_post'].str.strip()

    # Save Output
    print(f"Saving cleaned data to {output_file.split('/')[-1]}...")
    df.to_csv(output_file, index=False)

    return df

if __name__ == '__main__':
    config_path = 'configs/preprocessing/clean.json'
    clean_memo(config_path)