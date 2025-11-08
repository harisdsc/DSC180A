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

def clean_memo(config_path):
    # Load Configuration
    print("Starting preprocessing...")
    with open(config_path) as f:
        config = json.load(f)

    input_file = config['input_memos']
    output_file = config['output_memos']

    # Load Data
    print(f"Loading data from {input_file}...")
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

    # Final Cleaning
    df['memo_post'] = df['memo_post'].str.replace('-', '')
    df['memo_post'] = df['memo_post'].str.replace("'", '')
    df['memo_post'] = df['memo_post'].str.replace(':', '')
    df['memo_post'] = df['memo_post'].str.replace('.', '')
    df['memo_post'] = df['memo_post'].str.strip()

    mask = df['memo_post'].str.contains("*", regex=False)
    df.loc[mask, 'memo_post'] = df.loc[mask, 'memo_post'].str.split('*').str[1::].str.join('')

    mask = df['memo_post'].str.contains('#', regex=False)
    df.loc[mask, 'memo_post'] = df.loc[mask, 'memo_post'].str.split('#').str[0]

    # Save Output
    print(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)

    print("Preprocessing complete.")
    return df

if __name__ == '__main__':
    config_path = 'config/clean.json'
    clean_memo(config_path)