import os
import pandas as pd


def normalize_url(url):
    """Strips http/https prefix, recording the https flag before removal."""
    has_https = 1 if url.startswith('https://') else 0
    # Strip protocol so length/dot features match the training data distribution
    for prefix in ('https://', 'http://'):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return url, has_https


def extract_url_features(url):
    """Extracts 4 numerical features from a URL string."""
    normalized_url, has_https = normalize_url(url)
    return {
        'url_length': len(normalized_url),
        'dot_count': normalized_url.count('.'),
        'has_at_symbol': 1 if '@' in normalized_url else 0,  # '@' hides the true host
        'has_https': has_https,
    }


def process_dataset(input_file, output_file):
    """Loads raw CSV, extracts features via vectorized ops, and saves processed CSV."""
    print(f"Loading raw data from: {input_file}")
    df = pd.read_csv(input_file)

    # Sample large datasets to keep training times reasonable
    if len(df) > 20000:
        print("Large dataset detected — sampling 20,000 rows.")
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)

    print("Extracting features from URLs...")
    texts = df['text'].astype(str)

    # Vectorized extraction is significantly faster than row-by-row apply()
    features_df = pd.DataFrame({
        'url_length': texts.str.len(),
        'dot_count': texts.str.count(r'\.'),
        'has_at_symbol': texts.str.contains('@').astype(int),
        'has_https': texts.str.startswith('https://').astype(int),
    })

    processed_df = pd.concat([features_df, df['label']], axis=1)
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed data to: {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_data_path = os.path.join(base_dir, "data", "raw", "phishing_raw.csv")
    processed_data_path = os.path.join(base_dir, "data", "processed", "phishing_processed.csv")

    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data not found at {raw_data_path}")
        print("Please run data_loader.py first.")
    else:
        process_dataset(raw_data_path, processed_data_path)
