import os
import pandas as pd

def normalize_url(url):
    """
    Strips the protocol prefix (http:// or https://) from a URL.
    The training dataset's legitimate URLs had no protocol, so we must
    normalize user input to match that same distribution before predicting.
    """
    # Record whether the URL originally used HTTPS before stripping.
    has_https = 1 if url.startswith('https://') else 0
    
    # Strip the protocol so length and dot_count match the training data format.
    for prefix in ('https://', 'http://'):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    
    return url, has_https


def extract_url_features(url):
    """
    Extracts numerical features from a single URL string.
    Normalizes the URL to match the training data format before extracting.
    Returns a dictionary of features.
    """
    # Normalize the URL and extract the https flag in one step.
    # This ensures features are computed on the same format as the training data.
    normalized_url, has_https = normalize_url(url)
    
    # 1. URL length measured on the normalized (protocol-stripped) URL.
    url_length = len(normalized_url)
    
    # 2. Count of dots: Phishing URLs often use many subdomains.
    dot_count = normalized_url.count('.')
    
    # 3. Presence of '@': A common phishing trick to disguise the real host.
    has_at_symbol = 1 if '@' in normalized_url else 0
    
    # Return all extracted features as a dictionary.
    return {
        'url_length': url_length,
        'dot_count': dot_count,
        'has_at_symbol': has_at_symbol,
        'has_https': has_https
    }

def process_dataset(input_file, output_file):
    """
    Loads raw data, applies feature extraction, and saves the processed dataset.
    """
    print(f"Loading raw data from: {input_file}")
    # Read the dataset into a pandas DataFrame.
    df = pd.read_csv(input_file)
    
    # If the dataset is massive (e.g. 800k rows), we sample it to keep the project fast!
    if len(df) > 20000:
        print("Dataset is very large. Sampling 20,000 examples to maintain quick training times...")
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    
    print("Extracting features from URLs...")
    
    # We use highly efficient pandas vectorized operations instead of slow row-by-row apply!
    texts = df['text'].astype(str)
    
    features_df = pd.DataFrame({
        'url_length': texts.str.len(),
        'dot_count': texts.str.count(r'\.'),
        'has_at_symbol': texts.str.contains('@').astype(int),
        'has_https': texts.str.startswith('https://').astype(int)
    })
    
    # Combine the new features with the original label column.
    # We keep the label for training our models later.
    processed_df = pd.concat([features_df, df['label']], axis=1)
    
    print(f"Saving processed data to: {output_file}")
    # Save the result to a new CSV file without the row indices.
    processed_df.to_csv(output_file, index=False)
    print("Feature engineering complete!")

if __name__ == "__main__":
    # Define absolute paths using os.path.join to ensure cross-platform compatibility.
    # base_dir points to project-root since this script is in src/feature_engineering/.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Define paths for the raw input data and the processed output data.
    raw_data_path = os.path.join(base_dir, "data", "raw", "phishing_raw.csv")
    processed_data_path = os.path.join(base_dir, "data", "processed", "phishing_processed.csv")
    
    # Check if raw data exists before trying to process it.
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data not found at {raw_data_path}")
        print("Please ensure you have run data_loader.py first.")
    else:
        # Run the processing pipeline.
        process_dataset(raw_data_path, processed_data_path)
