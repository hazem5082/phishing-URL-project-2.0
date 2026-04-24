import os
import pandas as pd
from datasets import load_dataset

def get_data_path(subfolder="raw"):
    """Returns the absolute path to a data subfolder using relative logic."""
    # Using relative paths to avoid hardcoding [cite: 402]
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "data", subfolder)

def download_huggingface_dataset(dataset_name="ealvaradob/phishing-dataset"):
    """
    Downloads the dataset from Hugging Face and saves it to data/raw.
    Follows PEP8 standards with modular logic[cite: 389, 393].
    """
    try:
        print(f"Fetching {dataset_name}...")
        # Loading the dataset from Hugging Face directly from the json to avoid unsupported dataset scripts
        dataset = load_dataset("json", data_files="hf://datasets/ealvaradob/phishing-dataset/combined_reduced.json")
        
        # Convert to DataFrame (assuming 'train' split exists)
        df = pd.DataFrame(dataset['train'])
        
        # Define output path
        output_dir = get_data_path("raw")
        output_file = os.path.join(output_dir, "phishing_raw.csv")
        
        # Save to CSV [cite: 259]
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_huggingface_dataset()