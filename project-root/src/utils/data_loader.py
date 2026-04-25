import os
import pandas as pd
from datasets import load_dataset


def get_data_path(subfolder="raw"):
    """Returns the absolute path to a data subfolder."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "data", subfolder)


def download_huggingface_dataset(dataset_name="ealvaradob/phishing-dataset"):
    """Downloads the dataset from Hugging Face and saves it to data/raw."""
    try:
        print(f"Fetching {dataset_name}...")
        # Load raw JSON directly — the repo's dataset script is blocked by newer HF versions
        dataset = load_dataset("json", data_files="hf://datasets/ealvaradob/phishing-dataset/urls.json")

        df = pd.DataFrame(dataset['train'])
        output_file = os.path.join(get_data_path("raw"), "phishing_raw.csv")
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    download_huggingface_dataset()