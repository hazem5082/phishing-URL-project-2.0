import os
import sys

# Dynamically resolve project-root so imports work regardless of where pytest is run from
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.feature_engineering.url_features import normalize_url, extract_url_features, process_dataset
from src.utils.data_loader import get_data_path, download_huggingface_dataset
from src.demo import predict_url, run_demo
import pandas as pd
from unittest.mock import patch

def test_csv_creation():
    """Checks if the raw data was successfully created."""
    csv_path = os.path.join(base_dir, "data", "raw", "phishing_raw.csv")
    assert os.path.exists(csv_path), f"Error: CSV not found at {csv_path}"

def test_get_data_path():
    """Tests the data path resolution logic in data_loader."""
    path = get_data_path("raw")
    assert "data" in path
    assert "raw" in path

def test_normalize_url():
    """Tests that protocols are correctly stripped and recorded."""
    url1, https1 = normalize_url("https://www.google.com")
    assert url1 == "www.google.com"
    assert https1 == 1

    url2, https2 = normalize_url("http://test.com")
    assert url2 == "test.com"
    assert https2 == 0

    url3, https3 = normalize_url("no-protocol.com")
    assert url3 == "no-protocol.com"
    assert https3 == 0

def test_extract_url_features():
    """Tests the feature engineering logic for accurate metric extraction."""
    # Test a legitimate-looking URL
    features1 = extract_url_features("https://github.com")
    assert features1['url_length'] == len("github.com")
    assert features1['dot_count'] == 1
    assert features1['has_at_symbol'] == 0
    assert features1['has_https'] == 1

    # Test a phishing-looking URL
    features2 = extract_url_features("http://secure@login.bank.com.update.net")
    assert features2['url_length'] == len("secure@login.bank.com.update.net")
    assert features2['dot_count'] == 4
    assert features2['has_at_symbol'] == 1
    assert features2['has_https'] == 0

def test_process_dataset(tmp_path):
    """Tests the dataset processing pipeline using a tiny dummy dataset."""
    # Create a dummy raw CSV
    dummy_input = tmp_path / "dummy_raw.csv"
    dummy_output = tmp_path / "dummy_processed.csv"
    
    df = pd.DataFrame({
        "text": ["https://google.com", "http://bad@phish.com"],
        "label": [0, 1]
    })
    df.to_csv(dummy_input, index=False)
    
    # Process it
    process_dataset(str(dummy_input), str(dummy_output))
    
    # Verify output
    assert os.path.exists(dummy_output)
    processed_df = pd.read_csv(dummy_output)
    
    assert len(processed_df) == 2
    assert "url_length" in processed_df.columns
    assert "dot_count" in processed_df.columns
    assert "has_at_symbol" in processed_df.columns
    assert "has_https" in processed_df.columns
    assert "label" in processed_df.columns
    
    # Check specific values
    assert processed_df.loc[0, "has_https"] == 1
    assert processed_df.loc[1, "has_https"] == 0

class DummyModel:
    def predict(self, df):
        # Return 1 if URL has an @ symbol, else 0
        if df['has_at_symbol'].iloc[0] == 1:
            return [1]
        return [0]

def test_predict_url():
    """Tests the prediction wrapper using a mock model."""
    mock_model = DummyModel()
    
    pred1 = predict_url("https://safe.com", mock_model)
    assert pred1 == "Legitimate"
    
    pred2 = predict_url("http://bad@site.com", mock_model)
    assert pred2 == "Phishing"

@patch('src.utils.data_loader.load_dataset')
@patch('pandas.DataFrame.to_csv')
def test_download_huggingface_dataset(mock_to_csv, mock_load_dataset):
    """Tests the dataset downloader with mocked network and disk IO."""
    # Setup mock returns
    mock_load_dataset.return_value = {'train': [{'text': 'url1', 'label': 0}]}
    
    # Run the function
    result = download_huggingface_dataset()
    
    # Assert it completed successfully
    assert result is True
    
    # Assert the mock was called correctly
    mock_load_dataset.assert_called_once()
    mock_to_csv.assert_called_once()

@patch('src.utils.data_loader.load_dataset')
def test_download_huggingface_dataset_failure(mock_load_dataset):
    """Tests the failure branch of the downloader."""
    # Force an exception
    mock_load_dataset.side_effect = Exception("Mock Network Error")
    
    result = download_huggingface_dataset()
    assert result is False

@patch('src.demo.os.path.exists')
@patch('src.demo.joblib.load')
@patch('src.demo.sys.argv', ['demo.py', 'https://test.com'])
def test_run_demo(mock_joblib_load, mock_exists):
    """Tests the CLI script execution with a mocked model and CLI arguments."""
    # Ensure it thinks the model file exists
    mock_exists.return_value = True
    
    # Provide our dummy model
    mock_joblib_load.return_value = DummyModel()
    
    # Execute the demo, it shouldn't block on input because of mocked sys.argv
    # and it shouldn't crash because we provided a mock model.
    run_demo()
    
    mock_exists.assert_called()
    mock_joblib_load.assert_called_once()



