import os

def test_csv_creation():
    """Checks if the phishing_raw.csv was successfully created."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data", "raw", "phishing_raw.csv")
    
    assert os.path.exists(csv_path), f"Error: CSV not found at {csv_path}"
    print("Test passed: phishing_raw.csv exists!")

if __name__ == "__main__":
    test_csv_creation()
