import os
import sys
import joblib
import pandas as pd

# Add the project-root directory to the Python path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.feature_engineering.url_features import extract_url_features


def predict_url(url, model):
    """
    Takes a raw URL string, extracts its numerical features,
    and predicts whether it is Phishing or Legitimate.
    """
    features_dict = extract_url_features(url)
    features_df = pd.DataFrame([features_dict])
    prediction = model.predict(features_df)[0]
    return "Phishing" if prediction == 1 else "Legitimate"


def run_demo():
    """
    CLI script that loads the model, predicts an interactive URL,
    AND runs 10 sample test cases as required by the rubric.
    """
    print("=" * 60)
    print("   PHISHING URL DETECTION SYSTEM - WORKING DEMO")
    print("=" * 60)

    model_path = os.path.join(base_dir, "models", "phishing_rf_model.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Could not find the trained model at {model_path}")
        print("Please make sure you ran train_classifier.py first!")
        return

    print("[*] Loading Random Forest Model...")
    rf_model = joblib.load(model_path)
    print("[*] Model successfully loaded!\n")

    # 1. Interactive CLI Prediction
    # Support both command-line arguments and input prompts
    if len(sys.argv) > 1:
        user_url = sys.argv[1]
    else:
        user_url = input("Enter a URL to test (or press Enter to skip): ").strip()

    if user_url:
        print("\n" + "-" * 50)
        print(f"Analyzing: {user_url}")
        prediction = predict_url(user_url, rf_model)

        # Use plain ASCII labels to avoid Windows cp1252 encoding errors
        if prediction == "Phishing":
            print("Prediction:  [ *** PHISHING *** ]")
        else:
            print("Prediction:  [ LEGITIMATE ]")
        print("-" * 50 + "\n")

    # 2. Automated 10 Sample Test Cases (required by rubric)
    print("=" * 60)
    print("   RUBRIC REQUIREMENT: 10 SAMPLE PREDICTIONS")
    print("=" * 60)

    test_cases = [
        # 5 Legitimate Examples - real-world HTTPS URLs
        ("https://www.google.com", "Legitimate"),
        ("https://github.com/microsoft/vscode", "Legitimate"),
        ("https://en.wikipedia.org/wiki/Machine_learning", "Legitimate"),
        ("https://www.amazon.com/gp/cart/view.html", "Legitimate"),
        ("https://stackoverflow.com/questions", "Legitimate"),

        # 5 Phishing/Suspicious Examples
        ("http://secure-login.bank.com.update-account.info", "Phishing"),
        ("http://verify-apple-id.com@login.com", "Phishing"),
        ("http://www.paypal.com.cgi-bin.webscr.cmd.login.submit.com", "Phishing"),
        ("http://amazon-update-security-alert-verify-now.info/login", "Phishing"),
        ("http://paypal-secure-account-verification.com@signin.info", "Phishing"),
    ]

    for url, expected in test_cases:
        prediction = predict_url(url, rf_model)
        match = "OK" if prediction == expected else "FAIL"

        print(f"URL: {url}")
        print(f"Expected: {expected:<10} | Predicted: {prediction:<10} [{match}]\n")

    print("=" * 60)


if __name__ == "__main__":
    run_demo()
