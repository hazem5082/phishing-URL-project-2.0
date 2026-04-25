import os
import sys
import joblib
import pandas as pd

# Dynamically resolve project-root so imports work regardless of CWD
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.feature_engineering.url_features import extract_url_features


def predict_url(url, model):
    """Extracts features from a URL and returns a string prediction label."""
    features_df = pd.DataFrame([extract_url_features(url)])
    return "Phishing" if model.predict(features_df)[0] == 1 else "Legitimate"


def run_demo():
    """
    CLI demo: accepts an optional URL argument, then always prints 10 sample predictions.
    Run with: python demo.py [url]
    """
    print("=" * 60)
    print("   PHISHING URL DETECTION SYSTEM - WORKING DEMO")
    print("=" * 60)

    model_path = os.path.join(base_dir, "models", "phishing_rf_model.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_classifier.py first.")
        return

    print("[*] Loading Random Forest Model...")
    rf_model = joblib.load(model_path)
    print("[*] Model loaded.\n")

    # Accept URL from command-line arg or interactive prompt
    if len(sys.argv) > 1:
        user_url = sys.argv[1]
    else:
        user_url = input("Enter a URL to test (or press Enter to skip): ").strip()

    if user_url:
        print("\n" + "-" * 50)
        print(f"Analyzing: {user_url}")
        prediction = predict_url(user_url, rf_model)
        result = "[ *** PHISHING *** ]" if prediction == "Phishing" else "[ LEGITIMATE ]"
        print(f"Prediction:  {result}")
        print("-" * 50 + "\n")

    # 10 required sample predictions (rubric)
    print("=" * 60)
    print("   RUBRIC REQUIREMENT: 10 SAMPLE PREDICTIONS")
    print("=" * 60)

    test_cases = [
        # 5 Legitimate
        ("https://www.google.com", "Legitimate"),
        ("https://github.com/microsoft/vscode", "Legitimate"),
        ("https://en.wikipedia.org/wiki/Machine_learning", "Legitimate"),
        ("https://www.amazon.com/gp/cart/view.html", "Legitimate"),
        ("https://stackoverflow.com/questions", "Legitimate"),
        # 5 Phishing
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
