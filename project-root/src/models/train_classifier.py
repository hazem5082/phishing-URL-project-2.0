import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def train_and_evaluate_models(base_dir, processed_data_path):
    """Trains three models, evaluates them, and saves visual artifacts and the final model."""
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading processed data from: {processed_data_path}")
    df = pd.read_csv(processed_data_path)

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # class_weight='balanced' penalises errors on the minority class proportionally
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    roc_data = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Save confusion matrix heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        cm_path = os.path.join(results_dir, f"cm_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"-> Saved Confusion Matrix to {cm_path}")

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

        if name == "Random Forest":
            # Feature importance reveals which structural signals matter most
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(8, 6))
            sns.barplot(x=feat_imp, y=feat_imp.index, hue=feat_imp.index, legend=False, palette="viridis")
            plt.title("Feature Importance - Random Forest")
            plt.xlabel("Importance Score")
            plt.ylabel("URL Features")
            plt.tight_layout()
            feat_imp_path = os.path.join(results_dir, "feature_importance_rf.png")
            plt.savefig(feat_imp_path)
            plt.close()
            print(f"-> Saved Feature Importance Plot to {feat_imp_path}")

            model_path = os.path.join(models_dir, "phishing_rf_model.joblib")
            joblib.dump(model, model_path)
            print(f"-> Saved trained model to {model_path}")

    # Combined ROC-AUC comparison across all three models
    print(f"\n{'='*50}")
    print("Generating combined ROC-AUC Curve...")
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-AUC Curve Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "roc_auc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"-> Saved ROC Curve to {roc_path}")
    print(f"\n{'='*50}")
    print("All tasks completed. Visuals are in 'results/', model in 'models/'.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_data_path = os.path.join(base_dir, "data", "processed", "phishing_processed.csv")

    if not os.path.exists(processed_data_path):
        print(f"Error: Could not find {processed_data_path}")
        print("Please run url_features.py first.")
    else:
        train_and_evaluate_models(base_dir, processed_data_path)
