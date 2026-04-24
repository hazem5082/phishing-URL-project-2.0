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
    """
    Loads processed data, trains distinct models, evaluates them,
    and generates required visualizations (ROC, Confusion Matrix, Feature Importance).
    """
    # Create directories for saving models and results
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading processed data from: {processed_data_path}")
    # Load the feature-engineered dataset.
    df = pd.read_csv(processed_data_path)
    
    # Separate the features (X) from the target label (y).
    X = df.drop(columns=['label'])
    y = df['label']
    
    print("Splitting data into training and testing sets...")
    # Split the data: 80% for training the models, 20% for testing their performance.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the three models we want to compare.
    # We apply a balanced class weight to address dataset imbalances during training.
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Dictionary to store ROC curve data for the combined plot
    roc_data = {}
    
    # Train, evaluate, and generate visual artifacts
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        # Get probability of the positive class for the ROC curve
        probabilities = model.predict_proba(X_test)[:, 1]
        
        print(f"Evaluating {name}:")
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Save Confusion Matrix as a visual Heatmap
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
        plt.close() # Close plot to prevent overlap
        print(f"-> Saved Confusion Matrix Heatmap to {cm_path}")
        
        # Store ROC data for later
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)
        
        # 2. Generate Feature Importance Plot for Random Forest
        if name == "Random Forest":
            plt.figure(figsize=(8, 6))
            importances = model.feature_importances_
            feature_names = X.columns
            # Sort features by importance score
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            # Create a clean bar plot
            sns.barplot(x=feat_imp, y=feat_imp.index, hue=feat_imp.index, legend=False, palette="viridis")
            plt.title("Feature Importance - Random Forest")
            plt.xlabel("Importance Score")
            plt.ylabel("URL Features")
            plt.tight_layout()
            
            feat_imp_path = os.path.join(results_dir, "feature_importance_rf.png")
            plt.savefig(feat_imp_path)
            plt.close()
            print(f"-> Saved Feature Importance Plot to {feat_imp_path}")
            
            # 4. Save the Random Forest as the final trained model
            model_path = os.path.join(models_dir, "phishing_rf_model.joblib")
            joblib.dump(model, model_path)
            print(f"-> Saved final Random Forest model to {model_path}")

    # 3. Produce an ROC-AUC Curve comparing all three models on one graph
    print(f"\n{'='*50}")
    print("Generating combined ROC-AUC Curve...")
    plt.figure(figsize=(10, 8))
    
    # Plot each model's curve
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    
    # Plot the random chance diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    
    # Formatting the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-AUC Curve Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the combined ROC curve
    roc_path = os.path.join(results_dir, "roc_auc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"-> Saved ROC Curve comparison to {roc_path}")
    
    print(f"\n{'='*50}")
    print("All tasks completed successfully! Visuals are in 'results/' and model in 'models/'.")

if __name__ == "__main__":
    # Define absolute paths using os.path.join to ensure cross-platform compatibility.
    # base_dir points to project-root since this script is in src/models/.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Define the path for the processed data we will load.
    processed_data_path = os.path.join(base_dir, "data", "processed", "phishing_processed.csv")
    
    # Check if the processed file exists before trying to run.
    if not os.path.exists(processed_data_path):
        print(f"Error: Could not find {processed_data_path}")
        print("Please run src/feature_engineering/url_features.py first!")
    else:
        # Run the training and evaluation pipeline, passing base_dir for path saving.
        train_and_evaluate_models(base_dir, processed_data_path)
