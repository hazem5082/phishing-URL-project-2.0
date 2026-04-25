"""
eda_visualizations.py
---------------------
Generates the 3 EDA artifacts required by the rubric:
  1. Class Distribution Bar Chart  (results/eda_class_distribution.png)
  2. URL Length Histogram           (results/eda_url_length_histogram.png)
  3. Feature Correlation Heatmap    (results/eda_correlation_heatmap.png)

Run AFTER url_features.py has created data/processed/phishing_processed.csv.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve project-root regardless of CWD
base_dir = os.path.dirname(os.path.dirname(__file__))
processed_path = os.path.join(base_dir, "data", "processed", "phishing_processed.csv")
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)


def load_data(path):
    """Load the processed feature CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at: {path}\n"
            "Please run url_features.py first."
        )
    return pd.read_csv(path)


def plot_class_distribution(df, out_dir):
    """Figure 6 — Class Distribution Bar Chart."""
    counts = df['label'].value_counts().sort_index()
    labels = ['Legitimate', 'Phishing']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=['#2196F3', '#F44336'], width=0.5, edgecolor='white')

    # Annotate bar heights
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Class Distribution: Legitimate vs Phishing URLs', fontsize=14, pad=12)
    ax.set_ylabel('Number of Samples')
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "eda_class_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"-> Saved: {out_path}")


def plot_url_length_histogram(df, out_dir):
    """Figure 7 — URL Length Histogram split by class."""
    legit = df[df['label'] == 0]['url_length']
    phish = df[df['label'] == 1]['url_length']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(legit, bins=50, alpha=0.65, color='#2196F3', label='Legitimate', edgecolor='white')
    ax.hist(phish, bins=50, alpha=0.65, color='#F44336', label='Phishing', edgecolor='white')

    ax.set_title('Distribution of URL Lengths by Class', fontsize=14, pad=12)
    ax.set_xlabel('URL Length (characters, protocol stripped)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # Cap x-axis so outliers don't squash the chart
    ax.set_xlim(0, df['url_length'].quantile(0.99))
    plt.tight_layout()

    out_path = os.path.join(out_dir, "eda_url_length_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"-> Saved: {out_path}")


def plot_correlation_heatmap(df, out_dir):
    """Figure 8 — Feature Correlation Heatmap."""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, pad=12)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "eda_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"-> Saved: {out_path}")


if __name__ == "__main__":
    print("Loading processed data...")
    df = load_data(processed_path)
    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns\n")

    print("Generating EDA visualizations...")
    plot_class_distribution(df, results_dir)
    plot_url_length_histogram(df, results_dir)
    plot_correlation_heatmap(df, results_dir)

    print("\nAll 3 EDA figures saved to results/")
