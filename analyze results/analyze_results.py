import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, fbeta_score,
                             balanced_accuracy_score, matthews_corrcoef)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import argparse
from collections import defaultdict


# Function to parse command-line arguments
def parse_arguments():
    """
    Parse command-line arguments

    Returns:
        Namespace containing the arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze antisemitism detection with focus on precision and false positives')
    parser.add_argument('--input', type=str, default='comparative_model_results.csv',
                        help='Input CSV file with combined model results')
    parser.add_argument('--fp-analysis', action='store_true',
                        help='Perform detailed false positive analysis with topic modeling')
    parser.add_argument('--topics', type=int, default=5,
                        help='Number of topics for topic modeling (default: 5)')

    return parser.parse_args()


# Function to create output directories
def create_output_dirs():
    """
    Create directories for outputs

    Returns:
        Dictionary with paths to the directories
    """
    # Create base directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots/fp_analysis', exist_ok=True)

    return {
        'plots': 'plots',
        'results': 'results',
        'fp_plots': 'plots/fp_analysis'
    }


# Function to calculate F-beta score with beta=0.5 (emphasizing precision)
def calculate_fbeta(y_true, y_pred, beta=0.5):
    """
    Calculate F-beta score with beta=0.5 to emphasize precision over recall

    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta value (default: 0.5)

    Returns:
        F-beta score
    """
    return fbeta_score(y_true, y_pred, beta=beta)


# Function to display focused performance metrics
def display_focused_metrics(true_labels, predicted_labels, model_name):
    """
    Calculate and display focused performance metrics with emphasis on false positives

    Args:
        true_labels: Ground truth labels (0/1 where 1=antisemitic)
        predicted_labels: Model predictions (0/1 where 1=antisemitic)
        model_name: Name of the model/definition

    Returns:
        Dictionary of metrics
    """
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Core metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    # Focus metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 1 - false positive rate
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    f05 = calculate_fbeta(true_labels, predicted_labels, beta=0.5)

    # False positive rate
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0

    # Additional metrics
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

    # Total number of predicted positives
    predicted_positives = tp + fp

    print(f"\n===== Performance Metrics for {model_name} =====")
    print(
        f"Precision:          {precision:.4f} - Of tweets classified as antisemitic, {precision * 100:.1f}% were actually antisemitic")
    print(
        f"Specificity:        {specificity:.4f} - Correctly identified {specificity * 100:.1f}% of non-antisemitic tweets")
    print(
        f"False Positive Rate: {fp_rate:.4f} - Incorrectly classified {fp_rate * 100:.1f}% of non-antisemitic tweets as antisemitic")
    print(f"F0.5 Score:         {f05:.4f} - Weighted harmonic mean of precision and recall (precision weighted more)")
    print(f"F1 Score:           {f1:.4f} - Balanced harmonic mean of precision and recall")
    print(f"Recall/Sensitivity: {recall:.4f} - Found {recall * 100:.1f}% of all antisemitic tweets")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"Matthews Corr Coef: {mcc:.4f}")

    print("\nError Analysis:")
    print(f"False Positives:    {fp} tweets ({fp / predicted_positives * 100:.1f}% of all antisemitic predictions)")
    print(f"False Negatives:    {fn} tweets")
    print(f"Total Errors:       {fp + fn} tweets")

    # Return metrics dictionary
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'f05': f05,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'false_positive_rate': fp_rate,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'total_predicted_positive': predicted_positives
    }


# Function to analyze false positives by tweet length
def analyze_fp_by_length(df, model_col, dirs):
    """
    Analyze false positives by tweet length

    Args:
        df: DataFrame containing model results
        model_col: Column name for model predictions
        dirs: Dictionary with output directories

    Returns:
        DataFrame with false positive rates by length bin
    """
    # Create length bins
    bins = [0, 80, 120, 160, 200, 240, 280, 320, 400, 600, 1000]
    labels = ['0-80', '81-120', '121-160', '161-200', '201-240',
              '241-280', '281-320', '321-400', '401-600', '601+']

    df['length_bin'] = pd.cut(df['text_length'], bins=bins, labels=labels)

    # Calculate false positive metrics by length bin
    length_stats = []

    for length_bin in labels:
        bin_df = df[df['length_bin'] == length_bin]
        if len(bin_df) == 0:
            continue

        # Filter non-antisemitic tweets (ground truth = 0)
        non_antisemitic = bin_df[bin_df['Biased'] == 0]
        if len(non_antisemitic) == 0:
            continue

        # Count false positives
        fp = non_antisemitic[non_antisemitic[model_col] == 1]

        # Calculate metrics
        fp_rate = len(fp) / len(non_antisemitic)

        length_stats.append({
            'Length_Bin': length_bin,
            'Tweet_Count': len(bin_df),
            'Non_Antisemitic_Count': len(non_antisemitic),
            'False_Positive_Count': len(fp),
            'False_Positive_Rate': fp_rate
        })

    # Create DataFrame
    length_df = pd.DataFrame(length_stats)

    # Plot false positive rate by length
    plt.figure(figsize=(12, 6))
    plt.bar(length_df['Length_Bin'], length_df['False_Positive_Rate'], color='indianred')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('False Positive Rate')
    model_name = model_col.replace('_Binary', '')
    plt.title(f'False Positive Rate by Tweet Length - {model_name}')
    plt.ylim(0, min(1.0, length_df['False_Positive_Rate'].max() * 1.2))
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels
    for i, row in length_df.iterrows():
        plt.text(i, row['False_Positive_Rate'] + 0.01,
                 f"{row['False_Positive_Count']}/{row['Non_Antisemitic_Count']}",
                 ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(dirs['fp_plots'], f'fp_by_length_{model_name}.png'))

    return length_df


# Function to analyze false positives by keyword
def analyze_fp_by_keyword(df, model_col, dirs, top_n=15):
    """
    Analyze false positives by keyword

    Args:
        df: DataFrame containing model results
        model_col: Column name for model predictions
        dirs: Dictionary with output directories
        top_n: Number of top keywords to include in plot

    Returns:
        DataFrame with false positive rates by keyword
    """
    # Calculate false positive metrics by keyword
    keyword_stats = []

    for keyword in df['Keyword'].unique():
        keyword_df = df[df['Keyword'] == keyword]
        if len(keyword_df) < 5:  # Skip keywords with too few examples
            continue

        # Filter non-antisemitic tweets (ground truth = 0)
        non_antisemitic = keyword_df[keyword_df['Biased'] == 0]
        if len(non_antisemitic) == 0:
            continue

        # Count false positives
        fp = non_antisemitic[non_antisemitic[model_col] == 1]

        # Calculate metrics
        fp_rate = len(fp) / len(non_antisemitic)

        keyword_stats.append({
            'Keyword': keyword,
            'Tweet_Count': len(keyword_df),
            'Non_Antisemitic_Count': len(non_antisemitic),
            'False_Positive_Count': len(fp),
            'False_Positive_Rate': fp_rate
        })

    # Create DataFrame and sort by false positive rate
    keyword_df = pd.DataFrame(keyword_stats)
    keyword_df = keyword_df.sort_values('False_Positive_Rate', ascending=False)

    # Select top N keywords with most false positives
    top_keywords = keyword_df.head(top_n)

    # Plot false positive rate by keyword
    plt.figure(figsize=(14, 8))
    bars = plt.barh(top_keywords['Keyword'], top_keywords['False_Positive_Rate'], color='darkorange')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Keyword')
    model_name = model_col.replace('_Binary', '')
    plt.title(f'False Positive Rate by Keyword - {model_name}')
    plt.xlim(0, min(1.0, top_keywords['False_Positive_Rate'].max() * 1.2))
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add count labels
    for bar, (i, row) in zip(bars, top_keywords.iterrows()):
        plt.text(row['False_Positive_Rate'] + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{row['False_Positive_Count']}/{row['Non_Antisemitic_Count']}",
                 va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(dirs['fp_plots'], f'fp_by_keyword_{model_name}.png'))

    return keyword_df


# Function to perform topic modeling on false positives
def analyze_fp_topics(df, model_col, n_topics=5, dirs=None):
    """
    Perform topic modeling on false positive tweets

    Args:
        df: DataFrame containing model results
        model_col: Column name for model predictions
        n_topics: Number of topics to identify
        dirs: Dictionary with output directories

    Returns:
        Dictionary with topic modeling results
    """
    # Identify false positives
    false_positives = df[(df['Biased'] == 0) & (df[model_col] == 1)]

    if len(false_positives) < 10:
        print(f"Too few false positives for topic modeling: {len(false_positives)}")
        return None

    # Preprocess text
    texts = false_positives['Text'].tolist()

    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Perform LDA topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    # Extract top words per topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'words': top_words,
            'weights': topic[top_words_idx].tolist()
        })

    # Assign topics to false positives
    topic_assignments = lda.transform(dtm)
    false_positives['Topic'] = topic_assignments.argmax(axis=1)

    # Count tweets per topic
    topic_counts = false_positives['Topic'].value_counts().to_dict()

    # Store topic info
    for i, topic in enumerate(topics):
        topic['tweet_count'] = topic_counts.get(i, 0)
        topic['percentage'] = topic['tweet_count'] / len(false_positives)

    # Plot topic distribution
    if dirs:
        # Plot topic distribution
        topic_df = pd.DataFrame({
            'Topic': [f"Topic {i + 1}: {' '.join(topic['words'][:3])}" for i, topic in enumerate(topics)],
            'Count': [topic['tweet_count'] for topic in topics]
        })

        # Sort by count
        topic_df = topic_df.sort_values('Count', ascending=False)

        plt.figure(figsize=(14, 8))
        bars = plt.barh(topic_df['Topic'], topic_df['Count'], color='mediumseagreen')
        plt.xlabel('Number of False Positive Tweets')
        plt.ylabel('Topic')
        model_name = model_col.replace('_Binary', '')
        plt.title(f'False Positive Topics - {model_name}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add percentage labels
        for bar, (i, row) in zip(bars, topic_df.iterrows()):
            percentage = row['Count'] / topic_df['Count'].sum() * 100
            plt.text(row['Count'] + 1, bar.get_y() + bar.get_height() / 2,
                     f"{percentage:.1f}%", va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(dirs['fp_plots'], f'fp_topics_{model_name}.png'))

        # Create a more detailed HTML output with examples
        samples_per_topic = defaultdict(list)
        for _, row in false_positives.iterrows():
            if len(samples_per_topic[row['Topic']]) < 5:  # Get 5 examples per topic
                samples_per_topic[row['Topic']].append(row['Text'])

        html_content = f"<h1>False Positive Analysis for {model_name}</h1>\n"
        html_content += "<p>Topics identified in false positive tweets</p>\n"

        for i, topic in enumerate(topics):
            html_content += f"<h2>Topic {i + 1} ({topic['tweet_count']} tweets, {topic['percentage'] * 100:.1f}%)</h2>\n"
            html_content += "<h3>Top words:</h3>\n<ul>\n"
            for word, weight in zip(topic['words'], topic['weights']):
                html_content += f"<li>{word} ({weight:.3f})</li>\n"
            html_content += "</ul>\n"

            html_content += "<h3>Example tweets:</h3>\n<ul>\n"
            for sample in samples_per_topic.get(i, []):
                html_content += f"<li>{sample}</li>\n"
            html_content += "</ul>\n"

        with open(os.path.join(dirs['results'], f'fp_topic_analysis_{model_name}.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

    return {
        'topics': topics,
        'topic_assignments': topic_assignments,
        'false_positives': false_positives
    }


# Function to compare false positives between model sizes


def compare_model_sizes(df, dirs):
    """
    Compare false positive patterns between model sizes

    Args:
        df: DataFrame containing results from both model sizes
        dirs: Dictionary with output directories

    Returns:
        Dictionary with comparison results
    """
    # Extract metrics for different model sizes
    metrics_7b_ihra = display_focused_metrics(df['Biased'], df['IHRA_Binary_7B'], "IHRA Definition (7B)")
    metrics_8b_ihra = display_focused_metrics(df['Biased'], df['IHRA_Binary_8B'], "IHRA Definition (8B)")
    metrics_7b_jda = display_focused_metrics(df['Biased'], df['JDA_Binary_7B'], "JDA Definition (7B)")
    metrics_8b_jda = display_focused_metrics(df['Biased'], df['JDA_Binary_8B'], "JDA Definition (8B)")

    # Compile metrics for comparison
    metrics = [metrics_7b_ihra, metrics_8b_ihra, metrics_7b_jda, metrics_8b_jda]

    # Create model size comparison table
    comparison_data = {
        'Metric': ['Precision', 'Specificity', 'False Positive Rate', 'F0.5 Score', 'F1 Score', 'Recall'],
        'IHRA 7B': [metrics_7b_ihra['precision'], metrics_7b_ihra['specificity'],
                    metrics_7b_ihra['false_positive_rate'], metrics_7b_ihra['f05'],
                    metrics_7b_ihra['f1'], metrics_7b_ihra['recall']],
        'IHRA 8B': [metrics_8b_ihra['precision'], metrics_8b_ihra['specificity'],
                    metrics_8b_ihra['false_positive_rate'], metrics_8b_ihra['f05'],
                    metrics_8b_ihra['f1'], metrics_8b_ihra['recall']],
        'JDA 7B': [metrics_7b_jda['precision'], metrics_7b_jda['specificity'],
                   metrics_7b_jda['false_positive_rate'], metrics_7b_jda['f05'],
                   metrics_7b_jda['f1'], metrics_7b_jda['recall']],
        'JDA 8B': [metrics_8b_jda['precision'], metrics_8b_jda['specificity'],
                   metrics_8b_jda['false_positive_rate'], metrics_8b_jda['f05'],
                   metrics_8b_jda['f1'], metrics_8b_jda['recall']]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\n===== Model Size Comparison =====")
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(comparison_df)

    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(dirs['results'], 'model_size_comparison.csv'), index=False)

    # Calculate relative improvements from 7B to 8B
    improvement_data = {
        'Metric': ['Precision', 'Specificity', 'False Positive Rate', 'F0.5 Score', 'F1 Score', 'Recall'],
        'IHRA Improvement': [
            (metrics_8b_ihra['precision'] - metrics_7b_ihra['precision']) / max(0.0001, metrics_7b_ihra['precision']),
            (metrics_8b_ihra['specificity'] - metrics_7b_ihra['specificity']) / max(0.0001,
                                                                                    metrics_7b_ihra['specificity']),
            (metrics_7b_ihra['false_positive_rate'] - metrics_8b_ihra['false_positive_rate']) / max(0.0001,
                                                                                                    metrics_7b_ihra[
                                                                                                        'false_positive_rate']),
            (metrics_8b_ihra['f05'] - metrics_7b_ihra['f05']) / max(0.0001, metrics_7b_ihra['f05']),
            (metrics_8b_ihra['f1'] - metrics_7b_ihra['f1']) / max(0.0001, metrics_7b_ihra['f1']),
            (metrics_8b_ihra['recall'] - metrics_7b_ihra['recall']) / max(0.0001, metrics_7b_ihra['recall'])
        ],
        'JDA Improvement': [
            (metrics_8b_jda['precision'] - metrics_7b_jda['precision']) / max(0.0001, metrics_7b_jda['precision']),
            (metrics_8b_jda['specificity'] - metrics_7b_jda['specificity']) / max(0.0001,
                                                                                  metrics_7b_jda['specificity']),
            (metrics_7b_jda['false_positive_rate'] - metrics_8b_jda['false_positive_rate']) / max(0.0001,
                                                                                                  metrics_7b_jda[
                                                                                                      'false_positive_rate']),
            (metrics_8b_jda['f05'] - metrics_7b_jda['f05']) / max(0.0001, metrics_7b_jda['f05']),
            (metrics_8b_jda['f1'] - metrics_7b_jda['f1']) / max(0.0001, metrics_7b_jda['f1']),
            (metrics_8b_jda['recall'] - metrics_7b_jda['recall']) / max(0.0001, metrics_7b_jda['recall'])
        ]
    }

    improvement_df = pd.DataFrame(improvement_data)
    pd.set_option('display.float_format', '{:.2%}'.format)
    print("\n===== Relative Improvement (7B to 8B) =====")
    print(improvement_df)

    # Save improvement to CSV
    improvement_df.to_csv(os.path.join(dirs['results'], 'model_size_improvement.csv'), index=False)

    # Create comparison visualizations
    # Focus metrics comparison
    plt.figure(figsize=(14, 8))

    metrics_to_plot = ['Precision', 'Specificity', 'F0.5 Score', 'F1 Score', 'Recall']
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    plt.bar(x - width * 1.5, [metrics_7b_ihra[m.lower()] for m in ['precision', 'specificity', 'f05', 'f1', 'recall']],
            width, label='IHRA 7B', color='royalblue')
    plt.bar(x - width * 0.5, [metrics_8b_ihra[m.lower()] for m in ['precision', 'specificity', 'f05', 'f1', 'recall']],
            width, label='IHRA 8B', color='lightsteelblue')
    plt.bar(x + width * 0.5, [metrics_7b_jda[m.lower()] for m in ['precision', 'specificity', 'f05', 'f1', 'recall']],
            width, label='JDA 7B', color='darkgreen')
    plt.bar(x + width * 1.5, [metrics_8b_jda[m.lower()] for m in ['precision', 'specificity', 'f05', 'f1', 'recall']],
            width, label='JDA 8B', color='lightgreen')

    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison Between Model Sizes')
    plt.xticks(x, metrics_to_plot)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], 'model_size_performance_comparison.png'))

    # False positive rate comparison
    plt.figure(figsize=(10, 6))

    plt.bar(['IHRA 7B', 'IHRA 8B', 'JDA 7B', 'JDA 8B'],
            [metrics_7b_ihra['false_positive_rate'], metrics_8b_ihra['false_positive_rate'],
             metrics_7b_jda['false_positive_rate'], metrics_8b_jda['false_positive_rate']],
            color=['royalblue', 'lightsteelblue', 'darkgreen', 'lightgreen'])

    plt.xlabel('Model')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate Comparison Between Model Sizes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels
    for i, v in enumerate([metrics_7b_ihra['false_positive_rate'], metrics_8b_ihra['false_positive_rate'],
                           metrics_7b_jda['false_positive_rate'], metrics_8b_jda['false_positive_rate']]):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], 'model_size_fp_rate_comparison.png'))

    # Calculate difference in false positive classification between model sizes
    df['IHRA_FP_7B'] = ((df['Biased'] == 0) & (df['IHRA_Binary_7B'] == 1)).astype(int)
    df['IHRA_FP_8B'] = ((df['Biased'] == 0) & (df['IHRA_Binary_8B'] == 1)).astype(int)
    df['JDA_FP_7B'] = ((df['Biased'] == 0) & (df['JDA_Binary_7B'] == 1)).astype(int)
    df['JDA_FP_8B'] = ((df['Biased'] == 0) & (df['JDA_Binary_8B'] == 1)).astype(int)

    # Compare false positives: fixed in 8B vs introduced in 8B
    ihra_fixed = ((df['IHRA_FP_7B'] == 1) & (df['IHRA_FP_8B'] == 0)).sum()
    ihra_introduced = ((df['IHRA_FP_7B'] == 0) & (df['IHRA_FP_8B'] == 1)).sum()
    ihra_persistent = ((df['IHRA_FP_7B'] == 1) & (df['IHRA_FP_8B'] == 1)).sum()

    jda_fixed = ((df['JDA_FP_7B'] == 1) & (df['JDA_FP_8B'] == 0)).sum()
    jda_introduced = ((df['JDA_FP_7B'] == 0) & (df['JDA_FP_8B'] == 1)).sum()
    jda_persistent = ((df['JDA_FP_7B'] == 1) & (df['JDA_FP_8B'] == 1)).sum()

    fp_change_data = {
        'Category': ['Fixed in 8B', 'Introduced in 8B', 'Persistent'],
        'IHRA Count': [ihra_fixed, ihra_introduced, ihra_persistent],
        'JDA Count': [jda_fixed, jda_introduced, jda_persistent]
    }

    fp_change_df = pd.DataFrame(fp_change_data)
    print("\n===== False Positive Changes from 7B to 8B =====")
    print(fp_change_df)

    # Save FP change data
    fp_change_df.to_csv(os.path.join(dirs['results'], 'fp_changes_by_model_size.csv'), index=False)

    # Plot FP changes
    plt.figure(figsize=(10, 6))

    x = np.arange(len(fp_change_df['Category']))
    width = 0.35

    plt.bar(x - width / 2, fp_change_df['IHRA Count'], width, label='IHRA', color='royalblue')
    plt.bar(x + width / 2, fp_change_df['JDA Count'], width, label='JDA', color='darkgreen')

    plt.xlabel('Category')
    plt.ylabel('Number of False Positives')
    plt.title('False Positive Changes from 7B to 8B Model')
    plt.xticks(x, fp_change_df['Category'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels
    for i, v in enumerate(fp_change_df['IHRA Count']):
        plt.text(i - width / 2, v + 5, str(v), ha='center')

    for i, v in enumerate(fp_change_df['JDA Count']):
        plt.text(i + width / 2, v + 5, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], 'fp_changes_by_model_size.png'))

    return {
        'comparison': comparison_df,
        'improvement': improvement_df,
        'fp_changes': fp_change_df,
        'metrics': metrics
    }


# Main function to orchestrate the analysis
def main():
    """
    Main function to run the complete analysis pipeline
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Create output directories
    dirs = create_output_dirs()

    # Load the comparative model results dataset
    df = pd.read_csv(args.input)

    print(f"Loaded dataset with {len(df)} entries")

    # Compare model sizes
    comparison_results = compare_model_sizes(df, dirs)

    # Analyze false positives by tweet length for all model variants
    fp_length_ihra_7b = analyze_fp_by_length(df, 'IHRA_Binary_7B', dirs)
    fp_length_ihra_8b = analyze_fp_by_length(df, 'IHRA_Binary_8B', dirs)
    fp_length_jda_7b = analyze_fp_by_length(df, 'JDA_Binary_7B', dirs)
    fp_length_jda_8b = analyze_fp_by_length(df, 'JDA_Binary_8B', dirs)

    # Save false positive by length analysis
    fp_length_results = pd.concat([
        fp_length_ihra_7b.assign(Model='IHRA_7B'),
        fp_length_ihra_8b.assign(Model='IHRA_8B'),
        fp_length_jda_7b.assign(Model='JDA_7B'),
        fp_length_jda_8b.assign(Model='JDA_8B')
    ])
    fp_length_results.to_csv(os.path.join(dirs['results'], 'fp_by_length_analysis.csv'), index=False)

    # Create comparative length visualization
    plt.figure(figsize=(14, 8))

    plt.plot(fp_length_ihra_7b['Length_Bin'], fp_length_ihra_7b['False_Positive_Rate'],
             marker='o', label='IHRA 7B', color='royalblue')
    plt.plot(fp_length_ihra_8b['Length_Bin'], fp_length_ihra_8b['False_Positive_Rate'],
             marker='s', label='IHRA 8B', color='lightsteelblue', linestyle='--')
    plt.plot(fp_length_jda_7b['Length_Bin'], fp_length_jda_7b['False_Positive_Rate'],
             marker='^', label='JDA 7B', color='darkgreen')
    plt.plot(fp_length_jda_8b['Length_Bin'], fp_length_jda_8b['False_Positive_Rate'],
             marker='d', label='JDA 8B', color='lightgreen', linestyle='--')

    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate by Tweet Length - Model Size Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], 'fp_by_length_model_comparison.png'))

    # Analyze false positives by keyword for all model variants
    fp_keyword_ihra_7b = analyze_fp_by_keyword(df, 'IHRA_Binary_7B', dirs)
    fp_keyword_ihra_8b = analyze_fp_by_keyword(df, 'IHRA_Binary_8B', dirs)
    fp_keyword_jda_7b = analyze_fp_by_keyword(df, 'JDA_Binary_7B', dirs)
    fp_keyword_jda_8b = analyze_fp_by_keyword(df, 'JDA_Binary_8B', dirs)

    # Save false positive by keyword analysis
    fp_keyword_results = pd.concat([
        fp_keyword_ihra_7b.assign(Model='IHRA_7B'),
        fp_keyword_ihra_8b.assign(Model='IHRA_8B'),
        fp_keyword_jda_7b.assign(Model='JDA_7B'),
        fp_keyword_jda_8b.assign(Model='JDA_8B')
    ])
    fp_keyword_results.to_csv(os.path.join(dirs['results'], 'fp_by_keyword_analysis.csv'), index=False)

    # If topic analysis requested, perform additional analysis
    if args.fp_analysis:
        print("\nPerforming detailed false positive analysis with topic modeling...")

        # Perform topic modeling for each model variant
        topic_ihra_7b = analyze_fp_topics(df, 'IHRA_Binary_7B', args.topics, dirs)
        topic_ihra_8b = analyze_fp_topics(df, 'IHRA_Binary_8B', args.topics, dirs)
        topic_jda_7b = analyze_fp_topics(df, 'JDA_Binary_7B', args.topics, dirs)
        topic_jda_8b = analyze_fp_topics(df, 'JDA_Binary_8B', args.topics, dirs)

        # Compare topics between model sizes
        if topic_ihra_7b and topic_ihra_8b:
            # Extract false positive examples that were fixed in 8B
            fixed_fps = df[(df['IHRA_FP_7B'] == 1) & (df['IHRA_FP_8B'] == 0)]
            introduced_fps = df[(df['IHRA_FP_7B'] == 0) & (df['IHRA_FP_8B'] == 1)]

            # Save these examples for manual review
            fixed_fps.to_csv(os.path.join(dirs['results'], 'ihra_fps_fixed_in_8b.csv'), index=False)
            introduced_fps.to_csv(os.path.join(dirs['results'], 'ihra_fps_introduced_in_8b.csv'), index=False)

    print("\nAnalysis complete! The following files were created:")
    print("\nResults files:")
    print("  - results/model_size_comparison.csv - Core metrics comparison between model sizes")
    print("  - results/model_size_improvement.csv - Relative improvement from 7B to 8B model")
    print("  - results/fp_changes_by_model_size.csv - Analysis of false positives fixed vs introduced")
    print("  - results/fp_by_length_analysis.csv - False positive rates by tweet length")
    print("  - results/fp_by_keyword_analysis.csv - False positive rates by keyword")

    if args.fp_analysis:
        print("  - results/fp_topic_analysis_*.html - Detailed topic analysis of false positives")
        print("  - results/ihra_fps_fixed_in_8b.csv - Examples of false positives fixed in 8B model")
        print("  - results/ihra_fps_introduced_in_8b.csv - Examples of false positives introduced in 8B model")

    print("\nVisualization files:")
    print("  - plots/model_size_performance_comparison.png - Bar chart comparing key metrics")
    print("  - plots/model_size_fp_rate_comparison.png - Comparison of false positive rates")
    print("  - plots/fp_changes_by_model_size.png - Analysis of false positives fixed vs introduced")
    print("  - plots/fp_by_length_model_comparison.png - Line chart of FP rates by tweet length")
    print("  - plots/fp_analysis/ - Folder with detailed FP analysis visualizations")


if __name__ == "__main__":
    main()