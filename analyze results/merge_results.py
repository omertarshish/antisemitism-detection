# combined_model_results.csv - Combines all results from both models in an extended format
# comparative_model_results.csv - Organized data for direct comparison between models
import pandas as pd
import numpy as np

def merge_model_results():
    """
    Merge results from two different model sizes (7B and 8B) and convert Yes/No labels to 1/0
    """
    # Load datasets
    df_7b = pd.read_csv('analysis_full_deepseek-r1_7b.csv')
    df_8b = pd.read_csv('analysis_full_deepseek-r1_8b.csv')
    
    # Add model identifier column
    df_7b['Model'] = '7B'
    df_8b['Model'] = '8B'
    
    # Convert Yes/No decisions to 1/0
    for df in [df_7b, df_8b]:
        df['IHRA_Binary'] = (df['IHRA_Decision'] == 'Yes').astype(int)
        df['JDA_Binary'] = (df['JDA_Decision'] == 'Yes').astype(int)
    
    # Combine the datasets
    combined_df = pd.concat([df_7b, df_8b], ignore_index=True)
    
    # Create a version with one row per tweet (useful for comparing models)
    pivot_df = df_7b[['TweetID', 'Username', 'CreateDate', 'Biased', 'Keyword', 'Text', 
                      'IHRA_Binary', 'JDA_Binary']].copy()
    
    # Add columns for 8B model results
    model_8b_mapping = dict(zip(df_8b['TweetID'], zip(df_8b['IHRA_Binary'], df_8b['JDA_Binary'])))
    
    pivot_df['IHRA_Binary_8B'] = pivot_df['TweetID'].map(lambda x: model_8b_mapping.get(x, (np.nan, np.nan))[0])
    pivot_df['JDA_Binary_8B'] = pivot_df['TweetID'].map(lambda x: model_8b_mapping.get(x, (np.nan, np.nan))[1])
    
    # Rename columns for clarity
    pivot_df.rename(columns={
        'IHRA_Binary': 'IHRA_Binary_7B',
        'JDA_Binary': 'JDA_Binary_7B'
    }, inplace=True)
    
    # Calculate text length for complexity analysis
    pivot_df['text_length'] = pivot_df['Text'].str.len()
    
    # Save the merged datasets
    combined_df.to_csv('combined_model_results.csv', index=False)
    pivot_df.to_csv('comparative_model_results.csv', index=False)
    
    print("Merged datasets created successfully:")
    print("1. combined_model_results.csv - All results stacked")
    print("2. comparative_model_results.csv - Results side-by-side for comparison")
    
    return combined_df, pivot_df

if __name__ == "__main__":
    merge_model_results()