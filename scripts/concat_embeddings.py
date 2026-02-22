import pandas as pd
import numpy as np
import os
import glob

def main():
    print("=" * 70)
    print("CONCATENATE EMBEDDINGS AND RAW DATA")
    print("=" * 70)
    
    # Find the latest .npy file in data/
    npy_files = glob.glob("data/bert_embeddings_*.npy")
    if not npy_files:
        print("ERROR: No .npy embeddings file found in data/")
        return

    sentiments_files = glob.glob("data/sentiments_*.csv")
    latest_sentiment_file = max(sentiments_files, key=os.path.getmtime)
    
    # Sort by modification time to get the latest
    latest_npy = max(npy_files, key=os.path.getmtime)
    
    output_parquet = latest_npy.replace('.npy', '.parquet')

    print(f"Loading raw data from: {latest_sentiment_file}")
    try:
        df = pd.read_csv(latest_sentiment_file)
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"ERROR: File not found at {latest_sentiment_file}")
        return

    print(f"Loading embeddings from: {latest_npy}")
    try:
        embeddings = np.load(latest_npy)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
        return

    if len(df) != embeddings.shape[0]:
        print(f"ERROR: Row count mismatch! Dataframe has {len(df)} rows, but embeddings have {embeddings.shape[0]} rows.")
        return

    print("Creating embedding column...")
    
    # Combine original data with embeddings
    print("Concatenating dataframes...")
    df_with_embeddings = df.copy()
    df_with_embeddings['embedding'] = embeddings.tolist()
    
    # Drop raw columns to match transform_data.py
    raw_cols = ['Date', 'Data', 'English_Translation', 'Social_Media', 'code_switching', 'category_clean', 'sentiment_confidence']
    cols_to_drop = [col for col in raw_cols if col in df_with_embeddings.columns]
    df_with_embeddings = df_with_embeddings.drop(columns=cols_to_drop)
    
    print(f"Saving to parquet file: {output_parquet}")
    try:
        # Save to parquet
        df_with_embeddings.to_parquet(output_parquet, index=False, engine='pyarrow')
        print("Successfully saved to parquet!")
    except Exception as e:
        print(f"ERROR saving to parquet: {e}")
        print("Make sure you have pyarrow or fastparquet installed: pip install pyarrow")
        return

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df_with_embeddings)}")
    print(f"Total columns: {len(df_with_embeddings.columns)}")

    print("\nCategory Distribution:")
    for cat, count in df['category_clean'].value_counts().items():
        print(f"  {cat}: {count} ({count / len(df) * 100:.1f}%)")

    # Sentiment distribution
    if 'sentiment' in df.columns:
        print("\nSentiment Distribution:")
        for sent, count in df['sentiment'].value_counts().items():
            print(f"  {sent}: {count} ({count / len(df) * 100:.1f}%)")

    print(f"\nOutput file: {output_parquet}")
    print("=" * 70)

    # Display sample
    print("\nSample of data with embeddings:")
    sample_cols = ['sentiment', 'category', 'embedding']
    available_cols = [col for col in sample_cols if col in df_with_embeddings.columns]
    print(df_with_embeddings[available_cols].head())

if __name__ == "__main__":
    main()
