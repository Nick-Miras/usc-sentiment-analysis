import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
import os
import pickle
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
warnings.filterwarnings('ignore')
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


def get_model():
    """Load Transformer Model with left truncation"""
    model = SentenceTransformer(model_name)
    model.tokenizer.truncation_side = 'left'
    return model


def get_tokenizer():
    """Load Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'
    return tokenizer


def save_checkpoint(embeddings, batch_idx, checkpoint_file):
    """Save progress checkpoint"""
    checkpoint = {
        'embeddings': embeddings,
        'last_batch': batch_idx
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at batch {batch_idx}")


def load_checkpoint(checkpoint_file):
    """Load progress checkpoint"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Resuming from batch {checkpoint['last_batch']}")
        return checkpoint['embeddings'], checkpoint['last_batch']
    return [], 0


def generate_embeddings(texts, model, batch_size=16, checkpoint_file='checkpoint.pkl'):
    """
    Generate embeddings with checkpoint saving
    """
    # Load checkpoint if exists
    embeddings, start_batch = load_checkpoint(checkpoint_file)

    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Process in batches
    for i in tqdm(range(start_batch, total_batches), desc="Generating Embeddings",
                  initial=start_batch, total=total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        try:
            # Generate embeddings
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)

            # Save checkpoint every 10 batches
            if (i + 1) % 10 == 0:
                save_checkpoint(embeddings, i + 1, checkpoint_file)

        except Exception as e:
            print(f"\nError at batch {i}: {e}")
            print("Saving checkpoint before exit...")
            save_checkpoint(embeddings, i, checkpoint_file)
            raise e

    # Delete checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed (completed successfully)")

    # Concatenate all batch embeddings
    return np.vstack(embeddings)


def main():
    print("=" * 70)

    # File paths
    input_file = "data/raw_data.csv"
    os.makedirs(f"data/{model_name}", exist_ok=True)
    output_embeddings_file = f"data/{model_name}/embeddings_{timestamp}.npy"
    checkpoint_file = f"data/{model_name}/checkpoint_{timestamp}.pkl"


    # Load data
    print(f"\nReading file: {input_file}")

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Use preprocessed text column
    text_column = 'English_Translation'

    if text_column not in df.columns:
        print(f"ERROR: '{text_column}' column not found!")
        print(f"Available columns: {list(df.columns)}")
        return

    # Handle missing values
    print(f"\nChecking for missing values in '{text_column}'...")
    missing_count = df[text_column].isna().sum()
    if missing_count > 0:
        print(f"WARNING: Found {missing_count} missing values. Filling with empty strings...")
        df[text_column] = df[text_column].fillna('')

    # Convert to list
    texts = df[text_column].tolist()
    print(f"\nProcessing {len(texts)} text samples")
    print("\nLoading Sentence Transformer model...")
    print(f"   Using: {model_name}")

    try:
        model = get_model()
        print(f"Model loaded successfully")

    except Exception as e:
        print(f"ERROR loading Sentence Transformer model: {e}")
        print("\nPlease ensure you have sentence-transformers installed:")
        print("  pip install sentence-transformers")
        return

    # Generate embeddings
    print(f"\nGenerating embeddings...")
    print(f"   Batch size: 16")
    print(f"   Checkpoints: Every 10 batches")

    try:
        embeddings = generate_embeddings(
            texts=texts,
            model=model,
            batch_size=16,
            checkpoint_file=checkpoint_file
        )

        print(f"\nEmbeddings generated successfully!")
        print(f"   Shape: {embeddings.shape}")

    except Exception as e:
        print(f"\nERROR generating embeddings: {e}")
        print(f"\nTo resume: Just run the script again!")
        print(f"   Progress saved in: {checkpoint_file}")
        return

    # Save embeddings as numpy array
    print(f"\nSaving embeddings to: {output_embeddings_file}")
    np.save(output_embeddings_file, embeddings)
    print(f"Embeddings saved as .npy file")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {len(texts)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Original columns: {len(df.columns)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        import traceback

        traceback.print_exc()
        print("=" * 70)