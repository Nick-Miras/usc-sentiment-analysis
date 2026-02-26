import pandas as pd
from transformers import pipeline, Pipeline, AutoTokenizer
from datetime import datetime
import os


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = 'data/raw_data.csv'
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


def retrieve_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)

def yield_data(dataframe: pd.DataFrame, column_name: str):
    for value in dataframe[column_name]:
        yield value

def get_model() -> Pipeline:
    print("Loading sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.truncation_side = 'left'
    return pipeline(
        task='text-classification',
        model=MODEL_NAME,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512
    )

if __name__ == "__main__":
    df = retrieve_data()
    model = get_model()
    sentiments = []
    sentiment_confidence = []

    print("Generating sentiments...")
    for out in model(yield_data(df, 'English_Translation')):
        sentiments.append(out['label'])
        sentiment_confidence.append(out['score'])

    df['sentiment'] = sentiments
    df['sentiment_confidence'] = sentiment_confidence
    os.makedirs(f'data/{MODEL_NAME}', exist_ok=True)
    df.to_csv(f'data/{MODEL_NAME}/sentiments_{timestamp}.csv', index=False)
    print(f"Sentiment analysis completed and saved to data/{MODEL_NAME}/sentiments_{timestamp}.csv")
