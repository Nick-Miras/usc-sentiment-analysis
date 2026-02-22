import pandas as pd
from transformers import pipeline, Pipeline
from datetime import datetime
import os


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = 'data/raw_data.csv'
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


def retrieve_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)

def yield_data(dataframe: pd.DataFrame, column_name: str):
    for _, row in dataframe.iterrows():
        yield row[column_name]

def get_model() -> Pipeline:
    print("Loading sentiment analysis model...")
    return pipeline(
        task='text-classification',
        model=MODEL_NAME
    )

if __name__ == "__main__":
    df = retrieve_data()
    model = get_model()
    sentiments = []
    sentiment_confidence = []

    print("Generating sentiments...")
    for out in model(yield_data(df, 'sentiment')):
        sentiments.append(out['label'])
        sentiment_confidence.append(out['score'])

    df['sentiment'] = sentiments
    df['sentiment_confidence'] = sentiment_confidence
    os.makedirs(f'data/{MODEL_NAME}', exist_ok=True)
    df.to_csv(f'data/{MODEL_NAME}/sentiments_{timestamp}.csv', index=False)
    print(f"Sentiment analysis completed and saved to data/sentiments_{timestamp}.csv")
