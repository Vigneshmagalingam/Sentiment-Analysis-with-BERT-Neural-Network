from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np


def scrape_reviews(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*wiI7pd.*')
        results = soup.find_all('p', {'class': regex})
        reviews = [result.text for result in results]
        return reviews
    except Exception as e:
        print(f"Error occurred during scraping: {e}")
        return []

# Revised sentiment analysis function
def process_sentiment(reviews):
    sentiment_scores = []
    for review in reviews:
        try:
            tokens = tokenizer.encode(review, return_tensors='pt', max_length=512)
            with torch.no_grad():
                result = model(tokens)
            logits = result[0].detach().numpy()  # Unpack the tuple and access logits
            sentiment_score = np.argmax(logits)
            sentiment_scores.append(sentiment_score)
        except Exception as e:
            print(f"Error occurred during sentiment analysis for review: {e}")
            sentiment_scores.append(None)
    return sentiment_scores

# Load a sentiment analysis model that's fine-tuned on a sentiment analysis task
model_name = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


url = 'https://www.google.com/maps/place/Abhiruchi+Multicuisine+Family+Restaurant/@12.9112709,79.0900822,13z/data=!4m10!1m2!2m1!1srestaurants+near+me!3m6!1s0x3bad476f4bf7eb77:0xb5e9ecd5e1d01cfd!8m2!3d12.9584742!4d79.138354!15sChNyZXN0YXVyYW50cyBuZWFyIG1lIgOQAQFaFSITcmVzdGF1cmFudHMgbmVhciBtZZIBCnJlc3RhdXJhbnTgAQA!16s%2Fg%2F1td5vlg9?entry=ttu'

# Scrape reviews
reviews = scrape_reviews(url)

# Process sentiment of reviews
if reviews:
    sentiment_scores = process_sentiment(reviews)
    
    # Create DataFrame
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiment_scores})
    
    # Drop rows with missing sentiment scores
    df.dropna(subset=['sentiment'], inplace=True)
    
    # Print summary of DataFrame
    print(df.head())
else:
    print("No reviews found.")
