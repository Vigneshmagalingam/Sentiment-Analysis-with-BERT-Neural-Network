
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import shutil
import os

# Load tokenizer and model


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')



# Function to scrape reviews from Yelp
def scrape_reviews(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class': regex})
        reviews = [result.text for result in results]
        return reviews
    except Exception as e:
        print(f"Error occurred during scraping: {e}")
        return []

# Function to process sentiment of reviews
def process_sentiment(reviews):
    try:
        sentiment_scores = []
        for review in reviews:
            tokens = tokenizer.encode(review[:512], return_tensors='pt', max_length=512, truncation=True)
            result = model(tokens)
            sentiment_scores.append(int(torch.argmax(result.logits)) + 1)
        return sentiment_scores
    except Exception as e:
        print(f"Error occurred during sentiment analysis: {e}")
        return []

# URL of Yelp page
url = 'https://www.yelp.com/biz/kothai-republic-san-francisco?osq=Restaurants'

# Scrape reviews
reviews = scrape_reviews(url)

# Process sentiment of reviews
if reviews:
    sentiment_scores = process_sentiment(reviews)
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiment_scores})
    print(df)
else:
    print("No reviews found.")

