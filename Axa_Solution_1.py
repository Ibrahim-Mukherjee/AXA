#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().system('pip install pandas numpy transformers scikit-learn matplotlib seaborn gensim wordcloud')


# In[4]:


import zipfile
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Set up sentiment analysis pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def extract_customer_transcript(transcript):
    customer_lines = re.findall(r'Member: (.*?)(?=\n|$)', transcript)
    return ' '.join(customer_lines)

def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens
    if result['label'] == 'POSITIVE' and result['score'] > 0.6:
        return 'positive'
    elif result['label'] == 'NEGATIVE' and result['score'] > 0.6:
        return 'negative'
    else:
        return 'neutral'

def determine_outcome(text):
    positive_indicators = ['thank you', 'great', 'sounds good', 'that\'s all', 'resolved', 'fixed', 'solved']
    negative_indicators = ['not resolved', 'still have a problem', 'unhappy', 'disappointed', 'frustrated', 'doesn\'t work', 'issue persists']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
    
    if positive_count > negative_count and 'thank you' in text_lower:
        return 'issue resolved'
    elif negative_count > 0 or 'follow up' in text_lower or 'call back' in text_lower:
        return 'follow-up action needed'
    else:
        return 'unclear'

def extract_call_duration(transcript):
    duration_match = re.search(r'The conversation ends after (\d+) minutes', transcript)
    return int(duration_match.group(1)) if duration_match else None

def extract_call_type(transcript):
    if 'pre-authorization' in transcript.lower():
        return 'pre-authorization request'
    elif 'schedule an appointment' in transcript.lower():
        return 'appointment scheduling'
    else:
        return 'general inquiry'

def analyze_transcript(transcript):
    customer_transcript = extract_customer_transcript(transcript)
    sentiment = analyze_sentiment(customer_transcript)
    outcome = determine_outcome(customer_transcript)
    duration = extract_call_duration(transcript)
    call_type = extract_call_type(transcript)
    return sentiment, outcome, duration, call_type

def process_transcripts(zip_file):
    results = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('.txt'):
                with z.open(filename) as f:
                    transcript = f.read().decode('utf-8')
                    sentiment, outcome, duration, call_type = analyze_transcript(transcript)
                    results.append({
                        'filename': filename,
                        'sentiment': sentiment,
                        'outcome': outcome,
                        'duration': duration,
                        'call_type': call_type
                    })
    return pd.DataFrame(results)

def main():
    zip_file = 'transcripts_v3.zip'
    df = process_transcripts(zip_file)
    
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nOutcome Distribution:")
    print(df['outcome'].value_counts(normalize=True))
    
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts(normalize=True))
    
    print("\nCall Type Distribution:")
    print(df['call_type'].value_counts(normalize=True))
    
    # Save the DataFrame to a CSV file
    df.to_csv('transcript_analysis_results.csv', index=False)
    print("\nResults saved to 'transcript_analysis_results.csv'")

if __name__ == "__main__":
    main()


# In[5]:


import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_class_distribution(df, column):
    distribution = df[column].value_counts(normalize=True)
    return distribution

def calculate_sentiment_outcome_correlation(df):
    contingency_table = pd.crosstab(df['sentiment'], df['outcome'])
    return contingency_table

def plot_confusion_matrix(df, x, y, title):
    cm = confusion_matrix(df[x], df[y], labels=sorted(df[x].unique()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(df[x].unique()), 
                yticklabels=sorted(df[y].unique()))
    plt.title(title)
    plt.xlabel(y)
    plt.ylabel(x)
    plt.savefig(f'{x}_{y}_confusion_matrix.png')
    plt.close()

def calculate_confidence_metrics(df):
    sentiment_scores = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    outcome_scores = df['outcome'].map({'issue resolved': 1, 'unclear': 0, 'follow-up action needed': -1})
    
    sentiment_confidence = np.abs(sentiment_scores)
    outcome_confidence = np.abs(outcome_scores)
    
    return {
        'avg_sentiment_confidence': sentiment_confidence.mean(),
        'avg_outcome_confidence': outcome_confidence.mean()
    }

def analyze_model_performance(df):
    print("\nModel Performance Metrics:")
    
    print("\n1. Class Distribution:")
    print("Sentiment Distribution:")
    print(calculate_class_distribution(df, 'sentiment'))
    print("\nOutcome Distribution:")
    print(calculate_class_distribution(df, 'outcome'))
    print("\nCall Type Distribution:")
    print(calculate_class_distribution(df, 'call_type'))
    
    print("\n2. Sentiment-Outcome Correlation:")
    print(calculate_sentiment_outcome_correlation(df))
    
    print("\n3. Confusion Matrices:")
    plot_confusion_matrix(df, 'sentiment', 'outcome', 'Sentiment vs Outcome')
    plot_confusion_matrix(df, 'call_type', 'outcome', 'Call Type vs Outcome')
    print("Confusion matrices saved as PNG files.")
    
    print("\n4. Confidence Metrics:")
    confidence_metrics = calculate_confidence_metrics(df)
    print(f"Average Sentiment Confidence: {confidence_metrics['avg_sentiment_confidence']:.2f}")
    print(f"Average Outcome Confidence: {confidence_metrics['avg_outcome_confidence']:.2f}")
    
    print("\n5. Potential Biases:")
    print("Outcome by Call Type:")
    print(df.groupby('call_type')['outcome'].value_counts(normalize=True).unstack())

def main():
    zip_file = 'transcripts_v3.zip'
    df = process_transcripts(zip_file)
    
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    
    analyze_model_performance(df)
    
    df.to_csv('transcript_analysis_results.csv', index=False)
    print("\nResults saved to 'transcript_analysis_results.csv'")

if __name__ == "__main__":
    main()


# In[ ]:




