#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import pandas as pd
from src.data_loader import load_data
from src.sentiment_analysis import analyze_sentiment
from src.outcome_determination import determine_outcome
from src.call_type_classification import classify_call_type

class TestTranscriptAnalysis(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'filename': ['transcript_1.txt', 'transcript_2.txt'],
            'transcript': [
                "Member: Hi, I'm calling about my claim. Agent: I can help you with that. How may I assist you today?",
                "Member: I need to schedule an appointment. Agent: Certainly, I can help you with that. What type of appointment do you need?"
            ]
        })

    def test_load_data(self):
        data = load_data('tests/test_data.txt')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertTrue('filename' in data.columns)
        self.assertTrue('transcript' in data.columns)

    def test_analyze_sentiment(self):
        sentiments = self.sample_data['transcript'].apply(analyze_sentiment)
        self.assertEqual(len(sentiments), 2)
        self.assertTrue(all(sentiment in ['positive', 'negative', 'neutral'] for sentiment in sentiments))

    def test_determine_outcome(self):
        outcomes = self.sample_data['transcript'].apply(determine_outcome)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(outcome in ['issue resolved', 'follow-up action needed', 'unclear'] for outcome in outcomes))

    def test_classify_call_type(self):
        call_types = self.sample_data['transcript'].apply(classify_call_type)
        self.assertEqual(len(call_types), 2)
        self.assertTrue(all(call_type in ['general inquiry', 'appointment scheduling', 'pre-authorization request'] for call_type in call_types))

    def test_end_to_end(self):
        # Test the entire pipeline
        data = load_data('tests/test_data.txt')
        data['sentiment'] = data['transcript'].apply(analyze_sentiment)
        data['outcome'] = data['transcript'].apply(determine_outcome)
        data['call_type'] = data['transcript'].apply(classify_call_type)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue('sentiment' in data.columns)
        self.assertTrue('outcome' in data.columns)
        self.assertTrue('call_type' in data.columns)

if __name__ == '__main__':
    unittest.main()

