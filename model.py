from transformers import pipeline
import pickle

sentiment_analysis = pipeline("sentiment-analysis")
pickle.dump(sentiment_analysis, open("model_weights/sent.pkl", 'wb'))