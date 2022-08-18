from transformers import pipeline
import pickle

sentiment_analysis = pipeline("sentiment-analysis")
unmasker = pipeline("fill-mask")
question_answerer = pipeline("question-answering")
summarizer = pipeline("summarization")
en_fr_translator = pipeline("translation_en_to_fr")

pickle.dump(sentiment_analysis, open("model_weights/sent.pkl", 'wb'))
pickle.dump(unmasker, open("model_weights/unmask.pkl", 'wb'))
pickle.dump(question_answerer, open("model_weights/question_answerer.pkl", 'wb'))
pickle.dump(summarizer, open("model_weights/summarizer.pkl", 'wb'))
pickle.dump(en_fr_translator, open("model_weights/en_fr_translator.pkl", 'wb'))
