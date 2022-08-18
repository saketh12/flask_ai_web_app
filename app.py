from multiprocessing import context
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/sentiment_analysis')
def sent():
    return render_template('sentiment_analysis.html')

@app.route('/mlm')
def mlm():
    return render_template('mlm.html')

@app.route('/summarizer')
def summarizer():
    return render_template('summarizer.html')

@app.route('/question_answering')
def question_answering():
    return render_template('question_answering.html')

@app.route('/translation')
def translation():
    return render_template('translation.html')

@app.route('/predict', methods=['POST'])
def home():
    text = "I love this movie!!"
    print(sentiment_analysis(text)[0])
    model_name = list(request.form.keys())[-1]
    if model_name == 'sent' or model_name == 'translator':
        data = request.form['input_text']
        print(data)
        if model_name == 'sent':
            pred = sentiment_analysis(data)[0]
            result = f"label: {pred['label']}, with score: {round(pred['score'], 4)}"
        if model_name == 'translator':
            result = translator(data)[0]['translation_text']
    if model_name == 'mlm':
        text1 = request.form['input_1_text']
        text2 = request.form['input_2_text']
        total_text = text1 + " <mask> " + text2
        result = mlm(total_text)[0]['sequence']
    if model_name == 'summarizer':
        print('entered')
        text_data = request.form['text_data']
        minLength = int(request.form['minLen'])
        maxLength = int(request.form['maxLen'])
        result = summarizer(text_data, min_length = minLength, max_length=maxLength)[0]['summary_text']
    if model_name == 'question_answerer':
        question_data = request.form['question']
        context_data = request.form['context']
        result = qa(question_data, context_data)['answer']
    return render_template('after.html', data = result)
if __name__ == "__main__":
    sentiment_analysis = pickle.load(open("model_weights/sent.pkl", 'rb'))
    mlm = pickle.load(open("model_weights/unmask.pkl", 'rb'))
    translator = pickle.load(open("model_weights/en_fr_translator.pkl", 'rb'))
    summarizer = pickle.load(open("model_weights/summarizer.pkl", 'rb'))
    qa = pickle.load(open("model_weights/question_answerer.pkl", 'rb'))
    app.run(debug=True)

# python3 -m flask run