from flask import Flask, render_template, request
import pickle
model = pickle.load(open("model_weights/sent.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/sentiment_analysis')
def sent():
    return render_template('sentiment_analysis.html')
@app.route('/predict', methods=['POST'])
def home():
    data = request.form['input_text']
    pred = model(data)[0]
    text = f"label: {pred['label']}, with score: {round(pred['score'], 4)}"
    return render_template('after.html', data=text)

if __name__ == "__main__":
    app.run(debug=True)