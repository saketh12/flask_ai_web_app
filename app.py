from flask import Flask, render_template, request
import pickle
model = pickle.load(open("model_weights/sent.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['input_text']
    print("data 1", data1)
    pred = model(data1)[0]
    text = f"label: {pred['label']}, with score: {round(pred['score'], 4)}"
    return render_template('after.html', data=text)

if __name__ == "__main__":
    app.run(debug=True)